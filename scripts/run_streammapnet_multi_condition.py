#!/usr/bin/env python3
import argparse
import gc
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


CONDITIONS = {
    "clean_keep": ("clean", "none", "phase1_0_clean_keep"),
    "clean_reset_all": ("clean", "all", "phase1_0_reset_sanity/reset_all"),
    "clean_reset_query": ("clean", "query", "phase1_0_reset_sanity/reset_query"),
    "clean_reset_bev": ("clean", "bev", "phase1_0_reset_sanity/reset_bev"),
    "attack_keep": ("attack", "none", "phase1_0_attack_reset_ablation/attack_keep"),
    "attack_reset_all": ("attack", "all", "phase1_0_attack_reset_ablation/attack_reset_all"),
    "attack_reset_query": ("attack", "query", "phase1_0_attack_reset_ablation/attack_reset_query"),
    "attack_reset_bev": ("attack", "bev", "phase1_0_attack_reset_ablation/attack_reset_bev"),
}


def import_plugins(cfg):
    sys.path.append(os.path.abspath("."))
    if not getattr(cfg, "plugin", False):
        return
    plugin_dirs = cfg.plugin_dir
    if not isinstance(plugin_dirs, list):
        plugin_dirs = [plugin_dirs]
    for plugin_dir in plugin_dirs:
        module_dir = os.path.dirname(plugin_dir)
        module_path = ".".join(module_dir.split("/"))
        print(module_path)
        importlib.import_module(module_path)


def is_readable_pickle(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as f:
            pickle.load(f)
        return True
    except (EOFError, pickle.UnpicklingError, OSError):
        return False


def build_model_once(args):
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.debug_cfg = dict(
        query_memory=dict(enabled=True, out_dir=""),
        bev_memory=dict(enabled=True, out_dir="", save_full=False),
    )
    cfg.data.workers_per_gpu = 0
    import_plugins(cfg)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return cfg, model


def make_data_loader(cfg, ann_file):
    test_cfg = cfg.data.test.copy()
    test_cfg.ann_file = str(ann_file)
    test_cfg.test_mode = True
    samples_per_gpu = test_cfg.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        test_cfg.pipeline = replace_ImageToTensor(test_cfg.pipeline)
    dataset = build_dataset(test_cfg)
    from plugin.datasets.builder import build_dataloader

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return dataset, data_loader


def set_debug_cfg(model, out_dir):
    debug_cfg = dict(
        query_memory=dict(enabled=True, out_dir=str(out_dir / "query_memory")),
        bev_memory=dict(enabled=True, out_dir=str(out_dir / "bev_memory"), save_full=False),
    )
    if hasattr(model.module, "set_debug_cfg"):
        model.module.set_debug_cfg(debug_cfg)
    else:
        model.module.debug_cfg = debug_cfg


def run_condition(args, cfg, model, ann_file, out_dir, condition, reset_mode):
    out_dir = Path(out_dir)
    outputs_path = out_dir / "outputs.pkl"
    if args.skip_completed and is_readable_pickle(outputs_path):
        print(f"[MapEcho] skipping completed condition {condition}: {out_dir}", flush=True)
        return
    if outputs_path.exists() and args.skip_completed:
        print(f"[MapEcho] removing incomplete pickle: {outputs_path}", flush=True)
        outputs_path.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    set_debug_cfg(model, out_dir)
    if hasattr(model.module, "reset_temporal_state"):
        model.module.reset_temporal_state("all")

    dataset, data_loader = make_data_loader(cfg, ann_file)
    num_frames = len(dataset)
    results = []
    reset_applied = False
    prog_bar = mmcv.ProgressBar(num_frames)
    try:
        for i, data in enumerate(data_loader):
            sample = dataset.samples[i]
            if reset_mode != "none" and sample.get("mapecho_frame_offset") == args.reset_after_offset + 1:
                model.module.reset_temporal_state(reset_mode)
                reset_applied = True
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            results.extend(result)
            prog_bar.update()
        mmcv.dump(results, outputs_path)
        dataset.format_results(results, prefix=str(out_dir))
    finally:
        del data_loader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "condition": condition,
        "ann_file": str(ann_file),
        "out_dir": str(out_dir),
        "num_frames": num_frames,
        "reset_mode": reset_mode,
        "reset_after_offset": args.reset_after_offset,
        "reset_applied": reset_applied,
        "query_dump_count": len(list((out_dir / "query_memory").glob("scene-*/*.pt"))),
        "bev_dump_count": len(list((out_dir / "bev_memory").glob("scene-*/*.pt"))),
        "submission_path": str(out_dir / "submission_vector.json"),
        "model_loaded_once_for_multi_condition": True,
    }
    (out_dir / "condition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple StreamMapNet sequence conditions with one checkpoint load."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean-ann", required=True)
    parser.add_argument("--attack-ann", required=True)
    parser.add_argument("--token-root", required=True)
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS.keys()),
        help="Comma-separated condition names.",
    )
    parser.add_argument("--reset-after-offset", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true", default=False)
    args = parser.parse_args()

    cfg, model = build_model_once(args)
    token_root = Path(args.token_root)
    selected = [item.strip() for item in args.conditions.split(",") if item.strip()]
    for condition in selected:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition}. Valid: {sorted(CONDITIONS)}")
        ann_kind, reset_mode, rel_out = CONDITIONS[condition]
        ann_file = args.clean_ann if ann_kind == "clean" else args.attack_ann
        print(f"[MapEcho] running {condition}", flush=True)
        run_condition(
            args,
            cfg,
            model,
            ann_file,
            token_root / rel_out,
            condition,
            reset_mode,
        )


if __name__ == "__main__":
    main()
