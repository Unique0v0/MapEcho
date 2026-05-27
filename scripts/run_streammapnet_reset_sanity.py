#!/usr/bin/env python3
import argparse
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


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def tensor_finite(obj):
    if torch.is_tensor(obj):
        return bool(torch.isfinite(obj).all().item())
    if isinstance(obj, dict):
        return all(tensor_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(tensor_finite(v) for v in obj)
    return True


def summarize_debug(out_dir, ann_path, reset_mode, reset_offset):
    samples = load_pickle(ann_path)
    sample_by_idx = {int(sample["sample_idx"]): sample for sample in samples}
    reset_frame = next(
        sample for sample in samples if sample["mapecho_frame_offset"] == reset_offset + 1
    )
    before_frame = next(
        sample for sample in samples if sample["mapecho_frame_offset"] == reset_offset
    )

    query_files = sorted(Path(out_dir, "query_memory").glob("scene-*/*.pt"))
    bev_files = sorted(Path(out_dir, "bev_memory").glob("scene-*/*.pt"))

    def load_frame(files, sample):
        target = str(sample["sample_idx"])
        matches = [path for path in files if path.stem == target]
        if not matches:
            raise FileNotFoundError(f"missing debug dump for sample_idx={target}")
        return torch.load(matches[0], map_location="cpu")

    q_before = load_frame(query_files, before_frame)
    q_after = load_frame(query_files, reset_frame)
    b_before = load_frame(bev_files, before_frame)
    b_after = load_frame(bev_files, reset_frame)

    def prop_sum(payload):
        mask = payload.get("propagated_query_mask")
        return None if mask is None else int(mask.sum().item())

    query_finite = all(tensor_finite(torch.load(path, map_location="cpu")) for path in query_files)
    bev_finite = all(tensor_finite(torch.load(path, map_location="cpu")) for path in bev_files)

    expected = {
        "all": {"prop_mask_after_reset": 0, "warped_bev_after_reset": False},
        "query": {"prop_mask_after_reset": 0, "warped_bev_after_reset": True},
        "bev": {"prop_mask_after_reset": 33, "warped_bev_after_reset": False},
    }[reset_mode]

    prop_after = prop_sum(q_after)
    warped_after = "warped_history_bev_norm" in b_after
    summary = {
        "mode": reset_mode,
        "ann_file": str(ann_path),
        "out_dir": str(out_dir),
        "num_frames": len(samples),
        "query_dump_count": len(query_files),
        "bev_dump_count": len(bev_files),
        "reset_after_offset": reset_offset,
        "reset_before_offset": reset_offset + 1,
        "before_reset_sample_idx": before_frame["sample_idx"],
        "after_reset_sample_idx": reset_frame["sample_idx"],
        "prop_mask_sum_before_reset": prop_sum(q_before),
        "prop_mask_sum_after_reset": prop_after,
        "query_is_first_after_reset": bool(q_after.get("is_first_frame")),
        "bev_is_first_after_reset": bool(b_after.get("is_first_frame")),
        "warped_bev_before_reset": "warped_history_bev_norm" in b_before,
        "warped_bev_after_reset": warped_after,
        "query_finite": query_finite,
        "bev_finite": bev_finite,
        "expected": expected,
        "pass": (
            len(query_files) == len(samples)
            and len(bev_files) == len(samples)
            and query_finite
            and bev_finite
            and prop_after == expected["prop_mask_after_reset"]
            and warped_after == expected["warped_bev_after_reset"]
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run StreamMapNet reset sanity on one clean sequence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--mode", choices=["all", "query", "bev"], required=True)
    parser.add_argument("--reset-after-offset", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.debug_cfg = dict(
        query_memory=dict(
            enabled=True,
            out_dir=str(Path(args.out_root, f"reset_{args.mode}", "query_memory")),
        ),
        bev_memory=dict(
            enabled=True,
            out_dir=str(Path(args.out_root, f"reset_{args.mode}", "bev_memory")),
            save_full=False,
        ),
    )
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = 0
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    import_plugins(cfg)
    dataset = build_dataset(cfg.data.test)
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

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        sample = dataset.samples[i]
        if sample.get("mapecho_frame_offset") == args.reset_after_offset + 1:
            model.module.reset_temporal_state(args.mode)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        prog_bar.update()

    out_dir = Path(args.out_root, f"reset_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mmcv.dump(results, out_dir / "outputs.pkl")
    summary = summarize_debug(out_dir, args.ann_file, args.mode, args.reset_after_offset)
    summary_path = out_dir / "reset_sanity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
