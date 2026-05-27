#!/usr/bin/env python3
import argparse
import importlib
import json
import os
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


def main():
    parser = argparse.ArgumentParser(
        description="Run one StreamMapNet sequence condition with optional reset before t+1."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--reset-mode", choices=["none", "all", "query", "bev"], default="none")
    parser.add_argument("--reset-after-offset", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.debug_cfg = dict(
        query_memory=dict(
            enabled=True,
            out_dir=str(out_dir / "query_memory"),
        ),
        bev_memory=dict(
            enabled=True,
            out_dir=str(out_dir / "bev_memory"),
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
    reset_applied = False
    for i, data in enumerate(data_loader):
        sample = dataset.samples[i]
        if (
            args.reset_mode != "none"
            and sample.get("mapecho_frame_offset") == args.reset_after_offset + 1
        ):
            model.module.reset_temporal_state(args.reset_mode)
            reset_applied = True
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        prog_bar.update()

    mmcv.dump(results, out_dir / "outputs.pkl")
    dataset.format_results(results, prefix=str(out_dir))
    summary = {
        "condition": args.condition,
        "ann_file": args.ann_file,
        "out_dir": str(out_dir),
        "num_frames": len(dataset),
        "reset_mode": args.reset_mode,
        "reset_after_offset": args.reset_after_offset,
        "reset_applied": reset_applied,
        "query_dump_count": len(list((out_dir / "query_memory").glob("scene-*/*.pt"))),
        "bev_dump_count": len(list((out_dir / "bev_memory").glob("scene-*/*.pt"))),
        "submission_path": str(out_dir / "submission_vector.json"),
    }
    (out_dir / "condition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
