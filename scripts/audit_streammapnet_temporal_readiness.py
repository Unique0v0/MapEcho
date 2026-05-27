#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import torch
from mmcv import Config


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def count_matching(keys, patterns):
    return {pattern: sum(pattern in key for key in keys) for pattern in patterns}


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether the StreamMapNet config/checkpoint can support temporal hook sanity."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--phase1-tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=19)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model
    head_streaming_cfg = model_cfg.get("head_cfg", {}).get("streaming_cfg", {})
    bev_streaming_cfg = model_cfg.get("streaming_cfg", {})

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    keys = list(state_dict.keys())
    key_counts = count_matching(
        keys,
        [
            "stream_fusion_neck",
            "head.query_update",
            "head.transformer.decoder.prop_add_stage",
            "query_memory",
            "reference_points_memory",
            "bev_memory",
        ],
    )

    stream_samples = load_pickle(args.stream_ann)
    sample_by_token = {sample["token"]: sample for sample in stream_samples}
    scene_samples = {}
    for sample in stream_samples:
        scene_samples.setdefault(sample["scene_name"], []).append(sample)
    for samples in scene_samples.values():
        samples.sort(key=lambda sample: sample["sample_idx"])

    phase1_tokens = load_tokens(args.phase1_tokens)
    sequence_checks = []
    for token in phase1_tokens:
        if token not in sample_by_token:
            sequence_checks.append(
                {
                    "sample_token": token,
                    "scene_name": None,
                    "scene_pos": None,
                    "scene_len": None,
                    "present_in_ann": False,
                    "has_warmup": False,
                    "has_recovery": False,
                    "sequence_len_W_L_t": args.warmup + 1 + args.recovery,
                }
            )
            continue
        sample = sample_by_token[token]
        samples = scene_samples[sample["scene_name"]]
        scene_pos = next(i for i, item in enumerate(samples) if item["token"] == token)
        sequence_checks.append(
            {
                "sample_token": token,
                "scene_name": sample["scene_name"],
                "scene_pos": scene_pos,
                "scene_len": len(samples),
                "present_in_ann": True,
                "has_warmup": scene_pos >= args.warmup,
                "has_recovery": scene_pos + args.recovery < len(samples),
                "sequence_len_W_L_t": args.warmup + 1 + args.recovery,
            }
        )

    config_streaming_query = bool(head_streaming_cfg.get("streaming", False))
    config_streaming_bev = bool(bev_streaming_cfg.get("streaming_bev", False))
    checkpoint_has_query_temporal = key_counts["head.query_update"] > 0
    checkpoint_has_bev_temporal = key_counts["stream_fusion_neck"] > 0

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config_streaming_query": config_streaming_query,
        "config_streaming_bev": config_streaming_bev,
        "head_streaming_cfg": dict(head_streaming_cfg),
        "bev_streaming_cfg": dict(bev_streaming_cfg),
        "checkpoint_key_counts": key_counts,
        "checkpoint_has_query_temporal": checkpoint_has_query_temporal,
        "checkpoint_has_bev_temporal": checkpoint_has_bev_temporal,
        "phase1_tokens": len(phase1_tokens),
        "phase1_tokens_present_in_ann": sum(row["present_in_ann"] for row in sequence_checks),
        "phase1_tokens_missing_from_ann": sum(not row["present_in_ann"] for row in sequence_checks),
        "phase1_temporal_sequences_ok": all(
            row["present_in_ann"] and row["has_warmup"] and row["has_recovery"]
            for row in sequence_checks
        ),
        "can_run_clean_hook_sanity": (
            config_streaming_query
            and config_streaming_bev
            and checkpoint_has_query_temporal
            and checkpoint_has_bev_temporal
        ),
        "blocking_reason": "",
        "sequence_checks_preview": sequence_checks[:5],
    }
    if not summary["can_run_clean_hook_sanity"]:
        reasons = []
        if not config_streaming_query:
            reasons.append("config does not enable streaming query")
        if not config_streaming_bev:
            reasons.append("config does not enable streaming BEV")
        if not checkpoint_has_query_temporal:
            reasons.append("checkpoint has no query temporal weights")
        if not checkpoint_has_bev_temporal:
            reasons.append("checkpoint has no BEV temporal weights")
        summary["blocking_reason"] = "; ".join(reasons)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
