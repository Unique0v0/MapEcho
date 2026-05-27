#!/usr/bin/env python3
import argparse
import json
import math
import pickle
from pathlib import Path

import torch


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_frame(root, category, scene_name, sample_idx):
    path = Path(root, category, scene_name, f"{sample_idx}.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def tensor_delta(a, b):
    delta = (a.float() - b.float()).abs()
    return {
        "mean_abs": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
    }


def scalar_delta(a, b):
    return float(abs(float(a) - float(b)))


def finite_payload(payload):
    if torch.is_tensor(payload):
        return bool(torch.isfinite(payload).all().item())
    if isinstance(payload, dict):
        return all(finite_payload(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(finite_payload(value) for value in payload)
    return True


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 1.0 clean vs attack_keep dry run.")
    parser.add_argument(
        "--clean-root",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep",
    )
    parser.add_argument(
        "--attack-root",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep",
    )
    parser.add_argument(
        "--attack-ann",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep_one_sequence_ann.pkl",
    )
    parser.add_argument(
        "--attack-summary",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep/attack_at_t_ann_summary.json",
    )
    parser.add_argument(
        "--out",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep/attack_dry_run_summary.json",
    )
    args = parser.parse_args()

    samples = load_pickle(args.attack_ann)
    attack_summary = json.loads(Path(args.attack_summary).read_text())
    target = next(sample for sample in samples if sample["mapecho_frame_offset"] == 0)
    post = next(sample for sample in samples if sample["mapecho_frame_offset"] == 1)
    scene_name = target["scene_name"]

    counts = {}
    for label, root in [("clean", args.clean_root), ("attack", args.attack_root)]:
        counts[f"{label}_query_dumps"] = len(list(Path(root, "query_memory").glob("scene-*/*.pt")))
        counts[f"{label}_bev_dumps"] = len(list(Path(root, "bev_memory").glob("scene-*/*.pt")))

    rows = {}
    for frame_name, sample in [("t", target), ("t_plus_1", post)]:
        clean_q = load_frame(args.clean_root, "query_memory", scene_name, sample["sample_idx"])
        attack_q = load_frame(args.attack_root, "query_memory", scene_name, sample["sample_idx"])
        clean_b = load_frame(args.clean_root, "bev_memory", scene_name, sample["sample_idx"])
        attack_b = load_frame(args.attack_root, "bev_memory", scene_name, sample["sample_idx"])
        rows[frame_name] = {
            "sample_idx": sample["sample_idx"],
            "token": sample["token"],
            "frame_offset": sample["mapecho_frame_offset"],
            "query_score_delta": tensor_delta(
                clean_q["all_query_scores_raw_logit"],
                attack_q["all_query_scores_raw_logit"],
            ),
            "query_pred_vector_delta": tensor_delta(
                clean_q["all_query_pred_vectors"],
                attack_q["all_query_pred_vectors"],
            ),
            "topk_embedding_delta": tensor_delta(
                clean_q["topk_query_embedding"],
                attack_q["topk_query_embedding"],
            ),
            "prop_mask_clean_sum": int(clean_q["propagated_query_mask"].sum().item()),
            "prop_mask_attack_sum": int(attack_q["propagated_query_mask"].sum().item()),
            "current_bev_norm_delta": scalar_delta(
                clean_b["current_bev_norm"],
                attack_b["current_bev_norm"],
            ),
            "fused_bev_norm_delta": scalar_delta(
                clean_b["fused_bev_norm"],
                attack_b["fused_bev_norm"],
            ),
            "warped_clean": "warped_history_bev_norm" in clean_b,
            "warped_attack": "warped_history_bev_norm" in attack_b,
            "finite": all(
                finite_payload(payload)
                for payload in [clean_q, attack_q, clean_b, attack_b]
            ),
        }

    summary = {
        "attack_summary": attack_summary,
        "counts": counts,
        "frames": rows,
        "pass_dump_counts": counts == {
            "clean_query_dumps": 30,
            "clean_bev_dumps": 30,
            "attack_query_dumps": 30,
            "attack_bev_dumps": 30,
        },
        "pass_n_attack_1": bool(attack_summary.get("pass_n_attack_1")),
        "pass_attack_changes_target_query": rows["t"]["query_score_delta"]["max_abs"] > 0,
        "pass_attack_changes_target_bev": rows["t"]["current_bev_norm_delta"] > 0,
        "pass_recovery_input_clean": post.get("mapecho_attack_schedule") == "clean",
        "pass_finite": rows["t"]["finite"] and rows["t_plus_1"]["finite"],
    }
    summary["pass"] = all(
        summary[key]
        for key in [
            "pass_dump_counts",
            "pass_n_attack_1",
            "pass_attack_changes_target_query",
            "pass_attack_changes_target_bev",
            "pass_recovery_input_clean",
            "pass_finite",
        ]
    )
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
