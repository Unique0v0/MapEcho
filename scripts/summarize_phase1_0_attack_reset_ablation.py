#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path

import torch


METRICS = [
    "query_score_mean_abs",
    "query_score_max_abs",
    "pred_vector_mean_abs",
    "pred_vector_max_abs",
    "topk_embedding_mean_abs",
    "topk_embedding_max_abs",
    "current_bev_norm_delta",
    "fused_bev_norm_delta",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_frame(root, category, scene_name, sample_idx):
    path = Path(root, category, scene_name, f"{sample_idx}.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def abs_delta(a, b):
    delta = (a.float() - b.float()).abs()
    return float(delta.mean().item()), float(delta.max().item()), float(torch.linalg.vector_norm(delta).item())


def safe_reduction(keep_value, reset_value):
    if abs(keep_value) < 1e-12:
        return None
    return float(1.0 - reset_value / keep_value)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize single-sequence attack_keep vs reset ablation against clean_keep."
    )
    parser.add_argument(
        "--clean-root",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep",
    )
    parser.add_argument(
        "--ablation-root",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_reset_ablation",
    )
    parser.add_argument(
        "--clean-reset-root",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_reset_sanity",
    )
    parser.add_argument(
        "--ann",
        default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep_one_sequence_ann.pkl",
    )
    parser.add_argument("--offsets", default="0,1,2,3")
    args = parser.parse_args()

    samples = load_pickle(args.ann)
    by_offset = {int(sample["mapecho_frame_offset"]): sample for sample in samples}
    offsets = [int(value.strip()) for value in args.offsets.split(",") if value.strip()]
    scene_name = samples[0]["scene_name"]
    conditions = [
        "attack_keep",
        "attack_reset_all",
        "attack_reset_query",
        "attack_reset_bev",
    ]

    rows = []
    matched_rows = []
    for condition in conditions:
        root = Path(args.ablation_root, condition)
        for offset in offsets:
            sample = by_offset[offset]
            clean_q = load_frame(args.clean_root, "query_memory", scene_name, sample["sample_idx"])
            cond_q = load_frame(root, "query_memory", scene_name, sample["sample_idx"])
            clean_b = load_frame(args.clean_root, "bev_memory", scene_name, sample["sample_idx"])
            cond_b = load_frame(root, "bev_memory", scene_name, sample["sample_idx"])

            score_mean, score_max, score_l2 = abs_delta(
                clean_q["all_query_scores_raw_logit"],
                cond_q["all_query_scores_raw_logit"],
            )
            pred_mean, pred_max, pred_l2 = abs_delta(
                clean_q["all_query_pred_vectors"],
                cond_q["all_query_pred_vectors"],
            )
            emb_mean, emb_max, emb_l2 = abs_delta(
                clean_q["topk_query_embedding"],
                cond_q["topk_query_embedding"],
            )
            row = {
                "condition": condition,
                "frame_offset": offset,
                "sample_idx": sample["sample_idx"],
                "token": sample["token"],
                "schedule": sample.get("mapecho_attack_schedule", "clean"),
                "query_score_mean_abs": score_mean,
                "query_score_max_abs": score_max,
                "query_score_l2": score_l2,
                "pred_vector_mean_abs": pred_mean,
                "pred_vector_max_abs": pred_max,
                "pred_vector_l2": pred_l2,
                "topk_embedding_mean_abs": emb_mean,
                "topk_embedding_max_abs": emb_max,
                "topk_embedding_l2": emb_l2,
                "current_bev_norm_delta": abs(float(clean_b["current_bev_norm"]) - float(cond_b["current_bev_norm"])),
                "fused_bev_norm_delta": abs(float(clean_b["fused_bev_norm"]) - float(cond_b["fused_bev_norm"])),
                "prop_mask_sum": int(cond_q["propagated_query_mask"].sum().item()),
                "warped_bev_present": "warped_history_bev_norm" in cond_b,
                "query_is_first": bool(cond_q.get("is_first_frame")),
                "bev_is_first": bool(cond_b.get("is_first_frame")),
            }
            rows.append(row)

    matched_pairs = {
        "attack_keep": Path(args.clean_root),
        "attack_reset_all": Path(args.clean_reset_root, "reset_all"),
        "attack_reset_query": Path(args.clean_reset_root, "reset_query"),
        "attack_reset_bev": Path(args.clean_reset_root, "reset_bev"),
    }
    for condition in conditions:
        root = Path(args.ablation_root, condition)
        baseline_root = matched_pairs[condition]
        baseline_name = "clean_keep" if condition == "attack_keep" else condition.replace("attack_", "clean_")
        for offset in offsets:
            sample = by_offset[offset]
            base_q = load_frame(baseline_root, "query_memory", scene_name, sample["sample_idx"])
            cond_q = load_frame(root, "query_memory", scene_name, sample["sample_idx"])
            base_b = load_frame(baseline_root, "bev_memory", scene_name, sample["sample_idx"])
            cond_b = load_frame(root, "bev_memory", scene_name, sample["sample_idx"])

            score_mean, score_max, score_l2 = abs_delta(
                base_q["all_query_scores_raw_logit"],
                cond_q["all_query_scores_raw_logit"],
            )
            pred_mean, pred_max, pred_l2 = abs_delta(
                base_q["all_query_pred_vectors"],
                cond_q["all_query_pred_vectors"],
            )
            emb_mean, emb_max, emb_l2 = abs_delta(
                base_q["topk_query_embedding"],
                cond_q["topk_query_embedding"],
            )
            matched_rows.append(
                {
                    "condition": condition,
                    "matched_baseline": baseline_name,
                    "frame_offset": offset,
                    "sample_idx": sample["sample_idx"],
                    "token": sample["token"],
                    "schedule": sample.get("mapecho_attack_schedule", "clean"),
                    "query_score_mean_abs": score_mean,
                    "query_score_max_abs": score_max,
                    "query_score_l2": score_l2,
                    "pred_vector_mean_abs": pred_mean,
                    "pred_vector_max_abs": pred_max,
                    "pred_vector_l2": pred_l2,
                    "topk_embedding_mean_abs": emb_mean,
                    "topk_embedding_max_abs": emb_max,
                    "topk_embedding_l2": emb_l2,
                    "current_bev_norm_delta": abs(float(base_b["current_bev_norm"]) - float(cond_b["current_bev_norm"])),
                    "fused_bev_norm_delta": abs(float(base_b["fused_bev_norm"]) - float(cond_b["fused_bev_norm"])),
                    "prop_mask_sum": int(cond_q["propagated_query_mask"].sum().item()),
                    "baseline_prop_mask_sum": int(base_q["propagated_query_mask"].sum().item()),
                    "warped_bev_present": "warped_history_bev_norm" in cond_b,
                    "baseline_warped_bev_present": "warped_history_bev_norm" in base_b,
                    "query_is_first": bool(cond_q.get("is_first_frame")),
                    "baseline_query_is_first": bool(base_q.get("is_first_frame")),
                    "bev_is_first": bool(cond_b.get("is_first_frame")),
                    "baseline_bev_is_first": bool(base_b.get("is_first_frame")),
                }
            )

    row_by_condition_offset = {
        (row["condition"], row["frame_offset"]): row for row in rows
    }
    reductions = []
    for offset in [1, 2, 3]:
        keep = row_by_condition_offset[("attack_keep", offset)]
        for condition in ["attack_reset_all", "attack_reset_query", "attack_reset_bev"]:
            reset = row_by_condition_offset[(condition, offset)]
            reduction_row = {
                "condition": condition,
                "frame_offset": offset,
            }
            for metric in METRICS:
                reduction_row[f"{metric}_reduction"] = safe_reduction(
                    float(keep[metric]),
                    float(reset[metric]),
                )
            reductions.append(reduction_row)

    out_root = Path(args.ablation_root)
    csv_path = out_root / "phase1_0_single_sequence_reset_ablation_summary.csv"
    write_csv(csv_path, rows, list(rows[0].keys()))
    reductions_csv = out_root / "phase1_0_single_sequence_reset_ablation_reductions.csv"
    write_csv(reductions_csv, reductions, list(reductions[0].keys()))
    matched_csv = out_root / "phase1_0_single_sequence_reset_ablation_matched_baseline.csv"
    write_csv(matched_csv, matched_rows, list(matched_rows[0].keys()))

    matched_by_condition_offset = {
        (row["condition"], row["frame_offset"]): row for row in matched_rows
    }
    matched_reductions = []
    for offset in [1, 2, 3]:
        keep = matched_by_condition_offset[("attack_keep", offset)]
        for condition in ["attack_reset_all", "attack_reset_query", "attack_reset_bev"]:
            reset = matched_by_condition_offset[(condition, offset)]
            reduction_row = {
                "condition": condition,
                "frame_offset": offset,
                "baseline": reset["matched_baseline"],
            }
            for metric in METRICS:
                reduction_row[f"{metric}_reduction"] = safe_reduction(
                    float(keep[metric]),
                    float(reset[metric]),
                )
            matched_reductions.append(reduction_row)
    matched_reductions_csv = out_root / "phase1_0_single_sequence_reset_ablation_matched_reductions.csv"
    write_csv(matched_reductions_csv, matched_reductions, list(matched_reductions[0].keys()))

    dump_counts = {}
    expected_dump_count = len(samples)
    for condition in conditions:
        root = out_root / condition
        dump_counts[condition] = {
            "query": len(list((root / "query_memory").glob("scene-*/*.pt"))),
            "bev": len(list((root / "bev_memory").glob("scene-*/*.pt"))),
        }

    summary = {
        "clean_root": args.clean_root,
        "ablation_root": args.ablation_root,
        "ann": args.ann,
        "conditions": conditions,
        "offsets": offsets,
        "dump_counts": dump_counts,
        "summary_csv": str(csv_path),
        "reductions_csv": str(reductions_csv),
        "matched_baseline_csv": str(matched_csv),
        "matched_reductions_csv": str(matched_reductions_csv),
        "t_plus_1": {
            condition: row_by_condition_offset[(condition, 1)]
            for condition in conditions
        },
        "t_plus_1_reductions": [
            row for row in reductions if row["frame_offset"] == 1
        ],
        "t_plus_1_matched_baseline": {
            condition: matched_by_condition_offset[(condition, 1)]
            for condition in conditions
        },
        "t_plus_1_matched_reductions": [
            row for row in matched_reductions if row["frame_offset"] == 1
        ],
        "pass_dump_counts": all(
            counts["query"] == expected_dump_count and counts["bev"] == expected_dump_count
            for counts in dump_counts.values()
        ),
    }
    summary_path = out_root / "phase1_0_single_sequence_reset_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
