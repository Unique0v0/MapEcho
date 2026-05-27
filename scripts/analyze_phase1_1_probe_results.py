#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


MAP_CONDITIONS = [
    "attack_keep",
    "attack_reset_all",
    "attack_reset_query",
    "attack_reset_bev",
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def bootstrap_ci(rows, value_field, scene_field, statistic, num_bootstrap, seed):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row[scene_field]].append(row)
    scenes = sorted(by_scene)
    if not scenes:
        return None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(num_bootstrap):
        sampled_rows = []
        for _ in scenes:
            scene = rng.choice(scenes)
            sampled_rows.extend(by_scene[scene])
        values = [to_float(row[value_field]) for row in sampled_rows]
        if statistic == "median":
            stats.append(percentile(values, 0.5))
        elif statistic == "positive_rate":
            vals = [value for value in values if value is not None]
            stats.append(sum(value > 0.01 for value in vals) / len(vals) if vals else None)
        else:
            raise ValueError(statistic)
    stats = [value for value in stats if value is not None]
    return percentile(stats, 0.025), percentile(stats, 0.975)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1.1 probe ablation results.")
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    parser.add_argument("--tag-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--num-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir)
    assets = {row["sample_token"]: row for row in read_csv(args.asset_csv)}

    map_rows = read_csv(summary_dir / "overlap_map_matched_deltas_all.csv")
    internal_rows = read_csv(summary_dir / "overlap_internal_matched_reductions_all.csv")
    enriched_map_rows = []
    for row in map_rows:
        asset = assets[row["target_token"]]
        row = dict(row)
        row["scene_name"] = asset["scene_name"]
        row["tag_confidence"] = asset.get("tag_confidence", asset.get("mapecho_tag_confidence", ""))
        row["diverge_vpa_coverage"] = asset.get("diverge_vpa_coverage", "")
        row["is_primary_scene_sample"] = asset.get("is_primary_scene_sample", "False")
        conf = to_float(row["tag_confidence"])
        row["tag_confidence_group"] = (
            "high" if conf is not None and conf >= args.tag_confidence_threshold else "low_medium"
        )
        enriched_map_rows.append(row)
    write_csv(
        out_dir / "phase1_1_map_matched_deltas_enriched.csv",
        enriched_map_rows,
        list(enriched_map_rows[0].keys()) if enriched_map_rows else [],
    )

    summary_rows = []
    for condition in MAP_CONDITIONS:
        for offset in [1, 2]:
            rows = [
                row
                for row in enriched_map_rows
                if row["attack_condition"] == condition and int(row["frame_offset"]) == offset
            ]
            values = [to_float(row["delta_cd_to_diverge_m"]) for row in rows]
            vals = [value for value in values if value is not None]
            ci_med_lo, ci_med_hi = bootstrap_ci(
                rows,
                "delta_cd_to_diverge_m",
                "scene_name",
                "median",
                args.num_bootstrap,
                args.seed,
            )
            ci_pos_lo, ci_pos_hi = bootstrap_ci(
                rows,
                "delta_cd_to_diverge_m",
                "scene_name",
                "positive_rate",
                args.num_bootstrap,
                args.seed + 1,
            )
            summary_rows.append(
                {
                    "condition": condition,
                    "frame_offset": offset,
                    "n_frames": len(rows),
                    "n_scenes": len({row["scene_name"] for row in rows}),
                    "median_delta_cd_diverge_m": percentile(vals, 0.5),
                    "p75_delta_cd_diverge_m": percentile(vals, 0.75),
                    "p90_delta_cd_diverge_m": percentile(vals, 0.90),
                    "positive_rate_gt_0p01": sum(value > args.positive_threshold for value in vals) / len(vals) if vals else None,
                    "positive_count_gt_0p01": sum(value > args.positive_threshold for value in vals),
                    "cluster_bootstrap_median_ci_low": ci_med_lo,
                    "cluster_bootstrap_median_ci_high": ci_med_hi,
                    "cluster_bootstrap_positive_rate_ci_low": ci_pos_lo,
                    "cluster_bootstrap_positive_rate_ci_high": ci_pos_hi,
                }
            )
    write_csv(out_dir / "phase1_1_map_residue_summary.csv", summary_rows, list(summary_rows[0].keys()))

    strat_rows = []
    for group in ["high", "low_medium"]:
        for condition in MAP_CONDITIONS:
            for offset in [1, 2]:
                rows = [
                    row
                    for row in enriched_map_rows
                    if row["tag_confidence_group"] == group
                    and row["attack_condition"] == condition
                    and int(row["frame_offset"]) == offset
                ]
                values = [to_float(row["delta_cd_to_diverge_m"]) for row in rows]
                vals = [value for value in values if value is not None]
                strat_rows.append(
                    {
                        "tag_confidence_group": group,
                        "condition": condition,
                        "frame_offset": offset,
                        "n_frames": len(rows),
                        "n_scenes": len({row["scene_name"] for row in rows}),
                        "median_delta_cd_diverge_m": percentile(vals, 0.5),
                        "positive_rate_gt_0p01": sum(value > args.positive_threshold for value in vals) / len(vals) if vals else None,
                        "positive_count_gt_0p01": sum(value > args.positive_threshold for value in vals),
                    }
                )
    write_csv(out_dir / "phase1_1_map_residue_by_tag_confidence.csv", strat_rows, list(strat_rows[0].keys()))

    primary_rows = [
        row for row in enriched_map_rows if str(row.get("is_primary_scene_sample", "")).lower() == "true"
    ]
    primary_summary = []
    for condition in MAP_CONDITIONS:
        for offset in [1, 2]:
            rows = [
                row
                for row in primary_rows
                if row["attack_condition"] == condition and int(row["frame_offset"]) == offset
            ]
            vals = [to_float(row["delta_cd_to_diverge_m"]) for row in rows]
            vals = [value for value in vals if value is not None]
            primary_summary.append(
                {
                    "condition": condition,
                    "frame_offset": offset,
                    "n_frames": len(rows),
                    "n_scenes": len({row["scene_name"] for row in rows}),
                    "median_delta_cd_diverge_m": percentile(vals, 0.5),
                    "positive_rate_gt_0p01": sum(value > args.positive_threshold for value in vals) / len(vals) if vals else None,
                    "positive_count_gt_0p01": sum(value > args.positive_threshold for value in vals),
                }
            )
    write_csv(out_dir / "phase1_1_primary_scene_map_residue_summary.csv", primary_summary, list(primary_summary[0].keys()))

    internal_summary = []
    for condition in ["attack_reset_all", "attack_reset_query", "attack_reset_bev"]:
        for offset in [1, 2]:
            rows = [
                row
                for row in internal_rows
                if row["condition"] == condition and int(row["frame_offset"]) == offset
            ]
            item = {
                "condition": condition,
                "frame_offset": offset,
                "n_frames": len(rows),
            }
            for field in [
                "query_score_mean_abs_reduction",
                "pred_vector_mean_abs_reduction",
                "topk_embedding_mean_abs_reduction",
                "fused_bev_norm_delta_reduction",
            ]:
                vals = [to_float(row[field]) for row in rows]
                item[f"{field}_median"] = percentile(vals, 0.5)
                item[f"{field}_mean"] = mean(vals)
            internal_summary.append(item)
    write_csv(out_dir / "phase1_1_internal_reduction_summary.csv", internal_summary, list(internal_summary[0].keys()))

    report = {
        "input_summary_dir": str(summary_dir),
        "asset_csv": args.asset_csv,
        "n_tokens": len({row["target_token"] for row in enriched_map_rows}),
        "n_scenes": len({row["scene_name"] for row in enriched_map_rows}),
        "positive_threshold": args.positive_threshold,
        "tag_confidence_threshold": args.tag_confidence_threshold,
        "num_bootstrap": args.num_bootstrap,
        "map_summary_csv": str(out_dir / "phase1_1_map_residue_summary.csv"),
        "tag_confidence_csv": str(out_dir / "phase1_1_map_residue_by_tag_confidence.csv"),
        "primary_scene_csv": str(out_dir / "phase1_1_primary_scene_map_residue_summary.csv"),
        "internal_summary_csv": str(out_dir / "phase1_1_internal_reduction_summary.csv"),
    }
    (out_dir / "phase1_1_probe_analysis_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
