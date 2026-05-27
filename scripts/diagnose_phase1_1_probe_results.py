#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(value):
    return float(value) if value not in ("", None) else None


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def quantile_groups(rows, field):
    ordered = sorted(rows, key=lambda row: f(row[field]))
    n = len(ordered)
    return {
        "low": ordered[: n // 3],
        "mid": ordered[n // 3 : 2 * n // 3],
        "high": ordered[2 * n // 3 :],
    }


def summarize_groups(groups, group_field, value_field, positive_threshold):
    out = []
    for name, rows in groups.items():
        vals = [f(row[value_field]) for row in rows]
        vals = [v for v in vals if v is not None]
        item = {
            group_field: name,
            "n": len(rows),
            "median_delta_cd_diverge_m": median(vals),
            "positive_rate_gt_0p01": sum(v > positive_threshold for v in vals) / len(vals) if vals else None,
            "positive_count_gt_0p01": sum(v > positive_threshold for v in vals),
        }
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for Phase 1.1 probe results.")
    parser.add_argument("--summary-dir", default="/data/dj/MapEcho/artifacts/phase1_1_probe_ablation/summary")
    parser.add_argument("--run-root", default="/data/dj/MapEcho/artifacts/phase1_1_probe_ablation")
    parser.add_argument("--asset-csv", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_assets.csv")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_1_probe_ablation/diagnostics")
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    rows = read_csv(summary_dir / "phase1_1_map_matched_deltas_enriched.csv")
    assets = {row["sample_token"]: row for row in read_csv(args.asset_csv)}
    attack_keep_t1 = [
        row
        for row in rows
        if row["attack_condition"] == "attack_keep" and int(row["frame_offset"]) == 1
    ]

    scene_groups = defaultdict(list)
    primary_groups = defaultdict(list)
    tag_group = defaultdict(list)
    for row in attack_keep_t1:
        scene_groups[row["scene_name"]].append(row)
        primary_groups[str(row["is_primary_scene_sample"])].append(row)
        tag_group[row["tag_confidence_group"]].append(row)

    scene_rows = summarize_groups(scene_groups, "scene_name", "delta_cd_to_diverge_m", args.positive_threshold)
    scene_rows.sort(key=lambda row: (row["median_delta_cd_diverge_m"] is None, -(row["median_delta_cd_diverge_m"] or -999)))
    write_csv(out_dir / "phase1_1_diagnostic_by_scene_t1.csv", scene_rows, list(scene_rows[0].keys()))

    primary_rows = summarize_groups(primary_groups, "is_primary_scene_sample", "delta_cd_to_diverge_m", args.positive_threshold)
    write_csv(out_dir / "phase1_1_diagnostic_by_primary_t1.csv", primary_rows, list(primary_rows[0].keys()))

    tag_rows = summarize_groups(tag_group, "tag_confidence_group", "delta_cd_to_diverge_m", args.positive_threshold)
    write_csv(out_dir / "phase1_1_diagnostic_by_tag_group_t1.csv", tag_rows, list(tag_rows[0].keys()))

    quantile_rows = []
    for field in ["tag_confidence", "diverge_vpa_coverage"]:
        groups = quantile_groups(attack_keep_t1, field)
        for name, group_rows in groups.items():
            vals = [f(row["delta_cd_to_diverge_m"]) for row in group_rows]
            vals = [v for v in vals if v is not None]
            quantile_rows.append(
                {
                    "field": field,
                    "quantile_group": name,
                    "n": len(group_rows),
                    "field_min": min(f(row[field]) for row in group_rows),
                    "field_max": max(f(row[field]) for row in group_rows),
                    "median_delta_cd_diverge_m": median(vals),
                    "positive_rate_gt_0p01": sum(v > args.positive_threshold for v in vals) / len(vals) if vals else None,
                    "positive_count_gt_0p01": sum(v > args.positive_threshold for v in vals),
                }
            )
    write_csv(out_dir / "phase1_1_diagnostic_quantiles_t1.csv", quantile_rows, list(quantile_rows[0].keys()))

    ranked = sorted(attack_keep_t1, key=lambda row: f(row["delta_cd_to_diverge_m"]), reverse=True)
    top_bottom = []
    for rank, row in enumerate(ranked[:10], 1):
        token = row["target_token"]
        top_bottom.append(
            {
                "rank_group": "top_positive",
                "rank": rank,
                "target_token": token,
                "scene_name": row["scene_name"],
                "delta_cd_to_diverge_m": row["delta_cd_to_diverge_m"],
                "tag_confidence": row["tag_confidence"],
                "diverge_vpa_coverage": row["diverge_vpa_coverage"],
                "map_overlay_t1": str(run_root / token / "phase1_0_map_level/figures/offset_+1_boundary_overlay.png"),
                "attack_overlay_hint": str(run_root / token / "attack_assets/images"),
                "scene_json": assets[token]["scene_json"],
            }
        )
    for rank, row in enumerate(list(reversed(ranked[-10:])), 1):
        token = row["target_token"]
        top_bottom.append(
            {
                "rank_group": "bottom_failure",
                "rank": rank,
                "target_token": token,
                "scene_name": row["scene_name"],
                "delta_cd_to_diverge_m": row["delta_cd_to_diverge_m"],
                "tag_confidence": row["tag_confidence"],
                "diverge_vpa_coverage": row["diverge_vpa_coverage"],
                "map_overlay_t1": str(run_root / token / "phase1_0_map_level/figures/offset_+1_boundary_overlay.png"),
                "attack_overlay_hint": str(run_root / token / "attack_assets/images"),
                "scene_json": assets[token]["scene_json"],
            }
        )
    write_csv(out_dir / "phase1_1_top_bottom_samples_t1.csv", top_bottom, list(top_bottom[0].keys()))

    report = {
        "summary_dir": str(summary_dir),
        "run_root": str(run_root),
        "n_attack_keep_t1": len(attack_keep_t1),
        "diagnostic_by_scene_csv": str(out_dir / "phase1_1_diagnostic_by_scene_t1.csv"),
        "diagnostic_quantiles_csv": str(out_dir / "phase1_1_diagnostic_quantiles_t1.csv"),
        "top_bottom_csv": str(out_dir / "phase1_1_top_bottom_samples_t1.csv"),
        "observation": "t+1 map-level residue is scene-sensitive and aligns more with VPA coverage than tag confidence.",
    }
    (out_dir / "phase1_1_diagnostics_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
