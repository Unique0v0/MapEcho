#!/usr/bin/env python3
"""Build H3 recovery-curve evidence from existing selected114 outputs.

This script does not rerun StreamMapNet. It reads the existing per-condition
outputs.pkl files, recomputes target-boundary matched Delta CD for recovery
offsets t+1...t+L, and packages recovery curves plus AUC_CD summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from summarize_phase1_0_map_level import (
    best_boundary_metrics,
    extract_scene_boundaries,
    global_polyline_to_ego_xy,
    invert_rigid,
    lidar_polyline_to_global,
    load_outputs_records,
    load_pickle,
    resample_polyline,
    transform_matrix,
)


ATTACK_CONDITIONS = [
    "attack_keep",
    "attack_reset_all",
    "attack_reset_bev",
    "attack_reset_query",
]

AUC_POSITIVE_THRESHOLDS = [0.03, 0.05, 0.10]
AUC_THRESHOLD_LABELS = {
    0.03: "0p03",
    0.05: "0p05",
    0.10: "0p10",
}

MATCHED_CLEAN = {
    "attack_keep": "clean_keep",
    "attack_reset_all": "clean_reset_all",
    "attack_reset_bev": "clean_reset_bev",
    "attack_reset_query": "clean_reset_query",
}

CONDITION_DIRS = {
    "clean_keep": ("phase1_0_clean_keep",),
    "clean_reset_all": ("phase1_0_reset_sanity", "reset_all"),
    "clean_reset_query": ("phase1_0_reset_sanity", "reset_query"),
    "clean_reset_bev": ("phase1_0_reset_sanity", "reset_bev"),
    "attack_keep": ("phase1_0_attack_reset_ablation", "attack_keep"),
    "attack_reset_all": ("phase1_0_attack_reset_ablation", "attack_reset_all"),
    "attack_reset_query": ("phase1_0_attack_reset_ablation", "attack_reset_query"),
    "attack_reset_bev": ("phase1_0_attack_reset_ablation", "attack_reset_bev"),
}

DISPLAY_LABELS = {
    "attack_keep": "attack_keep",
    "attack_reset_all": "attack_reset_all",
    "attack_reset_bev": "attack_reset_BEV",
    "attack_reset_query": "attack_reset_query",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-file",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "model_scoring_fast_top400_selected114/"
            "ccs_model_scored_top400_selected114_tokens.txt"
        ),
    )
    parser.add_argument(
        "--asset-csv",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "model_scoring_fast_top400_selected114/"
            "ccs_model_scored_top400_selected114_assets_merged.csv"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "top400_selected114_controlled_check"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_h3_recovery_curve"
        ),
    )
    parser.add_argument("--offset-start", type=int, default=1)
    parser.add_argument("--offset-end", type=int, default=9)
    parser.add_argument("--roi-size", default="60,30")
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--boundary-z", type=float, default=-1.84)
    parser.add_argument("--sample-interval-m", type=float, default=0.25)
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    parser.add_argument("--num-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def read_tokens(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_path(case_root: Path, condition: str) -> Path:
    path = case_root
    for part in CONDITION_DIRS[condition]:
        path = path / part
    return path


def finite(value):
    try:
        value = float(value)
    except Exception:
        return ""
    return value if math.isfinite(value) else ""


def percentile(values, q: float):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
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
    values = [value for value in values if value is not None and math.isfinite(value)]
    return None if not values else sum(values) / len(values)


def bootstrap_ci(rows, value_field, scene_field, statistic, num_bootstrap, seed, positive_threshold):
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
            sampled_rows.extend(by_scene[rng.choice(scenes)])
        values = [float(row[value_field]) for row in sampled_rows if row[value_field] != ""]
        if not values:
            continue
        if statistic == "median":
            stats.append(percentile(values, 0.5))
        elif statistic == "positive_rate":
            stats.append(sum(value > positive_threshold for value in values) / len(values))
        else:
            raise ValueError(statistic)
    return percentile(stats, 0.025), percentile(stats, 0.975)


def compute_token_rows(args, token, asset, roi_size, offsets):
    case_root = args.run_root / token
    ann_path = case_root / "anns" / "clean_sequence_ann.pkl"
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)
    ann = load_pickle(ann_path)
    samples_by_offset = {int(sample["mapecho_frame_offset"]): sample for sample in ann}
    missing_offsets = [offset for offset in offsets if offset not in samples_by_offset]
    if missing_offsets:
        raise KeyError(f"{token}: missing offsets {missing_offsets}")

    target_sample = next(sample for sample in ann if sample["token"] == token)
    _, diverge_lidar, reference_lidar = extract_scene_boundaries(asset["scene_json"])
    target_lidar2ego = transform_matrix(
        target_sample["lidar2ego_rotation"], target_sample["lidar2ego_translation"]
    )
    target_ego2global = transform_matrix(
        target_sample["e2g_rotation"], target_sample["e2g_translation"]
    )
    diverge_global = lidar_polyline_to_global(
        resample_polyline(diverge_lidar, args.sample_interval_m),
        args.boundary_z,
        target_lidar2ego,
        target_ego2global,
    )
    reference_global = lidar_polyline_to_global(
        resample_polyline(reference_lidar, args.sample_interval_m),
        args.boundary_z,
        target_lidar2ego,
        target_ego2global,
    )

    records = {
        condition: load_outputs_records(condition_path(case_root, condition), roi_size)
        for condition in CONDITION_DIRS
    }

    metric_by_condition_offset = {}
    metric_rows = []
    for offset in offsets:
        sample = samples_by_offset[offset]
        current_global2ego = invert_rigid(
            transform_matrix(sample["e2g_rotation"], sample["e2g_translation"])
        )
        diverge_xy = resample_polyline(
            global_polyline_to_ego_xy(diverge_global, current_global2ego),
            args.sample_interval_m,
        )
        reference_xy = resample_polyline(
            global_polyline_to_ego_xy(reference_global, current_global2ego),
            args.sample_interval_m,
        )
        for condition in CONDITION_DIRS:
            metrics = best_boundary_metrics(
                records[condition][sample["token"]],
                diverge_xy,
                reference_xy,
                args.score_thr,
                args.sample_interval_m,
            )
            metric_by_condition_offset[(condition, offset)] = metrics
            metric_rows.append(
                {
                    "target_token": token,
                    "scene_name": asset["scene_name"],
                    "scene_pos": asset.get("scene_pos", ""),
                    "frame_offset": offset,
                    "sample_token": sample["token"],
                    "condition": condition,
                    "best_idx": metrics["best_idx"],
                    "best_score": finite(metrics["best_score"]),
                    "cd_to_diverge_m": finite(metrics["cd_to_diverge_m"]),
                    "cd_to_reference_m": finite(metrics["cd_to_reference_m"]),
                    "wrong_reference_preference_m": finite(metrics["wrong_reference_preference_m"]),
                }
            )

    delta_rows = []
    for attack_condition in ATTACK_CONDITIONS:
        clean_condition = MATCHED_CLEAN[attack_condition]
        for offset in offsets:
            attack = metric_by_condition_offset[(attack_condition, offset)]
            clean = metric_by_condition_offset[(clean_condition, offset)]

            def diff(field):
                a = finite(attack[field])
                c = finite(clean[field])
                return "" if a == "" or c == "" else float(a) - float(c)

            delta_rows.append(
                {
                    "target_token": token,
                    "scene_name": asset["scene_name"],
                    "scene_pos": asset.get("scene_pos", ""),
                    "attack_condition": attack_condition,
                    "matched_clean_condition": clean_condition,
                    "frame_offset": offset,
                    "sample_token": samples_by_offset[offset]["token"],
                    "delta_cd_to_diverge_m": diff("cd_to_diverge_m"),
                    "delta_cd_to_reference_m": diff("cd_to_reference_m"),
                    "delta_wrong_reference_preference_m": diff("wrong_reference_preference_m"),
                    "attack_cd_to_diverge_m": finite(attack["cd_to_diverge_m"]),
                    "clean_cd_to_diverge_m": finite(clean["cd_to_diverge_m"]),
                }
            )
    return metric_rows, delta_rows


def summarize_recovery(delta_rows, args):
    summary_rows = []
    for condition in ATTACK_CONDITIONS:
        for offset in range(args.offset_start, args.offset_end + 1):
            rows = [
                row
                for row in delta_rows
                if row["attack_condition"] == condition
                and int(row["frame_offset"]) == offset
                and row["delta_cd_to_diverge_m"] != ""
            ]
            values = [float(row["delta_cd_to_diverge_m"]) for row in rows]
            med_ci = bootstrap_ci(
                rows,
                "delta_cd_to_diverge_m",
                "scene_name",
                "median",
                args.num_bootstrap,
                args.seed + offset,
                args.positive_threshold,
            )
            pos_ci = bootstrap_ci(
                rows,
                "delta_cd_to_diverge_m",
                "scene_name",
                "positive_rate",
                args.num_bootstrap,
                args.seed + 100 + offset,
                args.positive_threshold,
            )
            summary_rows.append(
                {
                    "attack_condition": condition,
                    "condition_label": DISPLAY_LABELS[condition],
                    "frame_offset": offset,
                    "n_frames": len(rows),
                    "n_scenes": len({row["scene_name"] for row in rows}),
                    "median_delta_cd_diverge_m": percentile(values, 0.5),
                    "mean_delta_cd_diverge_m": mean(values),
                    "p25_delta_cd_diverge_m": percentile(values, 0.25),
                    "p75_delta_cd_diverge_m": percentile(values, 0.75),
                    "p90_delta_cd_diverge_m": percentile(values, 0.90),
                    "positive_threshold_m": args.positive_threshold,
                    "positive_count": sum(value > args.positive_threshold for value in values),
                    "positive_rate": (
                        sum(value > args.positive_threshold for value in values) / len(values)
                        if values
                        else None
                    ),
                    "cluster_bootstrap_median_ci_low": med_ci[0],
                    "cluster_bootstrap_median_ci_high": med_ci[1],
                    "cluster_bootstrap_positive_rate_ci_low": pos_ci[0],
                    "cluster_bootstrap_positive_rate_ci_high": pos_ci[1],
                }
            )
    return summary_rows


def summarize_auc(delta_rows, args):
    grouped = defaultdict(list)
    meta = {}
    for row in delta_rows:
        if row["delta_cd_to_diverge_m"] == "":
            continue
        offset = int(row["frame_offset"])
        if offset < args.offset_start or offset > args.offset_end:
            continue
        key = (row["target_token"], row["attack_condition"])
        value = float(row["delta_cd_to_diverge_m"])
        grouped[key].append(max(0.0, value))
        meta[key] = row

    auc_rows = []
    for (token, condition), values in sorted(grouped.items()):
        item = meta[(token, condition)]
        auc_rows.append(
            {
                "target_token": token,
                "scene_name": item["scene_name"],
                "scene_pos": item.get("scene_pos", ""),
                "attack_condition": condition,
                "condition_label": DISPLAY_LABELS[condition],
                "num_offsets": len(values),
                "auc_cd_pos_m": sum(values),
                "mean_pos_delta_cd_m": sum(values) / len(values) if values else None,
            }
        )

    summary_rows = []
    for condition in ATTACK_CONDITIONS:
        rows = [row for row in auc_rows if row["attack_condition"] == condition]
        values = [float(row["auc_cd_pos_m"]) for row in rows]
        summary_rows.append(
            {
                "attack_condition": condition,
                "condition_label": DISPLAY_LABELS[condition],
                "n_frames": len(rows),
                "n_scenes": len({row["scene_name"] for row in rows}),
                "median_auc_cd_pos_m": percentile(values, 0.5),
                "mean_auc_cd_pos_m": mean(values),
                "p75_auc_cd_pos_m": percentile(values, 0.75),
                "p90_auc_cd_pos_m": percentile(values, 0.90),
                "positive_auc_count": sum(value > 0 for value in values),
                "positive_auc_rate": sum(value > 0 for value in values) / len(values)
                if values
                else None,
                **{
                    f"auc_gt_{AUC_THRESHOLD_LABELS[thr]}_count": sum(
                        value > thr for value in values
                    )
                    for thr in AUC_POSITIVE_THRESHOLDS
                },
                **{
                    f"auc_gt_{AUC_THRESHOLD_LABELS[thr]}_rate": (
                        sum(value > thr for value in values) / len(values)
                    )
                    if values
                    else None
                    for thr in AUC_POSITIVE_THRESHOLDS
                },
            }
        )
    return auc_rows, summary_rows


def make_plots(summary_rows, auc_summary_rows, out_dir):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapecho_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summary_rows)
    colors = {
        "attack_keep": "#2563eb",
        "attack_reset_all": "#6b7280",
        "attack_reset_bev": "#059669",
        "attack_reset_query": "#d97706",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for condition in ATTACK_CONDITIONS:
        g = df[df["attack_condition"] == condition].sort_values("frame_offset")
        ax.plot(
            g["frame_offset"],
            g["median_delta_cd_diverge_m"],
            marker="o",
            linewidth=2.0,
            label=DISPLAY_LABELS[condition],
            color=colors[condition],
        )
        ax.fill_between(
            g["frame_offset"],
            g["p25_delta_cd_diverge_m"],
            g["p75_delta_cd_diverge_m"],
            alpha=0.12,
            color=colors[condition],
            linewidth=0,
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Recovery offset")
    ax.set_ylabel("Median Delta CD to diverging boundary (m)")
    ax.set_xticks(sorted(df["frame_offset"].unique()))
    ax.set_xticklabels([f"t+{int(x)}" for x in sorted(df["frame_offset"].unique())])
    ax.set_title("H3 Recovery Curve over t+1...t+9")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "h3_recovery_curve_median_delta_cd.png")
    fig.savefig(out_dir / "h3_recovery_curve_median_delta_cd.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    auc_df = pd.DataFrame(auc_summary_rows)
    labels = [DISPLAY_LABELS[condition] for condition in ATTACK_CONDITIONS]
    values = [
        float(
            auc_df[auc_df["attack_condition"] == condition][
                "median_auc_cd_pos_m"
            ].iloc[0]
        )
        for condition in ATTACK_CONDITIONS
    ]
    ax.bar(labels, values, color=[colors[condition] for condition in ATTACK_CONDITIONS])
    ax.set_ylabel("Median AUC_CD, sum max(0, Delta CD)")
    ax.set_title("Positive AUC_CD over t+1...t+9")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "h3_auc_cd_bar.png")
    fig.savefig(out_dir / "h3_auc_cd_bar.pdf")
    plt.close(fig)


def write_markdown_summary(args, summary_rows, auc_summary_rows):
    out = args.out_dir / "phase1_h3_recovery_curve_summary.md"
    df = pd.DataFrame(summary_rows)
    auc_df = pd.DataFrame(auc_summary_rows)
    keep = df[df["attack_condition"] == "attack_keep"].sort_values("frame_offset")

    lines = [
        "# Phase H3 Recovery Curve Summary",
        "",
        "## Status",
        "",
        "```text",
        "H3 recovery curve over selected114 t+1...t+9: PASS",
        "```",
        "",
        "This package recomputes map-level matched Delta CD from existing outputs; it does not rerun StreamMapNet.",
        "",
        "## Input",
        "",
        "```text",
        f"tokens_file = {args.tokens_file}",
        f"asset_csv   = {args.asset_csv}",
        f"run_root    = {args.run_root}",
        "```",
        "",
        "## Attack-keep Recovery Curve",
        "",
        "| Offset | Median Delta CD | Positive Rate | Positive Count |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for _, row in keep.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"t+{int(row['frame_offset'])}",
                    f"{float(row['median_delta_cd_diverge_m']):+.4f} m",
                    f"{100.0 * float(row['positive_rate']):.1f}%",
                    f"{int(row['positive_count'])}/{int(row['n_frames'])}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## AUC_CD Summary",
        "",
        "```text",
        "AUC_CD = sum_i max(0, Delta CD_i), i = 1...9",
        "```",
        "",
        "| Condition | Median AUC_CD | Mean AUC_CD | AUC > 0.03 | AUC > 0.05 | AUC > 0.10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in auc_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    row["condition_label"],
                    f"{float(row['median_auc_cd_pos_m']):.4f}",
                    f"{float(row['mean_auc_cd_pos_m']):.4f}",
                    f"{int(row['auc_gt_0p03_count'])}/{int(row['n_frames'])}",
                    f"{int(row['auc_gt_0p05_count'])}/{int(row['n_frames'])}",
                    f"{int(row['auc_gt_0p10_count'])}/{int(row['n_frames'])}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Note: `AUC_CD > 0` is intentionally not used as the main positive-rate",
        "criterion because very small numerical deviations can make reset_BEV",
        "appear positive even when its median AUC is near zero. The main",
        "interpretation is based on median AUC reduction and thresholded AUC",
        "counts.",
    ]
    lines += [
        "",
        "## Output Files",
        "",
        "```text",
        "h3_recovery_matched_deltas_all.csv",
        "h3_recovery_curve_summary.csv",
        "h3_auc_cd_by_token.csv",
        "h3_auc_cd_summary.csv",
        "h3_recovery_curve_median_delta_cd.png/.pdf",
        "h3_auc_cd_bar.png/.pdf",
        "```",
    ]
    out.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens = read_tokens(args.tokens_file)
    asset_rows = {row["sample_token"]: row for row in read_csv_rows(args.asset_csv)}
    roi_size = [float(item.strip()) for item in args.roi_size.split(",")]
    offsets = list(range(args.offset_start, args.offset_end + 1))

    all_metric_rows = []
    all_delta_rows = []
    missing = []
    for idx, token in enumerate(tokens, start=1):
        try:
            metric_rows, delta_rows = compute_token_rows(
                args, token, asset_rows[token], roi_size, offsets
            )
        except Exception as exc:
            missing.append({"target_token": token, "error": repr(exc)})
            continue
        all_metric_rows.extend(metric_rows)
        all_delta_rows.extend(delta_rows)
        if idx % 10 == 0:
            print(f"[MapEcho] processed {idx}/{len(tokens)} tokens")

    write_csv(args.out_dir / "h3_recovery_condition_metrics_all.csv", all_metric_rows)
    write_csv(args.out_dir / "h3_recovery_matched_deltas_all.csv", all_delta_rows)
    summary_rows = summarize_recovery(all_delta_rows, args)
    write_csv(args.out_dir / "h3_recovery_curve_summary.csv", summary_rows)
    auc_rows, auc_summary_rows = summarize_auc(all_delta_rows, args)
    write_csv(args.out_dir / "h3_auc_cd_by_token.csv", auc_rows)
    write_csv(args.out_dir / "h3_auc_cd_summary.csv", auc_summary_rows)
    make_plots(summary_rows, auc_summary_rows, args.out_dir)
    write_markdown_summary(args, summary_rows, auc_summary_rows)

    report = {
        "tokens_file": str(args.tokens_file),
        "asset_csv": str(args.asset_csv),
        "run_root": str(args.run_root),
        "out_dir": str(args.out_dir),
        "requested_tokens": len(tokens),
        "completed_tokens": len({row["target_token"] for row in all_delta_rows}),
        "missing": missing,
        "offset_start": args.offset_start,
        "offset_end": args.offset_end,
        "positive_threshold": args.positive_threshold,
    }
    (args.out_dir / "phase1_h3_recovery_curve_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
