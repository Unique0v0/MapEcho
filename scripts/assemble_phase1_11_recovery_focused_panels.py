#!/usr/bin/env python3
"""Assemble two-row recovery focused panels for Phase 1.11 cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_phase1_0_map_level import (
    best_boundary_metrics,
    extract_scene_boundaries,
    global_polyline_to_ego_xy,
    invert_rigid,
    lidar_polyline_to_global,
    load_csv,
    load_outputs_records,
    load_pickle,
    resample_polyline,
    transform_matrix,
)


CONDITION_DIRS = {
    "clean_keep": ("phase1_0_clean_keep",),
    "attack_keep": ("phase1_0_attack_reset_ablation", "attack_keep"),
    "attack_reset_all": ("phase1_0_attack_reset_ablation", "attack_reset_all"),
    "attack_reset_BEV": ("phase1_0_attack_reset_ablation", "attack_reset_bev"),
    "attack_reset_query": ("phase1_0_attack_reset_ablation", "attack_reset_query"),
}

PLOT_CONDITIONS = [
    "clean_keep",
    "attack_keep",
    "attack_reset_all",
    "attack_reset_BEV",
    "attack_reset_query",
]

COLORS = {
    "clean_keep": "#111827",
    "attack_keep": "#dc2626",
    "attack_reset_all": "#6b7280",
    "attack_reset_BEV": "#059669",
    "attack_reset_query": "#d97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_11_qualitative_clean_subset/"
            "phase1_11_selected_for_visual_review.csv"
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
            "phase1_11_qualitative_clean_subset/recovery_focused_panels"
        ),
    )
    parser.add_argument("--offsets", default="1,2")
    parser.add_argument("--roi-size", default="60,30")
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--boundary-z", type=float, default=-1.84)
    parser.add_argument("--sample-interval-m", type=float, default=0.25)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    try:
        value = float(value)
    except Exception:
        return ""
    return value if math.isfinite(value) else ""


def condition_path(root: Path, condition: str) -> Path:
    path = root
    for part in CONDITION_DIRS[condition]:
        path = path / part
    return path


def best_pred(record: dict, best_idx: int, interval: float):
    if best_idx < 0:
        return None
    return resample_polyline(record["vectors"][best_idx], interval)


def compute_case(args, case_row, asset_rows, offsets, roi_size):
    token = case_row["target_token"]
    root = args.run_root / token
    ann = load_pickle(root / "anns" / "clean_sequence_ann.pkl")
    target_sample = next(sample for sample in ann if sample["token"] == token)
    samples_by_offset = {int(sample["mapecho_frame_offset"]): sample for sample in ann}
    asset = asset_rows[token]

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
        condition: load_outputs_records(condition_path(root, condition), roi_size)
        for condition in PLOT_CONDITIONS
    }

    rows_by_offset = {}
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
        metrics = {}
        for condition in PLOT_CONDITIONS:
            metric = best_boundary_metrics(
                records[condition][sample["token"]],
                diverge_xy,
                reference_xy,
                args.score_thr,
                args.sample_interval_m,
            )
            metrics[condition] = metric
            metric_rows.append(
                {
                    "target_token": token,
                    "case_group": case_row.get("case_group", ""),
                    "case_rank": case_row.get("case_rank", ""),
                    "scene_name": case_row.get("scene_name", ""),
                    "frame_offset": offset,
                    "sample_token": sample["token"],
                    "condition": condition,
                    "best_idx": metric["best_idx"],
                    "best_score": finite(metric["best_score"]),
                    "cd_to_diverge_m": finite(metric["cd_to_diverge_m"]),
                    "cd_to_reference_m": finite(metric["cd_to_reference_m"]),
                    "wrong_reference_preference_m": finite(
                        metric["wrong_reference_preference_m"]
                    ),
                }
            )
        rows_by_offset[offset] = {
            "sample_token": sample["token"],
            "diverge_xy": diverge_xy,
            "reference_xy": reference_xy,
            "metrics": metrics,
            "records": {condition: records[condition][sample["token"]] for condition in PLOT_CONDITIONS},
        }
    return rows_by_offset, metric_rows


def plot_case(args, case_row, rows_by_offset, out_path: Path):
    offsets = list(rows_by_offset)
    nrows = len(offsets)
    ncols = len(PLOT_CONDITIONS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        dpi=180,
        sharex=True,
        sharey=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])

    for r, offset in enumerate(offsets):
        payload = rows_by_offset[offset]
        for c, condition in enumerate(PLOT_CONDITIONS):
            ax = axes[r, c]
            ax.plot(
                payload["diverge_xy"][:, 0],
                payload["diverge_xy"][:, 1],
                color="crimson",
                linewidth=2.6,
                label="diverge GT" if r == 0 and c == 0 else None,
            )
            ax.plot(
                payload["reference_xy"][:, 0],
                payload["reference_xy"][:, 1],
                color="seagreen",
                linewidth=2.0,
                label="reference" if r == 0 and c == 0 else None,
            )
            metric = payload["metrics"][condition]
            pred = best_pred(
                payload["records"][condition],
                int(metric["best_idx"]),
                args.sample_interval_m,
            )
            if pred is not None:
                ax.plot(
                    pred[:, 0],
                    pred[:, 1],
                    color=COLORS[condition],
                    linewidth=2.4,
                    alpha=0.96,
                    label="metric-selected pred" if r == 0 and c == 0 else None,
                )
            ax.set_title(
                f"{condition}\nCD={float(metric['cd_to_diverge_m']):.2f}",
                fontsize=8,
            )
            if c == 0:
                ax.set_ylabel(f"t+{offset}", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.22)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-15, 15)
    axes[0, 0].legend(loc="lower left", fontsize=6)
    fig.suptitle(
        f"{case_row.get('case_group')} #{case_row.get('case_rank')} | "
        f"{case_row.get('target_token')[:8]} | {case_row.get('scene_name', '')}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = read_csv(args.case_csv)
    asset_rows = {row["sample_token"]: row for row in load_csv(args.asset_csv)}
    offsets = [int(item.strip()) for item in args.offsets.split(",") if item.strip()]
    roi_size = [float(item.strip()) for item in args.roi_size.split(",")]

    all_metrics = []
    panel_paths = []
    for row in case_rows:
        rows_by_offset, metric_rows = compute_case(args, row, asset_rows, offsets, roi_size)
        all_metrics.extend(metric_rows)
        out_path = (
            args.out_dir
            / f"{row['case_group']}_{row['case_rank']}_{row['target_token'][:8]}_recovery_panel.png"
        )
        plot_case(args, row, rows_by_offset, out_path)
        panel_paths.append(str(out_path))
    write_csv(args.out_dir / "phase1_11_recovery_focused_panel_metrics.csv", all_metrics)
    (args.out_dir / "phase1_11_recovery_focused_panel_paths.txt").write_text(
        "\n".join(panel_paths) + "\n"
    )
    summary = {
        "case_csv": str(args.case_csv),
        "out_dir": str(args.out_dir),
        "num_cases": len(case_rows),
        "offsets": offsets,
        "panel_paths": panel_paths,
    }
    (args.out_dir / "phase1_11_recovery_focused_panel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
