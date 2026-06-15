#!/usr/bin/env python3
"""Focused qualitative map panels that highlight metric-selected boundaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_phase1_0_map_level import (
    LABEL_BOUNDARY,
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
    "clean_reset_all": ("phase1_0_reset_sanity", "reset_all"),
    "clean_reset_query": ("phase1_0_reset_sanity", "reset_query"),
    "clean_reset_BEV": ("phase1_0_reset_sanity", "reset_bev"),
    "attack_keep": ("phase1_0_attack_reset_ablation", "attack_keep"),
    "attack_reset_all": ("phase1_0_attack_reset_ablation", "attack_reset_all"),
    "attack_reset_query": ("phase1_0_attack_reset_ablation", "attack_reset_query"),
    "attack_reset_BEV": ("phase1_0_attack_reset_ablation", "attack_reset_bev"),
}

PLOT_CONDITIONS = [
    "clean_keep",
    "attack_keep",
    "attack_reset_all",
    "attack_reset_query",
    "attack_reset_BEV",
]

COLORS = {
    "clean_keep": "#111827",
    "attack_keep": "#dc2626",
    "attack_reset_all": "#6b7280",
    "attack_reset_query": "#d97706",
    "attack_reset_BEV": "#059669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_9_paper_evidence/cases/qualitative_case_selection.csv"
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
        "--asset-csv",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "model_scoring_fast_top400_selected114/"
            "ccs_model_scored_top400_selected114_assets_merged.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_10_qualitative_figures/focused_map_panels"
        ),
    )
    parser.add_argument("--tokens", default="")
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


def finite(value) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    return f"{value:.6f}" if math.isfinite(value) else ""


def condition_path(case_root: Path, condition: str) -> Path:
    path = case_root
    for part in CONDITION_DIRS[condition]:
        path = path / part
    return path


def best_pred_xy(record: dict, best_idx: int, sample_interval: float) -> np.ndarray | None:
    if best_idx < 0:
        return None
    return resample_polyline(record["vectors"][best_idx], sample_interval)


def compute_case(args, case_row, asset_rows, offsets, roi_size):
    token = case_row["target_token"]
    case_root = args.run_root / token
    ann = load_pickle(case_root / "anns" / "clean_sequence_ann.pkl")
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
        condition: load_outputs_records(condition_path(case_root, condition), roi_size)
        for condition in CONDITION_DIRS
    }

    metric_rows = []
    panel_paths = []
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

        metrics_by_condition = {}
        for condition in CONDITION_DIRS:
            record = records[condition][sample["token"]]
            metrics = best_boundary_metrics(
                record,
                diverge_xy,
                reference_xy,
                args.score_thr,
                args.sample_interval_m,
            )
            metrics_by_condition[condition] = metrics
            metric_rows.append(
                {
                    "target_token": token,
                    "case_group": case_row.get("case_group", ""),
                    "case_rank": case_row.get("case_rank", ""),
                    "scene_name": case_row.get("scene_name", ""),
                    "frame_offset": offset,
                    "sample_token": sample["token"],
                    "condition": condition,
                    "best_idx": metrics["best_idx"],
                    "best_score": finite(metrics["best_score"]),
                    "best_prop": metrics["best_prop"],
                    "cd_to_diverge_m": finite(metrics["cd_to_diverge_m"]),
                    "cd_to_reference_m": finite(metrics["cd_to_reference_m"]),
                    "wrong_reference_preference_m": finite(
                        metrics["wrong_reference_preference_m"]
                    ),
                }
            )

        panel_path = args.out_dir / f"{case_row['case_group']}_{case_row['case_rank']}_{token[:8]}_offset_{offset:+d}_focused.png"
        plot_focused_panel(
            panel_path,
            case_row,
            offset,
            sample["token"],
            diverge_xy,
            reference_xy,
            records,
            metrics_by_condition,
            args.sample_interval_m,
        )
        panel_paths.append(panel_path)
    return metric_rows, panel_paths


def plot_focused_panel(
    out_path: Path,
    case_row: dict[str, str],
    offset: int,
    sample_token: str,
    diverge_xy: np.ndarray,
    reference_xy: np.ndarray,
    records: dict,
    metrics_by_condition: dict,
    sample_interval: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180, sharex=True, sharey=True)

    def setup_axis(ax, title):
        ax.plot(diverge_xy[:, 0], diverge_xy[:, 1], color="crimson", linewidth=3.0, label="diverge GT")
        ax.plot(reference_xy[:, 0], reference_xy[:, 1], color="seagreen", linewidth=2.4, label="reference")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-15, 15)

    setup_axis(axes[0], "Metric-selected best boundary only")
    for condition in PLOT_CONDITIONS:
        record = records[condition][sample_token]
        metrics = metrics_by_condition[condition]
        pred = best_pred_xy(record, int(metrics["best_idx"]), sample_interval)
        if pred is None:
            continue
        axes[0].plot(
            pred[:, 0],
            pred[:, 1],
            color=COLORS[condition],
            linewidth=2.1 if condition != "clean_keep" else 2.8,
            alpha=0.95,
            label=f"{condition} cd={metrics['cd_to_diverge_m']:.2f}",
        )

    setup_axis(axes[1], "All high-score boundary predictions, best highlighted")
    for condition in ["clean_keep", "attack_keep", "attack_reset_all", "attack_reset_BEV"]:
        record = records[condition][sample_token]
        labels = record["labels"]
        scores = record["scores"]
        mask = (labels == LABEL_BOUNDARY) & (scores >= 0.1)
        indices = np.flatnonzero(mask)
        for idx in indices:
            pred = resample_polyline(record["vectors"][idx], sample_interval)
            axes[1].plot(pred[:, 0], pred[:, 1], color=COLORS[condition], alpha=0.14, linewidth=1.0)
        best_idx = int(metrics_by_condition[condition]["best_idx"])
        pred = best_pred_xy(record, best_idx, sample_interval)
        if pred is not None:
            axes[1].plot(
                pred[:, 0],
                pred[:, 1],
                color=COLORS[condition],
                alpha=0.98,
                linewidth=2.2,
                label=f"{condition} best",
            )

    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        f"{case_row.get('case_group')} #{case_row.get('case_rank')} | "
        f"{case_row.get('target_token')[:8]} | offset t+{offset} | frame={sample_token[:8]}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [int(item.strip()) for item in args.offsets.split(",") if item.strip()]
    roi_size = [float(item.strip()) for item in args.roi_size.split(",")]
    case_rows = read_csv(args.case_csv)
    if args.tokens:
        wanted = {item.strip() for item in args.tokens.split(",") if item.strip()}
        case_rows = [row for row in case_rows if row["target_token"] in wanted]
    asset_rows = {row["sample_token"]: row for row in load_csv(args.asset_csv)}

    all_metrics = []
    all_panels = []
    for case_row in case_rows:
        metric_rows, panel_paths = compute_case(args, case_row, asset_rows, offsets, roi_size)
        all_metrics.extend(metric_rows)
        all_panels.extend(str(path) for path in panel_paths)

    write_csv(args.out_dir / "focused_map_panel_metrics.csv", all_metrics)
    (args.out_dir / "focused_map_panel_paths.txt").write_text("\n".join(all_panels) + "\n")
    print(f"[MapEcho] focused map panels written to {args.out_dir}")


if __name__ == "__main__":
    main()
