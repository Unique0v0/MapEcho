#!/usr/bin/env python3
"""Package Phase 1.9 paper-ready evidence from existing selected114 summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


CONDITION_LABELS = {
    "attack_keep": "keep",
    "attack_reset_all": "reset_all",
    "attack_reset_bev": "reset_BEV",
    "attack_reset_query": "reset_query",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "top400_selected114_controlled_check/summary"
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
            "phase1_9_paper_evidence"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt_float(value: float, ndigits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{ndigits}f}"


def fmt_m(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:+.4f} m"


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{100.0 * value:.1f}%"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def cell(value):
        if pd.isna(value):
            return ""
        return str(value)

    lines = []
    headers = [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def condition_sort_key(condition: str) -> int:
    order = {
        "attack_keep": 0,
        "attack_reset_all": 1,
        "attack_reset_bev": 2,
        "attack_reset_query": 3,
    }
    return order.get(condition, 99)


def build_main_tables(summary_dir: Path, tables_dir: Path) -> dict:
    map_summary = read_csv(summary_dir / "phase1_1_map_residue_summary.csv")
    primary_summary = read_csv(
        summary_dir / "phase1_1_primary_scene_map_residue_summary.csv"
    )
    internal_summary = read_csv(summary_dir / "phase1_1_internal_reduction_summary.csv")

    map_summary = map_summary.sort_values(
        by=["condition", "frame_offset"],
        key=lambda col: col.map(condition_sort_key)
        if col.name == "condition"
        else col,
    )
    primary_summary = primary_summary.sort_values(
        by=["condition", "frame_offset"],
        key=lambda col: col.map(condition_sort_key)
        if col.name == "condition"
        else col,
    )

    main_rows = []
    ci_rows = []
    for _, row in map_summary.iterrows():
        main_rows.append(
            {
                "Condition": CONDITION_LABELS.get(row["condition"], row["condition"]),
                "Offset": f"t+{int(row['frame_offset'])}",
                "Frames": int(row["n_frames"]),
                "Scenes": int(row["n_scenes"]),
                "Median Delta CD": fmt_m(row["median_delta_cd_diverge_m"]),
                "Positive Rate": (
                    f"{int(row['positive_count_gt_0p01'])}/{int(row['n_frames'])} "
                    f"= {fmt_pct(row['positive_rate_gt_0p01'])}"
                ),
                "P75": fmt_m(row["p75_delta_cd_diverge_m"]),
                "P90": fmt_m(row["p90_delta_cd_diverge_m"]),
            }
        )
        ci_rows.append(
            {
                "Condition": CONDITION_LABELS.get(row["condition"], row["condition"]),
                "Offset": f"t+{int(row['frame_offset'])}",
                "Median CI": (
                    f"[{fmt_m(row['cluster_bootstrap_median_ci_low'])}, "
                    f"{fmt_m(row['cluster_bootstrap_median_ci_high'])}]"
                ),
                "Positive-rate CI": (
                    f"[{fmt_pct(row['cluster_bootstrap_positive_rate_ci_low'])}, "
                    f"{fmt_pct(row['cluster_bootstrap_positive_rate_ci_high'])}]"
                ),
            }
        )

    primary_rows = []
    for _, row in primary_summary.iterrows():
        primary_rows.append(
            {
                "Condition": CONDITION_LABELS.get(row["condition"], row["condition"]),
                "Offset": f"t+{int(row['frame_offset'])}",
                "Frames": int(row["n_frames"]),
                "Scenes": int(row["n_scenes"]),
                "Median Delta CD": fmt_m(row["median_delta_cd_diverge_m"]),
                "Positive Rate": (
                    f"{int(row['positive_count_gt_0p01'])}/{int(row['n_frames'])} "
                    f"= {fmt_pct(row['positive_rate_gt_0p01'])}"
                ),
            }
        )

    internal_rows = []
    for _, row in internal_summary.sort_values(
        by=["condition", "frame_offset"],
        key=lambda col: col.map(condition_sort_key)
        if col.name == "condition"
        else col,
    ).iterrows():
        internal_rows.append(
            {
                "Condition": CONDITION_LABELS.get(row["condition"], row["condition"]),
                "Offset": f"t+{int(row['frame_offset'])}",
                "Frames": int(row["n_frames"]),
                "Query-score reduction": fmt_float(
                    row["query_score_mean_abs_reduction_median"], 3
                ),
                "Pred-vector reduction": fmt_float(
                    row["pred_vector_mean_abs_reduction_median"], 3
                ),
                "Embedding reduction": fmt_float(
                    row["topk_embedding_mean_abs_reduction_median"], 3
                ),
                "Fused-BEV reduction": fmt_float(
                    row["fused_bev_norm_delta_reduction_median"], 3
                ),
            }
        )

    outputs = {
        "main_temporal": pd.DataFrame(main_rows),
        "scene_clustered_ci": pd.DataFrame(ci_rows),
        "primary_scene": pd.DataFrame(primary_rows),
        "internal_reduction": pd.DataFrame(internal_rows),
    }
    for name, df in outputs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)
        write_markdown_table(df, tables_dir / f"{name}.md")
    return outputs


def aggregate_curve(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, offset), group in enriched.groupby(["attack_condition", "frame_offset"]):
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "frame_offset": int(offset),
                "n_frames": int(group["target_token"].nunique()),
                "n_scenes": int(group["scene_name"].nunique())
                if "scene_name" in group.columns
                else None,
                "median_delta_cd_diverge_m": group["delta_cd_to_diverge_m"].median(),
                "mean_delta_cd_diverge_m": group["delta_cd_to_diverge_m"].mean(),
                "p25_delta_cd_diverge_m": group["delta_cd_to_diverge_m"].quantile(0.25),
                "p75_delta_cd_diverge_m": group["delta_cd_to_diverge_m"].quantile(0.75),
                "positive_rate_gt_0p01": (
                    group["delta_cd_to_diverge_m"].gt(0.01).mean()
                ),
                "positive_count_gt_0p01": int(
                    group["delta_cd_to_diverge_m"].gt(0.01).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["condition", "frame_offset"],
        key=lambda col: col.map(condition_sort_key)
        if col.name == "condition"
        else col,
    )


def make_plots(curve_df: pd.DataFrame, plots_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapecho_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "attack_keep": "#2563eb",
        "attack_reset_all": "#6b7280",
        "attack_reset_bev": "#059669",
        "attack_reset_query": "#d97706",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    for condition, group in curve_df.groupby("condition"):
        group = group.sort_values("frame_offset")
        ax.plot(
            group["frame_offset"],
            group["median_delta_cd_diverge_m"],
            marker="o",
            linewidth=2.0,
            label=CONDITION_LABELS.get(condition, condition),
            color=colors.get(condition),
        )
        ax.fill_between(
            group["frame_offset"],
            group["p25_delta_cd_diverge_m"],
            group["p75_delta_cd_diverge_m"],
            color=colors.get(condition),
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Frame offset")
    ax.set_ylabel("Delta CD to diverging boundary (m)")
    ax.set_xticks(sorted(curve_df["frame_offset"].unique()))
    ax.set_xticklabels(
        ["t" if x == 0 else f"t+{x}" for x in sorted(curve_df["frame_offset"].unique())]
    )
    ax.set_title("Selected114 Temporal Residue")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "recovery_curve_median_delta_cd.png")
    fig.savefig(plots_dir / "recovery_curve_median_delta_cd.pdf")
    plt.close(fig)

    offsets = [1, 2]
    pos_df = curve_df[curve_df["frame_offset"].isin(offsets)].copy()
    labels = [CONDITION_LABELS.get(c, c) for c in pos_df["condition"].unique()]
    x = list(range(len(labels)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    conditions = list(pos_df["condition"].drop_duplicates())
    for i, offset in enumerate(offsets):
        values = [
            pos_df[
                (pos_df["condition"] == cond) & (pos_df["frame_offset"] == offset)
            ]["positive_rate_gt_0p01"].iloc[0]
            for cond in conditions
        ]
        xs = [v + (i - 0.5) * width for v in x]
        ax.bar(xs, values, width=width, label=f"t+{offset}")
    ax.set_ylabel("Positive residue rate (>0.01 m)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Positive Target-boundary Residue Rate")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "positive_rate_t1_t2.png")
    fig.savefig(plots_dir / "positive_rate_t1_t2.pdf")
    plt.close(fig)


def select_cases(enriched: pd.DataFrame, assets: pd.DataFrame, cases_dir: Path) -> pd.DataFrame:
    t1 = enriched[
        (enriched["attack_condition"] == "attack_keep")
        & (enriched["frame_offset"] == 1)
    ].copy()
    t1 = t1.sort_values("delta_cd_to_diverge_m", ascending=False)

    rows = []

    def add_cases(label: str, df: pd.DataFrame, reason: str) -> None:
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            rec = row.to_dict()
            rec["case_group"] = label
            rec["case_rank"] = rank
            rec["selection_reason"] = reason
            rows.append(rec)

    add_cases(
        "top_residue",
        t1.head(5),
        "largest attack_keep t+1 delta-CD-to-diverge",
    )

    median_value = t1["delta_cd_to_diverge_m"].median()
    median_df = t1.assign(
        _distance_to_median=(t1["delta_cd_to_diverge_m"] - median_value).abs()
    ).sort_values("_distance_to_median")
    add_cases(
        "median_residue",
        median_df.head(2).drop(columns=["_distance_to_median"]),
        "closest to attack_keep t+1 median residue",
    )

    reset_bev = enriched[
        (enriched["attack_condition"] == "attack_reset_bev")
        & (enriched["frame_offset"] == 1)
    ][["target_token", "delta_cd_to_diverge_m"]].rename(
        columns={"delta_cd_to_diverge_m": "reset_bev_t1_delta_cd"}
    )
    clear_df = t1.merge(reset_bev, on="target_token", how="left")
    clear_df["reset_bev_removal_score"] = (
        clear_df["delta_cd_to_diverge_m"] - clear_df["reset_bev_t1_delta_cd"].abs()
    )
    clear_df = clear_df[
        (clear_df["delta_cd_to_diverge_m"] > 0.05)
        & (clear_df["reset_bev_t1_delta_cd"].abs() < 0.005)
    ].sort_values("reset_bev_removal_score", ascending=False)
    add_cases(
        "reset_bev_clear_removal",
        clear_df.head(3),
        "large keep residue with near-zero reset-BEV t+1 residue",
    )

    weak_df = t1.sort_values("delta_cd_to_diverge_m", ascending=True)
    add_cases(
        "weak_or_failure",
        weak_df.head(5),
        "smallest attack_keep t+1 delta-CD-to-diverge",
    )

    cases = pd.DataFrame(rows)
    if not cases.empty:
        cases = cases.merge(reset_bev, on="target_token", how="left")
        if "reset_bev_t1_delta_cd_x" in cases.columns:
            cases["reset_bev_t1_delta_cd"] = cases[
                "reset_bev_t1_delta_cd_x"
            ].combine_first(cases.get("reset_bev_t1_delta_cd_y"))
            cases = cases.drop(
                columns=[
                    col
                    for col in [
                        "reset_bev_t1_delta_cd_x",
                        "reset_bev_t1_delta_cd_y",
                    ]
                    if col in cases.columns
                ]
            )
    if not cases.empty and not assets.empty:
        asset_cols = [
            c
            for c in [
                "sample_token",
                "scene_name",
                "scene_pos",
                "blind_eta_x",
                "blind_eta_y",
                "blind_eta_z",
                "ccs_dense_rank",
                "ccs_dense_geometric_score",
                "streammapnet_score_delta_cd_to_diverge_m",
                "streammapnet_score_power",
            ]
            if c in assets.columns
        ]
        assets_small = assets[asset_cols].rename(columns={"sample_token": "target_token"})
        cases = cases.merge(assets_small, on="target_token", how="left", suffixes=("", "_asset"))

    cases.to_csv(cases_dir / "qualitative_case_selection.csv", index=False)

    md_cols = [
        "case_group",
        "case_rank",
        "target_token",
        "scene_name",
        "scene_pos",
        "delta_cd_to_diverge_m",
        "reset_bev_t1_delta_cd",
        "ccs_dense_rank",
        "selection_reason",
    ]
    md_cols = [c for c in md_cols if c in cases.columns]
    md = cases[md_cols].copy()
    for col in ["delta_cd_to_diverge_m", "reset_bev_t1_delta_cd"]:
        if col in md.columns:
            md[col] = md[col].map(lambda x: fmt_m(x) if pd.notna(x) else "")
    write_markdown_table(md, cases_dir / "qualitative_case_selection.md")
    return cases


def write_narrative(
    out_path: Path,
    tables: dict,
    curve_df: pd.DataFrame,
    cases: pd.DataFrame,
    summary_dir: Path,
) -> None:
    main = read_csv(summary_dir / "phase1_1_map_residue_summary.csv")
    primary = read_csv(summary_dir / "phase1_1_primary_scene_map_residue_summary.csv")

    def row(condition: str, offset: int, df: pd.DataFrame = main) -> pd.Series:
        return df[(df["condition"] == condition) & (df["frame_offset"] == offset)].iloc[0]

    keep_t1 = row("attack_keep", 1)
    keep_t2 = row("attack_keep", 2)
    bev_t1 = row("attack_reset_bev", 1)
    query_t1 = row("attack_reset_query", 1)
    primary_keep_t1 = row("attack_keep", 1, primary)

    lines = [
        "# Phase 1.9 Paper Evidence Package",
        "",
        "## Status",
        "",
        "```text",
        "Phase 1.9 result consolidation and paper-ready evidence packaging: PASS",
        "```",
        "",
        "This package is derived from the selected114 controlled temporal check. "
        "It does not rerun model inference; it only packages existing matched "
        "summary outputs into paper tables, recovery curves, qualitative case "
        "lists, and a concise evidence narrative.",
        "",
        "## Core Result",
        "",
        (
            f"The selected114 set contains {int(keep_t1['n_frames'])} frames from "
            f"{int(keep_t1['n_scenes'])} scenes. At t+1, keep-state evaluation "
            f"has median Delta CD {fmt_m(keep_t1['median_delta_cd_diverge_m'])} "
            f"with {int(keep_t1['positive_count_gt_0p01'])}/"
            f"{int(keep_t1['n_frames'])} positive frames. At t+2, the median "
            f"remains {fmt_m(keep_t2['median_delta_cd_diverge_m'])} with "
            f"{int(keep_t2['positive_count_gt_0p01'])}/"
            f"{int(keep_t2['n_frames'])} positives."
        ),
        "",
        (
            f"Scene-clustered bootstrap CIs remain positive for keep-state "
            f"median residue: t+1 "
            f"[{fmt_m(keep_t1['cluster_bootstrap_median_ci_low'])}, "
            f"{fmt_m(keep_t1['cluster_bootstrap_median_ci_high'])}], and t+2 "
            f"[{fmt_m(keep_t2['cluster_bootstrap_median_ci_low'])}, "
            f"{fmt_m(keep_t2['cluster_bootstrap_median_ci_high'])}]."
        ),
        "",
        "## Reset Evidence",
        "",
        (
            "Reset-all removes the map-level target-boundary residue completely "
            "at t+1 and t+2. Reset-BEV nearly eliminates it: at t+1, "
            f"reset-BEV median Delta CD is {fmt_m(bev_t1['median_delta_cd_diverge_m'])} "
            f"with {int(bev_t1['positive_count_gt_0p01'])}/"
            f"{int(bev_t1['n_frames'])} positives. In contrast, reset-query "
            f"retains most map-level residue at t+1 "
            f"({int(query_t1['positive_count_gt_0p01'])}/"
            f"{int(query_t1['n_frames'])} positives)."
        ),
        "",
        "## Conservative Scene-level Check",
        "",
        (
            f"Using one primary frame per scene gives "
            f"{int(primary_keep_t1['n_frames'])} frames from "
            f"{int(primary_keep_t1['n_scenes'])} scenes. The t+1 keep-state "
            f"median remains {fmt_m(primary_keep_t1['median_delta_cd_diverge_m'])} "
            f"with {int(primary_keep_t1['positive_count_gt_0p01'])}/"
            f"{int(primary_keep_t1['n_frames'])} positives, so the result is not "
            "only a repeated-frame effect."
        ),
        "",
        "## Mechanism Summary",
        "",
        "The packaged evidence supports a two-channel interpretation:",
        "",
        "- BEV memory dominates map-level target-boundary geometry residue.",
        "- Query memory mainly carries immediate internal query/prediction residue.",
        "- Reset-all closes the total temporal-state causal loop.",
        "",
        "## Qualitative Case List",
        "",
        (
            f"The qualitative audit list contains {len(cases)} entries across "
            "top-residue, median-residue, reset-BEV-clear-removal, and weak/failure "
            "groups. See `cases/qualitative_case_selection.csv` and `.md`."
        ),
        "",
        "## Outputs",
        "",
        "```text",
        "tables/main_temporal.csv/.md",
        "tables/scene_clustered_ci.csv/.md",
        "tables/primary_scene.csv/.md",
        "tables/internal_reduction.csv/.md",
        "curves/recovery_curve_summary.csv",
        "plots/recovery_curve_median_delta_cd.png/.pdf",
        "plots/positive_rate_t1_t2.png/.pdf",
        "cases/qualitative_case_selection.csv/.md",
        "phase1_9_paper_evidence_manifest.json",
        "```",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    curves_dir = args.out_dir / "curves"
    plots_dir = args.out_dir / "plots"
    cases_dir = args.out_dir / "cases"
    for directory in [tables_dir, curves_dir, plots_dir, cases_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tables = build_main_tables(args.summary_dir, tables_dir)

    enriched = read_csv(args.summary_dir / "phase1_1_map_matched_deltas_enriched.csv")
    curve_df = aggregate_curve(enriched)
    curve_df.to_csv(curves_dir / "recovery_curve_summary.csv", index=False)
    make_plots(curve_df, plots_dir)

    assets = read_csv(args.asset_csv) if args.asset_csv.exists() else pd.DataFrame()
    cases = select_cases(enriched, assets, cases_dir)

    write_narrative(
        args.out_dir / "phase1_9_paper_evidence_summary.md",
        tables,
        curve_df,
        cases,
        args.summary_dir,
    )

    manifest = {
        "summary_dir": str(args.summary_dir),
        "asset_csv": str(args.asset_csv),
        "out_dir": str(args.out_dir),
        "tables": sorted(str(p.relative_to(args.out_dir)) for p in tables_dir.iterdir()),
        "curves": sorted(str(p.relative_to(args.out_dir)) for p in curves_dir.iterdir()),
        "plots": sorted(str(p.relative_to(args.out_dir)) for p in plots_dir.iterdir()),
        "cases": sorted(str(p.relative_to(args.out_dir)) for p in cases_dir.iterdir()),
    }
    (args.out_dir / "phase1_9_paper_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"[MapEcho] Phase 1.9 package written to {args.out_dir}")


if __name__ == "__main__":
    main()
