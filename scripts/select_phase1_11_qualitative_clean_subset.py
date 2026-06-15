#!/usr/bin/env python3
"""Select Phase 1.11 qualitative-friendly and clean-quality subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h3-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_h3_recovery_curve"
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
            "phase1_11_qualitative_clean_subset"
        ),
    )
    parser.add_argument("--t1-delta-thr", type=float, default=0.05)
    parser.add_argument("--keep-auc-thr", type=float, default=0.15)
    parser.add_argument("--reset-bev-auc-thr", type=float, default=0.02)
    parser.add_argument("--reset-query-auc-thr", type=float, default=0.10)
    parser.add_argument("--clean-strict-quantile", type=float, default=0.50)
    parser.add_argument("--clean-relaxed-quantile", type=float, default=0.75)
    parser.add_argument("--max-review-cases", type=int, default=20)
    return parser.parse_args()


def pct_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def load_inputs(args: argparse.Namespace):
    auc = pd.read_csv(args.h3_dir / "h3_auc_cd_by_token.csv")
    deltas = pd.read_csv(args.h3_dir / "h3_recovery_matched_deltas_all.csv")
    metrics = pd.read_csv(args.h3_dir / "h3_recovery_condition_metrics_all.csv")
    assets = pd.read_csv(args.asset_csv)
    return auc, deltas, metrics, assets


def build_candidate_table(args: argparse.Namespace) -> pd.DataFrame:
    auc, deltas, metrics, assets = load_inputs(args)

    auc_wide = auc.pivot_table(
        index=["target_token", "scene_name", "scene_pos"],
        columns="attack_condition",
        values="auc_cd_pos_m",
        aggfunc="first",
    ).reset_index()
    auc_wide = auc_wide.rename(
        columns={
            "attack_keep": "attack_keep_auc_cd",
            "attack_reset_all": "attack_reset_all_auc_cd",
            "attack_reset_bev": "attack_reset_bev_auc_cd",
            "attack_reset_query": "attack_reset_query_auc_cd",
        }
    )

    t1 = deltas[deltas["frame_offset"] == 1].pivot_table(
        index="target_token",
        columns="attack_condition",
        values="delta_cd_to_diverge_m",
        aggfunc="first",
    )
    t2 = deltas[deltas["frame_offset"] == 2].pivot_table(
        index="target_token",
        columns="attack_condition",
        values="delta_cd_to_diverge_m",
        aggfunc="first",
    )
    t3 = deltas[deltas["frame_offset"] == 3].pivot_table(
        index="target_token",
        columns="attack_condition",
        values="delta_cd_to_diverge_m",
        aggfunc="first",
    )
    for df, prefix in [(t1, "t1"), (t2, "t2"), (t3, "t3")]:
        df.columns = [f"{condition}_{prefix}_delta_cd" for condition in df.columns]

    clean = metrics[metrics["condition"] == "clean_keep"].copy()
    clean_t1 = clean[clean["frame_offset"] == 1][
        ["target_token", "cd_to_diverge_m"]
    ].rename(columns={"cd_to_diverge_m": "clean_keep_t1_cd"})
    clean_t2 = clean[clean["frame_offset"] == 2][
        ["target_token", "cd_to_diverge_m"]
    ].rename(columns={"cd_to_diverge_m": "clean_keep_t2_cd"})
    clean_recovery = (
        clean[clean["frame_offset"].between(1, 9)]
        .groupby("target_token")["cd_to_diverge_m"]
        .agg(["median", "mean", "std"])
        .rename(
            columns={
                "median": "clean_keep_recovery_cd_median",
                "mean": "clean_keep_recovery_cd_mean",
                "std": "clean_keep_recovery_cd_std",
            }
        )
        .reset_index()
    )

    table = auc_wide.merge(t1.reset_index(), on="target_token", how="left")
    table = table.merge(t2.reset_index(), on="target_token", how="left")
    table = table.merge(t3.reset_index(), on="target_token", how="left")
    table = table.merge(clean_t1, on="target_token", how="left")
    table = table.merge(clean_t2, on="target_token", how="left")
    table = table.merge(clean_recovery, on="target_token", how="left")

    asset_cols = [
        col
        for col in [
            "sample_token",
            "ccs_dense_rank",
            "ccs_dense_geometric_score",
            "streammapnet_score_delta_cd_to_diverge_m",
            "streammapnet_score_power",
            "blind_eta_x",
            "blind_eta_y",
            "blind_eta_z",
        ]
        if col in assets.columns
    ]
    asset_small = assets[asset_cols].rename(columns={"sample_token": "target_token"})
    table = table.merge(asset_small, on="target_token", how="left")

    table["clean_keep_t1_cd_percentile"] = pct_rank(table["clean_keep_t1_cd"])
    table["clean_recovery_std_percentile"] = pct_rank(
        table["clean_keep_recovery_cd_std"].fillna(table["clean_keep_recovery_cd_std"].max())
    )

    table["h3_qualitative_signal_pass"] = (
        (table["attack_keep_t1_delta_cd"] > args.t1_delta_thr)
        & (table["attack_keep_auc_cd"] > args.keep_auc_thr)
        & (table["attack_reset_all_auc_cd"] <= 1e-12)
        & (table["attack_reset_bev_auc_cd"] < args.reset_bev_auc_thr)
        & (table["attack_reset_query_auc_cd"] > args.reset_query_auc_thr)
    )
    table["clean_quality_strict_pass"] = (
        (table["clean_keep_t1_cd_percentile"] <= args.clean_strict_quantile)
        & (table["clean_recovery_std_percentile"] <= args.clean_strict_quantile)
    )
    table["clean_quality_relaxed_pass"] = (
        (table["clean_keep_t1_cd_percentile"] <= args.clean_relaxed_quantile)
        & (table["clean_recovery_std_percentile"] <= args.clean_relaxed_quantile)
    )
    table["qualitative_strict_pass"] = (
        table["h3_qualitative_signal_pass"] & table["clean_quality_strict_pass"]
    )
    table["qualitative_relaxed_pass"] = (
        table["h3_qualitative_signal_pass"] & table["clean_quality_relaxed_pass"]
    )

    table["selection_score"] = (
        table["attack_keep_auc_cd"].fillna(0)
        + table["attack_keep_t1_delta_cd"].clip(lower=0).fillna(0)
        + table["attack_reset_query_auc_cd"].fillna(0)
        - 4.0 * table["attack_reset_bev_auc_cd"].fillna(0)
        - 0.02 * table["clean_keep_t1_cd_percentile"].fillna(1)
    )
    return table


def summarize_subset(table: pd.DataFrame, deltas: pd.DataFrame, auc: pd.DataFrame, name: str, mask: pd.Series):
    tokens = set(table.loc[mask, "target_token"])
    rows = []
    for condition in CONDITIONS:
        for offset in [1, 2]:
            sub = deltas[
                (deltas["target_token"].isin(tokens))
                & (deltas["attack_condition"] == condition)
                & (deltas["frame_offset"] == offset)
            ]
            vals = sub["delta_cd_to_diverge_m"].dropna()
            rows.append(
                {
                    "subset": name,
                    "attack_condition": condition,
                    "frame_offset": offset,
                    "n_frames": sub["target_token"].nunique(),
                    "n_scenes": sub["scene_name"].nunique(),
                    "median_delta_cd_diverge_m": vals.median() if len(vals) else np.nan,
                    "positive_count_gt_0p01": int((vals > 0.01).sum()),
                    "positive_rate_gt_0p01": float((vals > 0.01).mean()) if len(vals) else np.nan,
                }
            )

    auc_rows = []
    for condition in CONDITIONS:
        sub = auc[
            (auc["target_token"].isin(tokens))
            & (auc["attack_condition"] == condition)
        ]
        vals = sub["auc_cd_pos_m"].dropna()
        auc_rows.append(
            {
                "subset": name,
                "attack_condition": condition,
                "n_frames": sub["target_token"].nunique(),
                "n_scenes": sub["scene_name"].nunique(),
                "median_auc_cd_pos_m": vals.median() if len(vals) else np.nan,
                "mean_auc_cd_pos_m": vals.mean() if len(vals) else np.nan,
                "positive_auc_count": int((vals > 0).sum()),
                "positive_auc_rate": float((vals > 0).mean()) if len(vals) else np.nan,
                **{
                    f"auc_gt_{AUC_THRESHOLD_LABELS[thr]}_count": int((vals > thr).sum())
                    for thr in AUC_POSITIVE_THRESHOLDS
                },
                **{
                    f"auc_gt_{AUC_THRESHOLD_LABELS[thr]}_rate": float((vals > thr).mean())
                    if len(vals)
                    else np.nan
                    for thr in AUC_POSITIVE_THRESHOLDS
                },
            }
        )
    return rows, auc_rows


def make_case_csv(df: pd.DataFrame, out_path: Path, group: str, max_rows: int) -> pd.DataFrame:
    cols = []
    selected = df.sort_values(
        ["selection_score", "attack_keep_auc_cd", "attack_keep_t1_delta_cd"],
        ascending=False,
    ).head(max_rows).copy()
    rows = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        rows.append(
            {
                "case_group": group,
                "case_rank": rank,
                "target_token": row["target_token"],
                "scene_name": row["scene_name"],
                "scene_pos": row["scene_pos"],
                "delta_cd_to_diverge_m": row["attack_keep_t1_delta_cd"],
                "reset_bev_t1_delta_cd": row["attack_reset_bev_t1_delta_cd"],
                "ccs_dense_rank": row.get("ccs_dense_rank", ""),
                "attack_keep_auc_cd": row["attack_keep_auc_cd"],
                "attack_reset_bev_auc_cd": row["attack_reset_bev_auc_cd"],
                "attack_reset_query_auc_cd": row["attack_reset_query_auc_cd"],
                "clean_keep_t1_cd": row["clean_keep_t1_cd"],
                "clean_keep_t1_cd_percentile": row["clean_keep_t1_cd_percentile"],
                "selection_reason": (
                    "H3 AUC-qualified, reset-BEV removed, reset-query retained, "
                    "clean-quality screened"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_summary_md(args, table, strict_cases, relaxed_cases, subset_summary, auc_summary):
    out = args.out_dir / "phase1_11_selection_summary.md"
    lines = [
        "# Phase 1.11 Qualitative-friendly and Clean-quality Selection",
        "",
        "## Status",
        "",
        "```text",
        "Phase 1.11 selection from existing H3 outputs: PASS",
        "```",
        "",
        "## Gates",
        "",
        "```text",
        f"attack_keep_t1_delta_cd > {args.t1_delta_thr}",
        f"attack_keep_AUC_CD > {args.keep_auc_thr}",
        "attack_reset_all_AUC_CD = 0",
        f"attack_reset_BEV_AUC_CD < {args.reset_bev_auc_thr}",
        f"attack_reset_query_AUC_CD > {args.reset_query_auc_thr}",
        f"clean strict: t1 CD percentile <= {args.clean_strict_quantile}, recovery std percentile <= {args.clean_strict_quantile}",
        f"clean relaxed: t1 CD percentile <= {args.clean_relaxed_quantile}, recovery std percentile <= {args.clean_relaxed_quantile}",
        "```",
        "",
        "## Counts",
        "",
        "```text",
        f"all selected114 tokens = {len(table)}",
        f"H3 qualitative signal pass = {int(table['h3_qualitative_signal_pass'].sum())}",
        f"clean_quality_strict_pass = {int(table['clean_quality_strict_pass'].sum())}",
        f"clean_quality_relaxed_pass = {int(table['clean_quality_relaxed_pass'].sum())}",
        f"qualitative_strict_pass = {int(table['qualitative_strict_pass'].sum())}",
        f"qualitative_relaxed_pass = {int(table['qualitative_relaxed_pass'].sum())}",
        f"strict review cases written = {len(strict_cases)}",
        f"relaxed review cases written = {len(relaxed_cases)}",
        "```",
        "",
        "## Output Files",
        "",
        "```text",
        "phase1_11_candidate_table.csv",
        "phase1_11_qualitative_candidates_strict.csv",
        "phase1_11_qualitative_candidates_relaxed.csv",
        "phase1_11_selected_for_visual_review.csv",
        "phase1_11_clean_quality_subset_summary.csv",
        "phase1_11_clean_quality_auc_summary.csv",
        "phase1_11_selection_summary.json",
        "```",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = build_candidate_table(args)
    table.to_csv(args.out_dir / "phase1_11_candidate_table.csv", index=False)

    auc, deltas, _, _ = load_inputs(args)
    strict_mask = table["qualitative_strict_pass"]
    relaxed_mask = table["qualitative_relaxed_pass"]

    strict_cases = make_case_csv(
        table[strict_mask],
        args.out_dir / "phase1_11_qualitative_candidates_strict.csv",
        "phase1_11_strict",
        args.max_review_cases,
    )
    relaxed_cases = make_case_csv(
        table[relaxed_mask],
        args.out_dir / "phase1_11_qualitative_candidates_relaxed.csv",
        "phase1_11_relaxed",
        args.max_review_cases,
    )
    selected = strict_cases if len(strict_cases) >= 5 else relaxed_cases
    selected.to_csv(args.out_dir / "phase1_11_selected_for_visual_review.csv", index=False)
    (args.out_dir / "phase1_11_selected_for_visual_review_tokens.txt").write_text(
        "\n".join(selected["target_token"].astype(str).tolist()) + ("\n" if len(selected) else "")
    )

    subset_rows = []
    auc_rows = []
    for name, mask in [
        ("full_selected114", pd.Series(True, index=table.index)),
        ("clean_quality_strict", table["clean_quality_strict_pass"]),
        ("clean_quality_relaxed", table["clean_quality_relaxed_pass"]),
        ("qualitative_strict", strict_mask),
        ("qualitative_relaxed", relaxed_mask),
    ]:
        rows, arows = summarize_subset(table, deltas, auc, name, mask)
        subset_rows.extend(rows)
        auc_rows.extend(arows)
    subset_summary = pd.DataFrame(subset_rows)
    auc_summary = pd.DataFrame(auc_rows)
    subset_summary.to_csv(args.out_dir / "phase1_11_clean_quality_subset_summary.csv", index=False)
    auc_summary.to_csv(args.out_dir / "phase1_11_clean_quality_auc_summary.csv", index=False)

    report = {
        "out_dir": str(args.out_dir),
        "n_total": int(len(table)),
        "h3_qualitative_signal_pass": int(table["h3_qualitative_signal_pass"].sum()),
        "clean_quality_strict_pass": int(table["clean_quality_strict_pass"].sum()),
        "clean_quality_relaxed_pass": int(table["clean_quality_relaxed_pass"].sum()),
        "qualitative_strict_pass": int(strict_mask.sum()),
        "qualitative_relaxed_pass": int(relaxed_mask.sum()),
        "selected_for_visual_review": int(len(selected)),
        "selected_source": "strict" if len(strict_cases) >= 5 else "relaxed",
    }
    (args.out_dir / "phase1_11_selection_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    write_summary_md(args, table, strict_cases, relaxed_cases, subset_summary, auc_summary)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
