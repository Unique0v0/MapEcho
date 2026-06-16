#!/usr/bin/env python3
"""Select Phase 1.13/H5 pilot tokens from existing selected114 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h3-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_h3_recovery_curve"
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
        "--phase11-table",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_11_qualitative_clean_subset/phase1_11_candidate_table.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_13_h5_continuous_pilot"
        ),
    )
    parser.add_argument("--max-primary-per-scene", type=int, default=1)
    parser.add_argument("--max-qualitative-friendly", type=int, default=40)
    return parser.parse_args()


def write_tokens(path: Path, tokens: list[str]) -> None:
    path.write_text("\n".join(tokens) + ("\n" if tokens else ""))


def bool_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return table[column].fillna(False).astype(bool)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    auc = pd.read_csv(args.h3_dir / "h3_auc_cd_by_token.csv")
    deltas = pd.read_csv(args.h3_dir / "h3_recovery_matched_deltas_all.csv")
    assets = pd.read_csv(args.asset_csv)
    phase11 = pd.read_csv(args.phase11_table) if args.phase11_table.exists() else None

    keep_auc = auc[auc["attack_condition"] == "attack_keep"][
        ["target_token", "auc_cd_pos_m"]
    ].rename(columns={"auc_cd_pos_m": "attack_keep_auc_cd"})
    t1 = deltas[
        (deltas["attack_condition"] == "attack_keep") & (deltas["frame_offset"] == 1)
    ][["target_token", "delta_cd_to_diverge_m"]].rename(
        columns={"delta_cd_to_diverge_m": "attack_keep_t1_delta_cd"}
    )

    table = assets.rename(columns={"sample_token": "target_token"}).merge(
        keep_auc, on="target_token", how="left"
    )
    table = table.merge(t1, on="target_token", how="left")
    if phase11 is not None:
        phase_cols = [
            col
            for col in [
                "target_token",
                "h3_qualitative_signal_pass",
                "clean_quality_relaxed_pass",
                "qualitative_relaxed_pass",
                "selection_score",
            ]
            if col in phase11.columns
        ]
        table = table.merge(phase11[phase_cols], on="target_token", how="left")

    table["h5_selection_score"] = (
        table["attack_keep_auc_cd"].fillna(0.0)
        + table["attack_keep_t1_delta_cd"].clip(lower=0).fillna(0.0)
        + 0.20 * bool_series(table, "qualitative_relaxed_pass").astype(float)
        + 0.05 * bool_series(table, "clean_quality_relaxed_pass").astype(float)
    )

    primary_rows = []
    for _, scene_rows in table.sort_values(
        ["scene_name", "h5_selection_score"], ascending=[True, False]
    ).groupby("scene_name", sort=True):
        primary_rows.append(scene_rows.head(args.max_primary_per_scene))
    primary = pd.concat(primary_rows, ignore_index=True).sort_values(
        "h5_selection_score", ascending=False
    )

    qualitative = table[bool_series(table, "qualitative_relaxed_pass")].sort_values(
        "h5_selection_score", ascending=False
    )
    if len(qualitative) < min(args.max_qualitative_friendly, len(table)):
        fill = table[~table["target_token"].isin(set(qualitative["target_token"]))].sort_values(
            "h5_selection_score", ascending=False
        )
        qualitative = pd.concat([qualitative, fill], ignore_index=True).head(
            args.max_qualitative_friendly
        )
    else:
        qualitative = qualitative.head(args.max_qualitative_friendly)

    primary_csv = args.out_dir / "h5_primary_scene_pilot_tokens.csv"
    primary_tokens = args.out_dir / "h5_primary_scene_pilot_tokens.txt"
    qualitative_csv = args.out_dir / "h5_qualitative_friendly_pilot_tokens.csv"
    qualitative_tokens = args.out_dir / "h5_qualitative_friendly_pilot_tokens.txt"

    primary.to_csv(primary_csv, index=False)
    qualitative.to_csv(qualitative_csv, index=False)
    write_tokens(primary_tokens, list(primary["target_token"]))
    write_tokens(qualitative_tokens, list(qualitative["target_token"]))

    summary = {
        "out_dir": str(args.out_dir),
        "primary_scene_frames": int(primary["target_token"].nunique()),
        "primary_scene_scenes": int(primary["scene_name"].nunique()),
        "qualitative_friendly_frames": int(qualitative["target_token"].nunique()),
        "qualitative_friendly_scenes": int(qualitative["scene_name"].nunique()),
        "primary_tokens": str(primary_tokens),
        "qualitative_tokens": str(qualitative_tokens),
    }
    (args.out_dir / "h5_pilot_token_selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
