#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
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


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def metric(values, positive_threshold):
    vals = [value for value in values if value is not None]
    return {
        "median": percentile(vals, 0.5),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "mean": mean(vals),
        "positive_count_gt_0p01": sum(value > positive_threshold for value in vals),
        "positive_rate_gt_0p01": sum(value > positive_threshold for value in vals) / len(vals) if vals else None,
    }


def group_name_for_vpa(vpa):
    if vpa is None:
        return "missing"
    if vpa >= 0.30:
        return "vpa_ge_0p30"
    if vpa >= 0.25:
        return "vpa_0p25_0p30"
    if vpa >= 0.20:
        return "vpa_0p20_0p25"
    if vpa >= 0.15:
        return "vpa_0p15_0p20"
    return "vpa_lt_0p15"


def quantile_label(rows, token, field):
    vals = sorted(
        (to_float(row.get(field)), row["sample_token"])
        for row in rows
        if to_float(row.get(field)) is not None
    )
    vals = [(value, sample_token) for value, sample_token in vals]
    n = len(vals)
    if not n:
        return "missing"
    rank = {sample_token: idx for idx, (_, sample_token) in enumerate(vals)}
    idx = rank.get(token)
    if idx is None:
        return "missing"
    if idx < n / 3:
        return "low"
    if idx < 2 * n / 3:
        return "mid"
    return "high"


def load_group(name, summary_csv, asset_csv):
    rows = read_csv(summary_csv)
    assets = {row["sample_token"]: row for row in read_csv(asset_csv)}
    by_token = defaultdict(dict)
    for row in rows:
        if row["attack_condition"] != "attack_keep":
            continue
        token = row["target_token"]
        offset = int(row["frame_offset"])
        by_token[token][offset] = row
    out = []
    for token, offsets in by_token.items():
        if 0 not in offsets:
            continue
        asset = assets.get(token, {})
        item = {
            "group": name,
            "target_token": token,
            "scene_name": offsets[0].get("scene_name") or asset.get("scene_name", ""),
            "is_primary_scene_sample": offsets[0].get("is_primary_scene_sample", asset.get("is_primary_scene_sample", "")),
            "tag_confidence": to_float(offsets[0].get("tag_confidence") or asset.get("tag_confidence") or asset.get("mapecho_tag_confidence")),
            "diverge_vpa_coverage": to_float(offsets[0].get("diverge_vpa_coverage") or asset.get("diverge_vpa_coverage")),
            "asymmetry_score": to_float(asset.get("asymmetry_score")),
            "centrality_score": to_float(asset.get("centrality_score")),
            "selection_score": to_float(asset.get("selection_score")),
            "distance_to_reference_boundary_m": to_float(asset.get("mapecho_distance_to_reference_boundary_m")),
            "visible_cam": asset.get("visible_cam", ""),
            "selection_reason": asset.get("selection_reason", ""),
            "scene_json": asset.get("scene_json", ""),
        }
        for offset in [0, 1, 2]:
            row = offsets.get(offset, {})
            item[f"delta_cd_t{offset}"] = to_float(row.get("delta_cd_to_diverge_m"))
            item[f"delta_wrong_ref_pref_t{offset}"] = to_float(row.get("delta_wrong_reference_preference_m"))
            item[f"clean_cd_t{offset}"] = to_float(row.get("clean_cd_to_diverge_m"))
            item[f"attack_cd_t{offset}"] = to_float(row.get("attack_cd_to_diverge_m"))
            item[f"clean_wrong_ref_pref_t{offset}"] = to_float(row.get("clean_wrong_reference_preference_m"))
            item[f"attack_wrong_ref_pref_t{offset}"] = to_float(row.get("attack_wrong_reference_preference_m"))
        out.append(item)

    for item in out:
        item["vpa_bin"] = group_name_for_vpa(item["diverge_vpa_coverage"])
        item["tag_confidence_tertile"] = quantile_label(read_csv(asset_csv), item["target_token"], "tag_confidence")
        item["asymmetry_tertile"] = quantile_label(read_csv(asset_csv), item["target_token"], "asymmetry_score")
        item["clean_cd_t0_tertile"] = "missing"
    clean_sorted = sorted(
        (item["clean_cd_t0"], item["target_token"]) for item in out if item["clean_cd_t0"] is not None
    )
    n = len(clean_sorted)
    for item in out:
        ranks = {token: idx for idx, (_, token) in enumerate(clean_sorted)}
        idx = ranks.get(item["target_token"])
        if idx is None:
            continue
        if idx < n / 3:
            item["clean_cd_t0_tertile"] = "clean_low_cd"
        elif idx < 2 * n / 3:
            item["clean_cd_t0_tertile"] = "clean_mid_cd"
        else:
            item["clean_cd_t0_tertile"] = "clean_high_cd"
    return out


def summarize_subset(rows, subset_name, positive_threshold):
    out = {
        "subset": subset_name,
        "n_frames": len(rows),
        "n_scenes": len({row["scene_name"] for row in rows}),
        "median_vpa": percentile([row["diverge_vpa_coverage"] for row in rows], 0.5),
        "median_tag_confidence": percentile([row["tag_confidence"] for row in rows], 0.5),
        "median_asymmetry_score": percentile([row["asymmetry_score"] for row in rows], 0.5),
        "median_clean_cd_t0": percentile([row["clean_cd_t0"] for row in rows], 0.5),
        "median_attack_frame_delta_cd_t0": percentile([row["delta_cd_t0"] for row in rows], 0.5),
    }
    for field in ["delta_cd_t1", "delta_cd_t2"]:
        m = metric([row[field] for row in rows], positive_threshold)
        prefix = field.replace("delta_cd_", "")
        out[f"{prefix}_median_delta_cd"] = m["median"]
        out[f"{prefix}_p75_delta_cd"] = m["p75"]
        out[f"{prefix}_positive_rate_gt_0p01"] = m["positive_rate_gt_0p01"]
        out[f"{prefix}_positive_count_gt_0p01"] = m["positive_count_gt_0p01"]
    return out


def main():
    parser = argparse.ArgumentParser(description="Diagnose Phase 1.2 asset/sample quality.")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics")
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    parser.add_argument("--attack-effective-threshold", type=float, default=0.01)
    parser.add_argument(
        "--phase1-1-summary",
        default="/data/dj/MapEcho/artifacts/phase1_1_high_vpa_intensity/power_6000/summary/phase1_1_map_matched_deltas_enriched.csv",
    )
    parser.add_argument(
        "--phase1-1-assets",
        default="/data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset/phase1_1_high_vpa_assets.csv",
    )
    parser.add_argument(
        "--strict-summary",
        default="/data/dj/MapEcho/artifacts/phase1_2_strict_high_vpa_ablation_power6000/summary/phase1_1_map_matched_deltas_enriched.csv",
    )
    parser.add_argument(
        "--strict-assets",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_2_high_vpa_assets.csv",
    )
    parser.add_argument(
        "--expanded-summary",
        default="/data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000/summary/phase1_1_map_matched_deltas_enriched.csv",
    )
    parser.add_argument(
        "--expanded-assets",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    all_rows = []
    for name, summary, assets in [
        ("phase1_1_asymmetric_dist_high_vpa", args.phase1_1_summary, args.phase1_1_assets),
        ("phase1_2_ccs_candidate_strict_vpa025", args.strict_summary, args.strict_assets),
        ("phase1_2_ccs_candidate_expanded_vpa015", args.expanded_summary, args.expanded_assets),
    ]:
        all_rows.extend(load_group(name, summary, assets))

    fields = [
        "group",
        "target_token",
        "scene_name",
        "is_primary_scene_sample",
        "tag_confidence",
        "diverge_vpa_coverage",
        "vpa_bin",
        "asymmetry_score",
        "centrality_score",
        "selection_score",
        "distance_to_reference_boundary_m",
        "visible_cam",
        "selection_reason",
        "tag_confidence_tertile",
        "asymmetry_tertile",
        "clean_cd_t0_tertile",
        "delta_cd_t0",
        "delta_cd_t1",
        "delta_cd_t2",
        "clean_cd_t0",
        "clean_cd_t1",
        "clean_cd_t2",
        "attack_cd_t0",
        "attack_cd_t1",
        "attack_cd_t2",
        "delta_wrong_ref_pref_t0",
        "delta_wrong_ref_pref_t1",
        "delta_wrong_ref_pref_t2",
        "clean_wrong_ref_pref_t0",
        "clean_wrong_ref_pref_t1",
        "clean_wrong_ref_pref_t2",
        "attack_wrong_ref_pref_t0",
        "attack_wrong_ref_pref_t1",
        "attack_wrong_ref_pref_t2",
        "scene_json",
    ]
    write_csv(out_dir / "phase1_2_asset_quality_per_sample.csv", all_rows, fields)

    summary_rows = []
    for group in sorted({row["group"] for row in all_rows}):
        rows = [row for row in all_rows if row["group"] == group]
        summary_rows.append(summarize_subset(rows, group, args.positive_threshold))
        effective = [row for row in rows if (row["delta_cd_t0"] or 0.0) > args.attack_effective_threshold]
        weak = [row for row in rows if row["delta_cd_t0"] is not None and row["delta_cd_t0"] <= args.attack_effective_threshold]
        summary_rows.append(summarize_subset(effective, f"{group}__attack_effective_t0", args.positive_threshold))
        summary_rows.append(summarize_subset(weak, f"{group}__attack_weak_t0", args.positive_threshold))
    write_csv(out_dir / "phase1_2_asset_quality_by_group.csv", summary_rows)

    strat_rows = []
    for group in sorted({row["group"] for row in all_rows}):
        group_rows = [row for row in all_rows if row["group"] == group]
        for field in ["vpa_bin", "tag_confidence_tertile", "asymmetry_tertile", "clean_cd_t0_tertile", "visible_cam"]:
            for value in sorted({str(row.get(field, "")) for row in group_rows}):
                rows = [row for row in group_rows if str(row.get(field, "")) == value]
                item = summarize_subset(rows, f"{group}__{field}={value}", args.positive_threshold)
                item["strat_field"] = field
                item["strat_value"] = value
                strat_rows.append(item)
    write_csv(out_dir / "phase1_2_asset_quality_stratified.csv", strat_rows)

    scene_rows = []
    for group in sorted({row["group"] for row in all_rows}):
        for scene in sorted({row["scene_name"] for row in all_rows if row["group"] == group}):
            rows = [row for row in all_rows if row["group"] == group and row["scene_name"] == scene]
            scene_rows.append(summarize_subset(rows, f"{group}__{scene}", args.positive_threshold))
    write_csv(out_dir / "phase1_2_scene_quality_summary.csv", scene_rows)

    top_bottom = []
    expanded = [row for row in all_rows if row["group"] == "phase1_2_ccs_candidate_expanded_vpa015"]
    ranked_t1 = sorted(expanded, key=lambda row: row["delta_cd_t1"] if row["delta_cd_t1"] is not None else -999, reverse=True)
    for label, rows in [("top_t1_residue", ranked_t1[:15]), ("bottom_t1_failure", list(reversed(ranked_t1[-15:])))]:
        for rank, row in enumerate(rows, 1):
            item = {field: row.get(field) for field in fields}
            item["rank_group"] = label
            item["rank"] = rank
            item["map_overlay_t1"] = (
                f"/data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000/"
                f"{row['target_token']}/phase1_0_map_level/figures/offset_+1_boundary_overlay.png"
            )
            item["attack_overlay_hint"] = (
                f"/data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000/"
                f"{row['target_token']}/attack_assets/images"
            )
            top_bottom.append(item)
    write_csv(out_dir / "phase1_2_top_bottom_failure_cases.csv", top_bottom)

    report = {
        "out_dir": str(out_dir),
        "n_total_rows": len(all_rows),
        "groups": {
            group: {
                "n_frames": len([row for row in all_rows if row["group"] == group]),
                "n_scenes": len({row["scene_name"] for row in all_rows if row["group"] == group}),
            }
            for group in sorted({row["group"] for row in all_rows})
        },
        "attack_effective_threshold": args.attack_effective_threshold,
        "positive_threshold": args.positive_threshold,
        "per_sample_csv": str(out_dir / "phase1_2_asset_quality_per_sample.csv"),
        "by_group_csv": str(out_dir / "phase1_2_asset_quality_by_group.csv"),
        "stratified_csv": str(out_dir / "phase1_2_asset_quality_stratified.csv"),
        "scene_csv": str(out_dir / "phase1_2_scene_quality_summary.csv"),
        "top_bottom_csv": str(out_dir / "phase1_2_top_bottom_failure_cases.csv"),
    }
    (out_dir / "phase1_2_asset_quality_diagnostics_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
