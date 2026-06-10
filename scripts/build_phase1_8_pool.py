#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase1_6_ccs_style_pool import (
    as_bool,
    as_float,
    bool_text,
    calculate_curvature,
    calculate_region_curvature,
    check_point_distances,
    continuous_turning,
    heading_tail_symmetric,
    identify_diverging_boundary,
    lane_width_gate,
    load_boundaries,
    percentile,
    prefix_blacklisted,
    scene_blacklisted,
    vpa_index,
)


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


def write_tokens(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row["sample_token"] for row in rows) + ("\n" if rows else ""))


def direction_change_score(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    vectors = points[1:] - points[:-1]
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-6
    if valid.sum() < 2:
        return 0.0
    dirs = vectors[valid] / norms[valid, None]
    start = dirs[: min(3, len(dirs))].mean(axis=0)
    end = dirs[-min(3, len(dirs)) :].mean(axis=0)
    if np.linalg.norm(start) <= 1e-6 or np.linalg.norm(end) <= 1e-6:
        return 0.0
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    return float(np.degrees(np.arccos(np.clip(np.dot(start, end), -1.0, 1.0))))


def token_prefix_blacklist():
    # User visual audit on 2026-06-04/09: straight/invalid or center-road placement.
    return [
        "005cfc",
        "51c1ee",
        "87eb02",
        "282c9a",
        "bb9632",
        "c6ceba",
        "c943bd",
        "d85a67",
        "e9c518",
        "d0a9cf",
    ]


def ccs_scene_blacklist():
    # CCS'25 dataset_processing/config.py SCENES_TO_REMOVE, plus zero-padding guard.
    return [
        "scene-0329",
        "scene-0907",
        "scene-0908",
        "scene-0557",
        "scene-0560",
        "scene-0561",
        "scene-0632",
        "scene-0109",
        "scene-0784",
        "scene-907",
    ]


def summarize_rows(name, rows):
    return {
        "set": name,
        "frames": len(rows),
        "scenes": len({row.get("scene_name", "") for row in rows}),
        "median_vpa": percentile([as_float(row.get("vpa_coverage")) for row in rows], 50),
        "median_selection_score": percentile([as_float(row.get("phase1_8_selection_score")) for row in rows], 50),
        "median_curvature_diff": percentile([as_float(row.get("max_curvature_diff")) for row in rows], 50),
        "median_lane_width_gain": percentile([as_float(row.get("lane_width_max_gain")) for row in rows], 50),
        "median_diverge_turn_deg": percentile([as_float(row.get("diverge_direction_change_deg")) for row in rows], 50),
    }


def select_scene_diverse(rows, max_per_scene, target_max):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_name"]].append(row)
    selected = []
    for scene_name in sorted(by_scene):
        scene_rows = by_scene[scene_name]
        scene_rows.sort(
            key=lambda row: (
                -as_float(row["phase1_8_selection_score"], 0.0),
                int(as_float(row.get("scene_pos"), 0) or 0),
                row["sample_token"],
            )
        )
        selected.extend(scene_rows[:max_per_scene])
    selected.sort(key=lambda row: (-as_float(row["phase1_8_selection_score"], 0.0), row["scene_name"], row["sample_token"]))
    return selected[:target_max] if target_max > 0 else selected


def main():
    parser = argparse.ArgumentParser(description="Build Phase 1.8-A newsplit CCS-style candidate pool.")
    parser.add_argument(
        "--asset-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv",
    )
    parser.add_argument(
        "--vpa-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv",
    )
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild")
    parser.add_argument("--min-vpa", type=float, default=0.05)
    parser.add_argument("--preferred-vpa", type=float, default=0.15)
    parser.add_argument("--max-per-scene", type=int, default=5)
    parser.add_argument("--target-max", type=int, default=80)
    parser.add_argument("--manual-blacklist-prefixes", default="")
    parser.add_argument("--scene-blacklist", default="")
    args = parser.parse_args()

    manual_prefixes = sorted(
        set(token_prefix_blacklist() + [p.strip() for p in args.manual_blacklist_prefixes.split(",") if p.strip()])
    )
    scene_blacklist = sorted(
        set(ccs_scene_blacklist() + [s.strip() for s in args.scene_blacklist.split(",") if s.strip()])
    )

    assets = read_csv(args.asset_csv)
    vpa_rows = vpa_index(read_csv(args.vpa_csv))
    rows = []
    for asset in assets:
        token = asset["sample_token"]
        left, right = load_boundaries(asset["scene_json"])

        dist_pass, init_dist, width_gain, left_aligned, right_aligned = lane_width_gate(left, right)
        min_len = min(len(left_aligned), len(right_aligned))
        length_gate = min_len >= 5
        left_trim = left_aligned[:min_len] if length_gate else left_aligned
        right_trim = right_aligned[:min_len] if length_gate else right_aligned

        heading_sym = heading_tail_symmetric(left_trim, right_trim) if length_gate else False
        diverge_tag, tag_conf, left_score, right_score = identify_diverging_boundary(left_trim, right_trim)
        diverge = left_trim if diverge_tag == "left" else right_trim
        reference = right_trim if diverge_tag == "left" else left_trim

        left_turn_deg = direction_change_score(left_trim)
        right_turn_deg = direction_change_score(right_trim)
        diverge_turn_deg = direction_change_score(diverge)
        reference_turn_deg = direction_change_score(reference)
        diverge_cont_turn = continuous_turning(diverge)
        reference_cont_turn = continuous_turning(reference)

        if len(diverge) >= 5 and len(reference) >= 5:
            diverge_curv = calculate_curvature(diverge)
            reference_curv = calculate_curvature(reference)
            curv_diff = diverge_curv - reference_curv
            reference_region = calculate_region_curvature(reference, 5)
            max_curv_diff = float(np.max(curv_diff))
            max_reference_region = float(np.max(reference_region)) if len(reference_region) else 0.0
            large_diff_idx = np.where(curv_diff > 0.1)[0]
            points_to_check = diverge[large_diff_idx]
            mean_diverge_curvature = float(np.mean(diverge_curv))
            mean_reference_curvature = float(np.mean(reference_curv))
        else:
            max_curv_diff = 0.0
            max_reference_region = math.nan
            points_to_check = np.zeros((0, 2), dtype=np.float64)
            mean_diverge_curvature = math.nan
            mean_reference_curvature = math.nan

        curvature_pass = bool(max_curv_diff > 0.1 and max_reference_region < 0.3)
        point_dist_pass, closest_large_diff, min_y_large_diff = check_point_distances(points_to_check)
        min_dist_to_diverge = float(np.min(np.linalg.norm(diverge, axis=1))) if len(diverge) else math.nan
        diverge_near_ego_pass = bool(min_dist_to_diverge <= 10)

        straight_like = bool(
            max(left_turn_deg, right_turn_deg) < 12.0
            and max(mean_diverge_curvature if not math.isnan(mean_diverge_curvature) else 0.0, 0.0) < 0.05
            and width_gain < 8.0
        )
        weak_asymmetry = bool(max_curv_diff <= 0.1 and width_gain < 8.0)

        diverge_vpa = vpa_rows[token].get("diverge_boundary")
        reference_vpa = vpa_rows[token].get("reference_boundary")
        vpa_coverage = as_float(diverge_vpa.get("vpa_point_coverage") if diverge_vpa else None, 0.0)
        diverge_vpa_pass = bool(diverge_vpa and as_bool(diverge_vpa.get("vpa_pass")))
        reference_vpa_pass = bool(reference_vpa and as_bool(reference_vpa.get("vpa_pass")))
        vpa_gate = bool(diverge_vpa_pass and not reference_vpa_pass and vpa_coverage >= args.min_vpa)
        preferred_vpa_gate = bool(diverge_vpa_pass and not reference_vpa_pass and vpa_coverage >= args.preferred_vpa)

        manual_blacklist = prefix_blacklisted(token, manual_prefixes)
        ccs_scene_blocked = scene_blacklisted(asset["scene_name"], scene_blacklist)
        ccs_geometry_pass = bool(
            length_gate
            and dist_pass
            and not heading_sym
            and curvature_pass
            and point_dist_pass
            and diverge_near_ego_pass
        )
        phase18_pass = bool(
            ccs_geometry_pass
            and vpa_gate
            and not straight_like
            and not weak_asymmetry
            and not manual_blacklist
            and not ccs_scene_blocked
        )

        selection_score = (
            2.0 * vpa_coverage
            + 0.4 * max(width_gain, 0.0)
            + 4.0 * max(max_curv_diff, 0.0)
            + 0.25 * tag_conf
            + 0.02 * max(diverge_turn_deg - reference_turn_deg, 0.0)
            + 0.5 * diverge_cont_turn
        )

        fail_reasons = []
        for name, passed in [
            ("length", length_gate),
            ("lane_width", dist_pass),
            ("not_heading_symmetric", not heading_sym),
            ("curvature", curvature_pass),
            ("point_distance", point_dist_pass),
            ("diverge_near_ego", diverge_near_ego_pass),
            ("vpa", vpa_gate),
            ("not_straight_like", not straight_like),
            ("not_weak_asymmetry", not weak_asymmetry),
            ("not_manual_blacklist", not manual_blacklist),
            ("not_ccs_scene_blacklist", not ccs_scene_blocked),
        ]:
            if not passed:
                fail_reasons.append(name)

        out = dict(asset)
        out.update(
            {
                "length_gate": bool_text(length_gate),
                "lane_width_gate": bool_text(dist_pass),
                "lane_width_init_dist": init_dist,
                "lane_width_max_gain": width_gain,
                "heading_tail_symmetric": bool_text(heading_sym),
                "phase1_8_diverge_boundary_tag": diverge_tag,
                "phase1_8_tag_confidence": tag_conf,
                "left_score": left_score,
                "right_score": right_score,
                "left_direction_change_deg": left_turn_deg,
                "right_direction_change_deg": right_turn_deg,
                "diverge_direction_change_deg": diverge_turn_deg,
                "reference_direction_change_deg": reference_turn_deg,
                "diverge_continuous_turn": diverge_cont_turn,
                "reference_continuous_turn": reference_cont_turn,
                "mean_diverge_curvature": mean_diverge_curvature,
                "mean_reference_curvature": mean_reference_curvature,
                "max_curvature_diff": max_curv_diff,
                "max_reference_region_curvature": max_reference_region,
                "curvature_gate": bool_text(curvature_pass),
                "point_distance_gate": bool_text(point_dist_pass),
                "closest_large_curvature_point_dist": closest_large_diff,
                "min_y_large_curvature_point": min_y_large_diff,
                "min_dist_to_diverge_boundary": min_dist_to_diverge,
                "diverge_near_ego_gate": bool_text(diverge_near_ego_pass),
                "vpa_coverage": vpa_coverage,
                "diverge_vpa_pass": bool_text(diverge_vpa_pass),
                "reference_vpa_pass": bool_text(reference_vpa_pass),
                "vpa_gate": bool_text(vpa_gate),
                "preferred_vpa_gate": bool_text(preferred_vpa_gate),
                "straight_like_gate_fail": bool_text(straight_like),
                "weak_asymmetry_gate_fail": bool_text(weak_asymmetry),
                "manual_blacklist": bool_text(manual_blacklist),
                "ccs_scene_blacklist": bool_text(ccs_scene_blocked),
                "ccs_geometry_pass": bool_text(ccs_geometry_pass),
                "phase1_8_pass": bool_text(phase18_pass),
                "phase1_8_selection_score": selection_score,
                "fail_reasons": ";".join(fail_reasons),
            }
        )
        rows.append(out)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []

    pass_rows = [row for row in rows if as_bool(row["phase1_8_pass"])]
    selected = select_scene_diverse(pass_rows, args.max_per_scene, args.target_max)
    rejected = [row for row in rows if not as_bool(row["phase1_8_pass"])]
    straight_excluded = [
        row
        for row in rows
        if as_bool(row["straight_like_gate_fail"]) or as_bool(row["weak_asymmetry_gate_fail"]) or as_bool(row["manual_blacklist"])
    ]

    scene_rows = []
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_name"]].append(row)
    selected_tokens = {row["sample_token"] for row in selected}
    for scene_name, scene_rows_all in sorted(by_scene.items()):
        pass_scene = [row for row in scene_rows_all if as_bool(row["phase1_8_pass"])]
        selected_scene = [row for row in scene_rows_all if row["sample_token"] in selected_tokens]
        scene_rows.append(
            {
                "scene_name": scene_name,
                "all_frames": len(scene_rows_all),
                "phase1_8_pass_frames": len(pass_scene),
                "selected_frames": len(selected_scene),
                "median_selection_score": percentile([as_float(row["phase1_8_selection_score"]) for row in pass_scene], 50),
                "max_selection_score": percentile([as_float(row["phase1_8_selection_score"]) for row in pass_scene], 100),
            }
        )

    summary_rows = [
        summarize_rows("all_assets", rows),
        summarize_rows("phase1_8_pass_pool", pass_rows),
        summarize_rows("phase1_8_selected_pool", selected),
        summarize_rows("straight_or_invalid_excluded", straight_excluded),
    ]
    fail_counts = Counter()
    for row in rejected:
        for reason in row["fail_reasons"].split(";"):
            if reason:
                fail_counts[reason] += 1

    write_csv(out_dir / "geometry_quality_v2_table.csv", rows, fieldnames)
    write_csv(out_dir / "phase1_8_candidate_pool.csv", pass_rows, fieldnames)
    write_csv(out_dir / "phase1_8_selected_candidates.csv", selected, fieldnames)
    write_csv(out_dir / "phase1_8_rejected_candidates.csv", rejected, fieldnames)
    write_csv(out_dir / "phase1_8_excluded_straight_cases.csv", straight_excluded, fieldnames)
    write_csv(out_dir / "phase1_8_scene_coverage.csv", scene_rows)
    write_csv(out_dir / "phase1_8_geometry_summary.csv", summary_rows)
    write_tokens(out_dir / "phase1_8_candidate_pool_tokens.txt", pass_rows)
    write_tokens(out_dir / "phase1_8_selected_tokens.txt", selected)

    summary = {
        "asset_csv": args.asset_csv,
        "vpa_csv": args.vpa_csv,
        "min_vpa": args.min_vpa,
        "preferred_vpa": args.preferred_vpa,
        "max_per_scene": args.max_per_scene,
        "target_max": args.target_max,
        "manual_blacklist_prefixes": manual_prefixes,
        "scene_blacklist": scene_blacklist,
        "sets": summary_rows,
        "fail_counts": dict(sorted(fail_counts.items())),
        "outputs": {
            "geometry_quality_v2_table": str(out_dir / "geometry_quality_v2_table.csv"),
            "candidate_pool": str(out_dir / "phase1_8_candidate_pool.csv"),
            "selected_candidates": str(out_dir / "phase1_8_selected_candidates.csv"),
            "selected_tokens": str(out_dir / "phase1_8_selected_tokens.txt"),
            "scene_coverage": str(out_dir / "phase1_8_scene_coverage.csv"),
            "excluded_straight_cases": str(out_dir / "phase1_8_excluded_straight_cases.csv"),
        },
    }
    (out_dir / "phase1_8_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
