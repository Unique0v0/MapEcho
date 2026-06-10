#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PADDING_VALUE = -10000


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


def as_float(value, default=None):
    if value in ("", None):
        return default
    return float(value)


def as_bool(value):
    return str(value).strip().lower() == "true"


def bool_text(value):
    return "true" if value else "false"


def calculate_curvature(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    dx, dy = np.gradient(points[:, 0]), np.gradient(points[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    numerator = np.abs(ddx * dy - ddy * dx)
    denominator = (dx**2 + dy**2) ** 1.5
    out = np.zeros_like(numerator)
    valid = denominator > 1e-12
    out[valid] = numerator[valid] / denominator[valid]
    return out


def calculate_region_curvature(points, region_size=5):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < region_size:
        return np.zeros(len(points), dtype=np.float64)
    values = []
    for i in range(len(points) - region_size + 1):
        values.append(float(np.mean(calculate_curvature(points[i : i + region_size]))))
    return np.asarray(values, dtype=np.float64)


def compute_heading(points):
    diffs = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    return np.arctan2(diffs[:, 1], diffs[:, 0])


def average_heading(points):
    if len(points) < 2:
        return None
    headings = compute_heading(points)
    if len(headings) == 0:
        return None
    x_sum = np.sum(np.cos(headings))
    y_sum = np.sum(np.sin(headings))
    if abs(x_sum) <= 1e-12 and abs(y_sum) <= 1e-12:
        return None
    return float(np.arctan2(y_sum, x_sum))


def angle_diff(a, b):
    diff = abs(a - b) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)


def heading_tail_symmetric(left, right, angle_thresh_rad=np.radians(30), tail_len=3):
    left_heading = average_heading(left[-tail_len:])
    right_heading = average_heading(right[-tail_len:])
    if left_heading is None or right_heading is None:
        return False
    return abs(angle_diff(left_heading, right_heading) - np.pi) < angle_thresh_rad


def direction_change(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    vectors = points[1:] - points[:-1]
    dirs = []
    for vec in vectors:
        norm = np.linalg.norm(vec)
        dirs.append(vec / norm if norm > 1e-6 else np.zeros(2))
    dirs = np.asarray(dirs)
    start = dirs[:3].mean(axis=0)
    end = dirs[-3:].mean(axis=0)
    if np.linalg.norm(start) > 1e-6:
        start = start / np.linalg.norm(start)
    if np.linalg.norm(end) > 1e-6:
        end = end / np.linalg.norm(end)
    return float(np.degrees(np.arccos(np.clip(np.dot(start, end), -1.0, 1.0))) / 180.0)


def continuous_turning(points, window_size=5):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < window_size + 1:
        return 0.0
    vectors = points[1:] - points[:-1]
    scores = []
    for i in range(len(vectors) - window_size + 1):
        a = vectors[i]
        b = vectors[i + window_size - 1]
        if np.linalg.norm(a) <= 1e-6 or np.linalg.norm(b) <= 1e-6:
            continue
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        scores.append(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))
    return float(min(max(scores) / 90.0, 1.0)) if scores else 0.0


def identify_diverging_boundary(left, right):
    left_dir = direction_change(left)
    right_dir = direction_change(right)
    left_turn = continuous_turning(left)
    right_turn = continuous_turning(right)
    left_score = 0.5 * left_dir + 0.2 * left_turn
    right_score = 0.5 * right_dir + 0.2 * right_turn
    confidence = abs(left_score - right_score) / max(left_score + right_score, 1e-6)
    tag = "left" if left_score > right_score else "right"
    return tag, confidence, left_score, right_score


def align_forward(left_points, right_points):
    left = np.asarray(left_points, dtype=np.float64)
    right = np.asarray(right_points, dtype=np.float64)

    def check_direction(points, start_idx):
        next_idx = min(start_idx + 1, len(points) - 1)
        prev_idx = max(start_idx - 1, 0)
        forward_vec = points[next_idx] - points[start_idx]
        backward_vec = points[prev_idx] - points[start_idx]
        return np.dot(forward_vec, np.array([0, 1])) > np.dot(backward_vec, np.array([0, 1]))

    left_start = int(np.argmin(np.linalg.norm(left - [0, 0], axis=1)))
    right_start = int(np.argmin(np.linalg.norm(right - [0, 0], axis=1)))
    if not check_direction(left, left_start):
        left = left[::-1]
        left_start = len(left) - 1 - left_start
    if not check_direction(right, right_start):
        right = right[::-1]
        right_start = len(right) - 1 - right_start
    return left[left_start:], right[right_start:]


def lane_width_gate(left, right, threshold=5.0):
    left_aligned, right_aligned = align_forward(left, right)
    min_len = min(len(left_aligned), len(right_aligned))
    if min_len < 2:
        return False, 0.0, 0.0, left_aligned, right_aligned
    init_dist = float(np.linalg.norm(left_aligned[0] - right_aligned[0]))
    distances = np.linalg.norm(left_aligned[:min_len] - right_aligned[:min_len], axis=1)
    max_gain = float(np.max(distances - init_dist))
    return bool(np.any(distances > init_dist + threshold)), init_dist, max_gain, left_aligned, right_aligned


def check_point_distances(points, min_dist_closest=3, max_dist_closest=15, min_y=3):
    if len(points) == 0:
        return False, math.nan, math.nan
    distances = np.linalg.norm(points, axis=1)
    closest = float(np.min(distances))
    min_y_value = float(np.min(points[:, 1]))
    return min_dist_closest <= closest <= max_dist_closest and min_y_value >= min_y, closest, min_y_value


def load_boundaries(scene_json):
    with open(scene_json) as f:
        scene = json.load(f)
    boundaries = {
        item["tag"]: np.asarray(item["coordinates"], dtype=np.float64)
        for item in scene["map_elements"]
    }
    left = boundaries["left"]
    right = boundaries["right"]
    left = left[left[:, 0] != PADDING_VALUE]
    right = right[right[:, 0] != PADDING_VALUE]
    return left[:, :2], right[:, :2]


def vpa_index(rows):
    out = defaultdict(dict)
    for row in rows:
        out[row["sample_id"]][row["target_type"]] = row
    return out


def prefix_blacklisted(token, prefixes):
    return any(token.startswith(prefix) for prefix in prefixes)


def scene_blacklisted(scene_name, scene_names):
    return scene_name in scene_names


def percentile(values, q):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_set(name, rows):
    scenes = {row["scene_name"] for row in rows}
    return {
        "set": name,
        "frames": len(rows),
        "scenes": len(scenes),
        "median_vpa": percentile([as_float(row.get("vpa_coverage")) for row in rows], 50),
        "median_curvature_diff": percentile([as_float(row.get("max_curvature_diff")) for row in rows], 50),
        "median_lane_width_gain": percentile([as_float(row.get("lane_width_max_gain")) for row in rows], 50),
    }


def select_scene_diverse(rows, max_per_scene):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_name"]].append(row)
    selected = []
    for scene_rows in by_scene.values():
        scene_rows.sort(key=lambda row: (-as_float(row["selection_score"], 0.0), row["sample_token"]))
        selected.extend(scene_rows[:max_per_scene])
    selected.sort(key=lambda row: (-as_float(row["selection_score"], 0.0), row["scene_name"], row["sample_token"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Build CCS-style Phase 1.6 candidate pool.")
    parser.add_argument(
        "--asset-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv",
    )
    parser.add_argument(
        "--vpa-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv",
    )
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool")
    parser.add_argument("--min-vpa", type=float, default=0.15)
    parser.add_argument("--max-per-scene", type=int, default=2)
    parser.add_argument("--manual-blacklist-prefixes", default="")
    parser.add_argument("--scene-blacklist", default="")
    args = parser.parse_args()

    default_manual_prefixes = [
        # User visual audit on 2026-06-04: straight/invalid or center-road placement.
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
    extra_prefixes = [p.strip() for p in args.manual_blacklist_prefixes.split(",") if p.strip()]
    manual_prefixes = sorted(set(default_manual_prefixes + extra_prefixes))
    default_scene_blacklist = [
        # CCS'25 dataset_processing/config.py SCENES_TO_REMOVE.
        "scene-0329",
        "scene-0907",
        "scene-0908",
        "scene-0557",
        "scene-0560",
        "scene-0561",
        "scene-0632",
        "scene-0109",
        "scene-0784",
        # Preserve the original non-zero-padded spelling as a guard.
        "scene-907",
    ]
    extra_scene_blacklist = [s.strip() for s in args.scene_blacklist.split(",") if s.strip()]
    scene_blacklist = sorted(set(default_scene_blacklist + extra_scene_blacklist))

    assets = read_csv(args.asset_csv)
    vpa_rows = vpa_index(read_csv(args.vpa_csv))
    rows = []
    for asset in assets:
        token = asset["sample_token"]
        left, right = load_boundaries(asset["scene_json"])

        dist_pass, init_dist, width_gain, left_aligned, right_aligned = lane_width_gate(left, right)
        min_len = min(len(left_aligned), len(right_aligned))
        length_gate = min_len >= 5
        if length_gate:
            left_trim = left_aligned[:min_len]
            right_trim = right_aligned[:min_len]
        else:
            left_trim = left_aligned
            right_trim = right_aligned

        heading_sym = heading_tail_symmetric(left_trim, right_trim) if length_gate else False
        diverge_tag, tag_conf, left_score, right_score = identify_diverging_boundary(left_trim, right_trim)
        diverge = left_trim if diverge_tag == "left" else right_trim
        reference = right_trim if diverge_tag == "left" else left_trim

        if len(diverge) >= 5 and len(reference) >= 5:
            diverge_curv = calculate_curvature(diverge)
            reference_curv = calculate_curvature(reference)
            curv_diff = diverge_curv - reference_curv
            reference_region = calculate_region_curvature(reference, 5)
            max_curv_diff = float(np.max(curv_diff))
            max_reference_region = float(np.max(reference_region)) if len(reference_region) else 0.0
            large_diff_idx = np.where(curv_diff > 0.1)[0]
            points_to_check = diverge[large_diff_idx]
        else:
            max_curv_diff = 0.0
            max_reference_region = math.nan
            points_to_check = np.zeros((0, 2), dtype=np.float64)

        curvature_pass = bool(max_curv_diff > 0.1 and max_reference_region < 0.3)
        point_dist_pass, closest_large_diff, min_y_large_diff = check_point_distances(points_to_check)
        min_dist_to_diverge = float(np.min(np.linalg.norm(diverge, axis=1))) if len(diverge) else math.nan
        diverge_near_ego_pass = bool(min_dist_to_diverge <= 10)

        diverge_vpa = vpa_rows[token].get("diverge_boundary")
        reference_vpa = vpa_rows[token].get("reference_boundary")
        vpa_coverage = as_float(diverge_vpa.get("vpa_point_coverage") if diverge_vpa else None, 0.0)
        diverge_vpa_pass = bool(diverge_vpa and as_bool(diverge_vpa.get("vpa_pass")))
        reference_vpa_pass = bool(reference_vpa and as_bool(reference_vpa.get("vpa_pass")))
        vpa_gate = bool(diverge_vpa_pass and not reference_vpa_pass and vpa_coverage >= args.min_vpa)
        manual_blacklist = prefix_blacklisted(token, manual_prefixes)
        ccs_scene_blacklist = scene_blacklisted(asset["scene_name"], scene_blacklist)

        ccs_rule_pass = bool(
            length_gate
            and dist_pass
            and not heading_sym
            and curvature_pass
            and point_dist_pass
            and diverge_near_ego_pass
        )
        phase16_pass = bool(ccs_rule_pass and vpa_gate and not manual_blacklist and not ccs_scene_blacklist)
        selection_score = (
            2.0 * vpa_coverage
            + 0.5 * max(width_gain, 0.0)
            + 4.0 * max(max_curv_diff, 0.0)
            + 0.5 * tag_conf
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
            ("not_manual_blacklist", not manual_blacklist),
            ("not_ccs_scene_blacklist", not ccs_scene_blacklist),
        ]:
            if not passed:
                fail_reasons.append(name)

        out = dict(asset)
        out.update(
            {
                "manual_blacklist": bool_text(manual_blacklist),
                "ccs_scene_blacklist": bool_text(ccs_scene_blacklist),
                "length_gate": bool_text(length_gate),
                "lane_width_gate": bool_text(dist_pass),
                "lane_width_init_dist": init_dist,
                "lane_width_max_gain": width_gain,
                "heading_tail_symmetric": bool_text(heading_sym),
                "diverge_boundary_id_v16": diverge_tag,
                "tag_confidence_v16": tag_conf,
                "left_score_v16": left_score,
                "right_score_v16": right_score,
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
                "ccs_rule_pass": bool_text(ccs_rule_pass),
                "phase1_6_pass": bool_text(phase16_pass),
                "selection_score": selection_score,
                "fail_reasons": ";".join(fail_reasons),
            }
        )
        rows.append(out)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_dir / "phase1_6_ccs_rule_table.csv", rows, fieldnames)

    ccs_rule = [row for row in rows if as_bool(row["ccs_rule_pass"])]
    vpa_pass = [row for row in rows if as_bool(row["vpa_gate"])]
    final_pool = [row for row in rows if as_bool(row["phase1_6_pass"])]
    final_selected = select_scene_diverse(final_pool, args.max_per_scene)
    rejected = [row for row in rows if not as_bool(row["phase1_6_pass"])]

    write_csv(out_dir / "phase1_6_high_quality_pool_assets.csv", final_pool, fieldnames)
    write_csv(out_dir / "phase1_6_high_quality_selected_assets.csv", final_selected, fieldnames)
    write_csv(out_dir / "phase1_6_rejected_assets.csv", rejected, fieldnames)
    (out_dir / "phase1_6_high_quality_pool_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in final_pool) + ("\n" if final_pool else "")
    )
    (out_dir / "phase1_6_high_quality_selected_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in final_selected) + ("\n" if final_selected else "")
    )

    summary_rows = [
        summarize_set("all_assets", rows),
        summarize_set("ccs_rule_pass", ccs_rule),
        summarize_set("vpa_gate_pass", vpa_pass),
        summarize_set("phase1_6_high_quality_pool", final_pool),
        summarize_set("phase1_6_scene_diverse_selected", final_selected),
    ]
    write_csv(out_dir / "phase1_6_set_summary.csv", summary_rows)

    fail_counts = defaultdict(int)
    for row in rejected:
        for reason in row["fail_reasons"].split(";"):
            if reason:
                fail_counts[reason] += 1

    summary = {
        "asset_csv": args.asset_csv,
        "vpa_csv": args.vpa_csv,
        "min_vpa": args.min_vpa,
        "max_per_scene": args.max_per_scene,
        "manual_blacklist_prefixes": manual_prefixes,
        "scene_blacklist": scene_blacklist,
        "sets": summary_rows,
        "fail_counts": dict(sorted(fail_counts.items())),
        "outputs": {
            "rule_table": str(out_dir / "phase1_6_ccs_rule_table.csv"),
            "high_quality_pool_assets": str(out_dir / "phase1_6_high_quality_pool_assets.csv"),
            "high_quality_selected_assets": str(out_dir / "phase1_6_high_quality_selected_assets.csv"),
            "high_quality_selected_tokens": str(out_dir / "phase1_6_high_quality_selected_tokens.txt"),
        },
    }
    (out_dir / "phase1_6_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
