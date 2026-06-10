#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

from ccs_blind_renderer import camera_pose_in_lidar


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


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_boundaries(scene_json):
    with open(scene_json) as f:
        scene = json.load(f)
    boundaries = {
        item["tag"]: np.asarray(item["coordinates"], dtype=np.float64)
        for item in scene["map_elements"]
    }
    left = boundaries["left"]
    right = boundaries["right"]
    left = left[left[:, 0] != PADDING_VALUE][:, :2]
    right = right[right[:, 0] != PADDING_VALUE][:, :2]
    tag_info = scene.get("diverge_boundary_tag")
    if isinstance(tag_info, (list, tuple)) and tag_info:
        diverge_tag = tag_info[0]
    else:
        raise ValueError(f"missing diverge_boundary_tag in {scene_json}")
    diverge = left if diverge_tag == "left" else right
    reference = right if diverge_tag == "left" else left
    return diverge_tag, diverge, reference


def polyline_lengths(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(1, dtype=np.float64)
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def interpolate_polyline(points, distance):
    points = np.asarray(points, dtype=np.float64)
    lengths = polyline_lengths(points)
    if len(points) == 0:
        raise ValueError("empty polyline")
    if len(points) == 1 or distance <= 0:
        return points[0].copy()
    if distance >= lengths[-1]:
        return points[-1].copy()
    idx = int(np.searchsorted(lengths, distance, side="right") - 1)
    denom = lengths[idx + 1] - lengths[idx]
    if denom <= 1e-12:
        return points[idx].copy()
    alpha = (distance - lengths[idx]) / denom
    return points[idx] * (1.0 - alpha) + points[idx + 1] * alpha


def sample_boundary_at_interval(boundary_pts, interval=0.5):
    lengths = polyline_lengths(boundary_pts)
    total_length = float(lengths[-1])
    num_points = max(2, int(total_length / interval) + 1)
    sampled = []
    for i in range(num_points):
        distance = i * interval
        if distance > total_length:
            break
        sampled.append(interpolate_polyline(boundary_pts, distance))
    return np.asarray(sampled, dtype=np.float64)


def generate_sampled_points(center, grid_size=1.0, num_points=1, rng=None):
    rng = rng if rng is not None else np.random
    x = rng.random(num_points) * grid_size - grid_size / 2 + center[0]
    y = rng.random(num_points) * grid_size - grid_size / 2 + center[1]
    return np.stack([x, y], axis=1)


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


def get_asymmetry_anchors(diverge_boundary_pts, reference_boundary_pts, threshold=0.1, top_k=5):
    min_length = min(len(diverge_boundary_pts), len(reference_boundary_pts))
    diverge_curvatures = calculate_curvature(diverge_boundary_pts[:min_length])
    reference_curvatures = calculate_curvature(reference_boundary_pts[:min_length])
    curvature_diff = np.abs(diverge_curvatures - reference_curvatures)
    large_diff_indices = np.where(curvature_diff > threshold)[0]
    if len(large_diff_indices) == 0:
        top_indices = np.argsort(curvature_diff)[::-1][:top_k]
        anchors = diverge_boundary_pts[top_indices]
    else:
        anchors = diverge_boundary_pts[large_diff_indices]
    z = np.ones((anchors.shape[0], 1), dtype=np.float64) * -1.84
    return np.hstack([anchors, z])


def calculate_combined_score(point, sample, divergent_points, max_beam_angle=np.radians(40)):
    camera_positions = [
        camera_pose_in_lidar(sample, cam_name)["position"]
        for cam_name in sample["cams"]
    ]
    total_score = 0.0
    for cam_pos in camera_positions:
        cam_to_point = point - cam_pos
        cam_to_point_dist = float(np.linalg.norm(cam_to_point))
        if cam_to_point_dist <= 1e-12:
            continue
        cam_to_point_dir = cam_to_point / cam_to_point_dist
        for div_point in divergent_points:
            cam_to_div = div_point - cam_pos
            cam_to_div_dist = float(np.linalg.norm(cam_to_div))
            if cam_to_div_dist <= 1e-12:
                continue
            cam_to_div_dir = cam_to_div / cam_to_div_dist
            angle = math.acos(float(np.clip(np.dot(cam_to_point_dir, cam_to_div_dir), -1.0, 1.0)))
            if angle < max_beam_angle:
                angle_score = 1.0 - (angle / max_beam_angle)
                distance_score = 1.0 / (cam_to_point_dist**2)
                total_score += angle_score * distance_score
    return float(total_score)


def build_candidates(asset, sample, args, rng):
    diverge_tag, diverge, reference = load_boundaries(asset["scene_json"])
    anchors = get_asymmetry_anchors(
        diverge,
        reference,
        threshold=args.curvature_diff_threshold,
        top_k=args.anchor_topk,
    )

    dense_locs = sample_boundary_at_interval(diverge, interval=args.sample_interval)
    all_locs = dense_locs.copy()
    if args.sample:
        local_count = max(args.samples_per_loc - 1, 0)
        if local_count:
            local = [
                generate_sampled_points(loc, args.sample_range, local_count, rng)
                for loc in dense_locs
            ]
            all_locs = np.vstack([all_locs] + local)

    rows = []
    heights = np.linspace(-1.84, 0.0, args.locs_height_num)
    for loc_idx, loc in enumerate(all_locs):
        source = "boundary" if loc_idx < len(dense_locs) else "local_random"
        for height_idx, height in enumerate(heights):
            point = np.array([loc[0], loc[1], height], dtype=np.float64)
            score = calculate_combined_score(
                point,
                sample,
                anchors,
                max_beam_angle=np.radians(args.max_beam_angle_deg),
            )
            rows.append(
                {
                    "sample_token": asset["sample_token"],
                    "scene_name": asset["scene_name"],
                    "scene_pos": asset.get("scene_pos", ""),
                    "diverge_boundary_tag": diverge_tag,
                    "candidate_source": source,
                    "candidate_loc_idx": loc_idx,
                    "height_idx": height_idx,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "geometric_score": score,
                    "num_dense_boundary_locs": len(dense_locs),
                    "num_all_xy_locs": len(all_locs),
                    "num_anchors": len(anchors),
                    "scene_json": asset["scene_json"],
                }
            )
    rows.sort(key=lambda row: row["geometric_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["is_topk"] = rank <= args.total_locs
    return rows[: args.total_locs], rows


def main():
    parser = argparse.ArgumentParser(
        description="Build CCS-style dense glare-source location candidates and geometric ranking."
    )
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--stream-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokens", default="")
    parser.add_argument("--total-locs", type=int, default=400)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--locs-height-num", type=int, default=4)
    parser.add_argument("--sample", action="store_true", default=True)
    parser.add_argument("--no-sample", dest="sample", action="store_false")
    parser.add_argument("--samples-per-loc", type=int, default=2)
    parser.add_argument("--sample-range", type=float, default=1.0)
    parser.add_argument("--max-beam-angle-deg", type=float, default=40.0)
    parser.add_argument("--curvature-diff-threshold", type=float, default=0.1)
    parser.add_argument("--anchor-topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = read_csv(args.asset_csv)
    if args.tokens:
        token_set = {line.strip() for line in Path(args.tokens).read_text().splitlines() if line.strip()}
        assets = [row for row in assets if row["sample_token"] in token_set]
    stream_infos = load_pickle(args.stream_ann)
    samples = {sample["token"]: sample for sample in stream_infos}
    rng = np.random.default_rng(args.seed)

    top_rows = []
    selected_rows = []
    summary_rows = []
    for asset in assets:
        token = asset["sample_token"]
        if token not in samples:
            raise KeyError(f"{token} not found in {args.stream_ann}")
        topk, all_rows = build_candidates(asset, samples[token], args, rng)
        top_rows.extend(topk)
        best = topk[0]
        selected = dict(asset)
        selected.update(
            {
                "has_blind_eta_loc": True,
                "blind_eta_x": best["x"],
                "blind_eta_y": best["y"],
                "blind_eta_z": best["z"],
                "mapecho_loc_method": "ccs_dense_geometric_top1_unscored",
                "ccs_dense_geometric_score": best["geometric_score"],
                "ccs_dense_num_top_locations": len(topk),
                "ccs_dense_num_all_candidates": len(all_rows),
                "ccs_dense_num_anchors": best["num_anchors"],
            }
        )
        selected_rows.append(selected)
        summary_rows.append(
            {
                "sample_token": token,
                "scene_name": asset["scene_name"],
                "num_all_candidates": len(all_rows),
                "num_top_locations": len(topk),
                "best_x": best["x"],
                "best_y": best["y"],
                "best_z": best["z"],
                "best_geometric_score": best["geometric_score"],
                "num_dense_boundary_locs": best["num_dense_boundary_locs"],
                "num_all_xy_locs": best["num_all_xy_locs"],
                "num_anchors": best["num_anchors"],
            }
        )

    top_fields = [
        "sample_token",
        "scene_name",
        "scene_pos",
        "diverge_boundary_tag",
        "rank",
        "is_topk",
        "candidate_source",
        "candidate_loc_idx",
        "height_idx",
        "x",
        "y",
        "z",
        "geometric_score",
        "num_dense_boundary_locs",
        "num_all_xy_locs",
        "num_anchors",
        "scene_json",
    ]
    write_csv(out_dir / "ccs_dense_top_locations.csv", top_rows, top_fields)
    write_csv(out_dir / "ccs_dense_selected_top1_assets_unscored.csv", selected_rows)
    write_csv(out_dir / "ccs_dense_location_summary.csv", summary_rows)
    report = {
        "asset_csv": args.asset_csv,
        "stream_ann": args.stream_ann,
        "tokens": args.tokens,
        "out_dir": str(out_dir),
        "num_samples": len(selected_rows),
        "num_scenes": len({row["scene_name"] for row in selected_rows}),
        "total_locs": args.total_locs,
        "sample_interval": args.sample_interval,
        "locs_height_num": args.locs_height_num,
        "sample": args.sample,
        "samples_per_loc": args.samples_per_loc,
        "sample_range": args.sample_range,
        "max_beam_angle_deg": args.max_beam_angle_deg,
        "seed": args.seed,
        "outputs": {
            "top_locations": str(out_dir / "ccs_dense_top_locations.csv"),
            "selected_top1_assets_unscored": str(out_dir / "ccs_dense_selected_top1_assets_unscored.csv"),
            "summary": str(out_dir / "ccs_dense_location_summary.csv"),
        },
        "note": (
            "These are CCS-style dense candidates ranked by geometric feasibility only. "
            "They are not final model-scored selected locations."
        ),
    }
    (out_dir / "ccs_dense_location_generation_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
