#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from build_ccs_dense_location_candidates import (
    calculate_combined_score,
    get_asymmetry_anchors,
    load_boundaries,
    write_csv,
)
from ccs_patch_utils import (
    create_pseudo_area,
    generate_sampled_points,
    get_patch_heading_facing_ego,
    get_proj_scale,
    sample_boundary_at_interval,
)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_by_token(clean_ann):
    samples = load_pickle(clean_ann)
    return {sample["token"]: sample for sample in samples}


def build_patch_candidates(asset, sample, args, rng):
    diverge_tag, diverge, reference = load_boundaries(asset["scene_json"])
    anchors = get_asymmetry_anchors(
        diverge,
        reference,
        threshold=args.curvature_diff_threshold,
        top_k=args.anchor_topk,
    )

    cam0 = next(iter(sample["cams"].values()))
    import cv2

    img = cv2.imread(cam0["img_fpath"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(cam0["img_fpath"])
    ori_h, ori_w = img.shape[:2]
    ori_img_shape = (3, ori_h, ori_w)

    dense_locs = sample_boundary_at_interval(diverge, interval=args.sample_interval)
    patch_centers = []
    patch_headings = []
    patch_sources = []
    is_left = diverge_tag == "left"

    for loc in dense_locs:
        patch_center = np.asarray(loc, dtype=np.float64).copy()
        patch_center[0] = (
            patch_center[0] - args.patch_width / 2.0
            if is_left
            else patch_center[0] + args.patch_width / 2.0
        )
        patch_cfg = {
            "type": args.patch_type,
            "lat": float(patch_center[0]),
            "long": float(patch_center[1]),
            "width": args.patch_width,
            "height": args.patch_height,
            "heading": get_patch_heading_facing_ego(patch_center),
            "lidar2vehfront": 0.94,
            "lidar2ground": 1.84,
        }
        proj_scale = get_proj_scale(patch_cfg["lat"], patch_cfg["long"], ori_w)
        pseudo_area = create_pseudo_area(patch_cfg, ori_img_shape, proj_scale)
        if pseudo_area is None:
            continue
        patch_centers.append(patch_center)
        patch_headings.append(patch_cfg["heading"])
        patch_sources.append("boundary_offset")

    if args.sample:
        sampled_locs = max(args.samples_per_loc - 1, 0)
        for patch_center, base_heading in zip(list(patch_centers), list(patch_headings)):
            points = generate_sampled_points(
                patch_center,
                grid_size=args.sample_range,
                num_points=sampled_locs,
                mode=diverge_tag,
            )
            offset_range = np.radians(args.heading_jitter_deg)
            heading_offsets = rng.random(len(points)) * offset_range * 2.0 - offset_range
            for point, heading_offset in zip(points, heading_offsets):
                patch_centers.append(point)
                patch_headings.append(float(base_heading + heading_offset))
                patch_sources.append("local_random")

    rows = []
    for idx, (patch_center, patch_heading, source) in enumerate(
        zip(patch_centers, patch_headings, patch_sources)
    ):
        patch_center_3d = np.asarray(
            [patch_center[0], patch_center[1], -1.84 + args.patch_height / 2.0],
            dtype=np.float64,
        )
        score = calculate_combined_score(
            patch_center_3d,
            sample,
            anchors,
            max_beam_angle=np.radians(args.max_beam_angle_deg),
        )
        rows.append(
            {
                "sample_token": asset["sample_token"],
                "scene_name": asset.get("scene_name", ""),
                "scene_pos": asset.get("scene_pos", ""),
                "diverge_boundary_tag": diverge_tag,
                "candidate_source": source,
                "candidate_loc_idx": idx,
                "x": float(patch_center_3d[0]),
                "y": float(patch_center_3d[1]),
                "z": float(patch_center_3d[2]),
                "patch_heading": float(patch_heading),
                "patch_width": args.patch_width,
                "patch_height": args.patch_height,
                "patch_type": args.patch_type,
                "geometric_score": float(score),
                "num_dense_boundary_locs": len(dense_locs),
                "num_valid_patch_centers": len(patch_centers),
                "num_anchors": len(anchors),
                "scene_json": asset["scene_json"],
            }
        )

    rows.sort(key=lambda row: row["geometric_score"], reverse=True)
    total_locs = args.total_locs // args.step_per_loc
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["is_topk"] = rank <= total_locs
    return rows[:total_locs], rows


def main():
    parser = argparse.ArgumentParser(
        description="Build CCS patch-location candidates using the original patch branch geometry rules."
    )
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--clean-ann-root", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--all-candidates-out", default="")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--total-locs", type=int, default=400)
    parser.add_argument("--step-per-loc", type=int, default=20)
    parser.add_argument("--sample", action="store_true", default=True)
    parser.add_argument("--samples-per-loc", type=int, default=2)
    parser.add_argument("--sample-range", type=float, default=1.0)
    parser.add_argument("--heading-jitter-deg", type=float, default=30.0)
    parser.add_argument("--patch-type", choices=["vertical"], default="vertical")
    parser.add_argument("--patch-width", type=float, default=3.0)
    parser.add_argument("--patch-height", type=float, default=2.0)
    parser.add_argument("--max-beam-angle-deg", type=float, default=20.0)
    parser.add_argument("--curvature-diff-threshold", type=float, default=0.1)
    parser.add_argument("--anchor-topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    assets_by_token = {row["sample_token"]: row for row in read_csv(args.asset_csv)}
    tokens = [line.strip() for line in Path(args.tokens).read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(args.seed)

    selected_rows = []
    all_rows = []
    for token in tokens:
        asset = assets_by_token[token]
        clean_ann = Path(args.clean_ann_root) / token / "anns" / "clean_sequence_ann.pkl"
        samples = sample_by_token(clean_ann)
        if token not in samples:
            raise KeyError(f"target token {token} not found in {clean_ann}")
        top_rows, token_rows = build_patch_candidates(asset, samples[token], args, rng)
        selected_rows.extend(top_rows)
        all_rows.extend(token_rows)

    write_csv(Path(args.out_csv), selected_rows)
    if args.all_candidates_out:
        write_csv(Path(args.all_candidates_out), all_rows)

    summary = {
        "tokens": len(tokens),
        "selected_candidates": len(selected_rows),
        "all_candidates": len(all_rows),
        "top_locs_per_token": args.total_locs // args.step_per_loc,
        "sample_interval": args.sample_interval,
        "step_per_loc": args.step_per_loc,
        "patch_width": args.patch_width,
        "patch_height": args.patch_height,
        "out_csv": args.out_csv,
        "all_candidates_out": args.all_candidates_out,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
