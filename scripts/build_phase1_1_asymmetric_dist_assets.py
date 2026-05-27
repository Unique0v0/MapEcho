#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
import shutil
from pathlib import Path

import numpy as np


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def calculate_direction_change(points):
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
    cos_angle = float(np.clip(np.dot(start, end), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)) / 180.0)


def calculate_continuous_turning(points, window_size=5):
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
        scores.append(np.degrees(np.arccos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))
    return float(min(max(scores) / 90.0, 1.0)) if scores else 0.0


def identify_diverging_boundary(left_points, right_points):
    left_dir = calculate_direction_change(left_points)
    right_dir = calculate_direction_change(right_points)
    left_turn = calculate_continuous_turning(left_points)
    right_turn = calculate_continuous_turning(right_points)
    left_score = 0.5 * left_dir + 0.2 * left_turn
    right_score = 0.5 * right_dir + 0.2 * right_turn
    confidence = abs(left_score - right_score) / max(left_score + right_score, 1e-6)
    return (
        "left" if left_score > right_score else "right",
        float(confidence),
        float(left_score),
        float(right_score),
    )


def point_to_polyline_distance(point, polyline):
    point = np.asarray(point, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    best = math.inf
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            dist = float(np.linalg.norm(point - start))
        else:
            t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
            dist = float(np.linalg.norm(point - (start + t * segment)))
        best = min(best, dist)
    return best


def choose_anchor(diverge, reference, min_y):
    diverge = np.asarray(diverge, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    min_len = min(len(diverge), len(reference))
    candidates = []
    for idx in range(min_len):
        point = diverge[idx]
        if point[1] < min_y:
            continue
        reference_dist = point_to_polyline_distance(point, reference)
        ego_dist = float(np.linalg.norm(point))
        if ego_dist < 1.5:
            continue
        # Prefer points that separate strongly from the reference while staying near the ego.
        score = reference_dist - 0.03 * ego_dist
        candidates.append((score, idx, reference_dist, ego_dist))
    if not candidates:
        idx = int(np.argmax(diverge[:min_len, 1])) if min_len else 0
        return idx, 0.0, float(np.linalg.norm(diverge[idx]))
    score, idx, reference_dist, ego_dist = max(candidates, key=lambda item: item[0])
    return idx, reference_dist, ego_dist


def scene_index(infos):
    by_token = {sample["token"]: sample for sample in infos}
    by_scene = {}
    for sample in infos:
        by_scene.setdefault(sample["scene_name"], []).append(sample)
    for samples in by_scene.values():
        samples.sort(key=lambda sample: sample["sample_idx"])
    pos = {}
    for scene_name, samples in by_scene.items():
        for i, sample in enumerate(samples):
            pos[sample["token"]] = (i, len(samples))
    return by_token, pos


def main():
    parser = argparse.ArgumentParser(
        description="Build Phase 1.1 asymmetric_dist boundary tags and ETA-like assets."
    )
    parser.add_argument("--membership-csv", default="/data/dj/MapEcho/artifacts/newsplit_candidates/ccs_stage_newsplit_membership.csv")
    parser.add_argument("--newsplit-val-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl")
    parser.add_argument("--source-stage", default="ccs_asymmetric_dist")
    parser.add_argument("--eligibility-key", default="eligible_W10_L9")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist")
    parser.add_argument("--light-z", type=float, default=-0.6133333333333333)
    parser.add_argument("--min-anchor-y", type=float, default=3.0)
    args = parser.parse_args()

    infos = load_pickle(args.newsplit_val_ann)
    by_token, pos_by_token = scene_index(infos)
    rows = read_csv(args.membership_csv)
    selected = [
        row
        for row in rows
        if row["stage"] == args.source_stage
        and row["newsplit_split"] == "new_val"
        and str(row.get(args.eligibility_key, "")).lower() == "true"
    ]

    out_dir = Path(args.out_dir)
    scene_dir = out_dir / "scene_json"
    scene_dir.mkdir(parents=True, exist_ok=True)
    tag_rows = []
    asset_rows = []
    for row in selected:
        token = row["sample_token"]
        sample = by_token[token]
        with open(row["scene_json"]) as f:
            scene = json.load(f)
        boundaries = {element["tag"]: np.asarray(element["coordinates"], dtype=np.float64) for element in scene["map_elements"]}
        left = boundaries["left"]
        right = boundaries["right"]
        tag, confidence, left_score, right_score = identify_diverging_boundary(left, right)
        diverge = left if tag == "left" else right
        reference = right if tag == "left" else left
        anchor_idx, reference_dist, ego_dist = choose_anchor(diverge, reference, args.min_anchor_y)
        anchor = diverge[anchor_idx]
        scene["diverge_boundary_tag"] = [tag, confidence, left_score, right_score]
        scene["diverge_points"] = [anchor.tolist()]
        scene["mapecho_generation"] = {
            "source_stage": args.source_stage,
            "tag_source": "mapecho_geometry_heuristic",
            "anchor_idx": int(anchor_idx),
            "anchor_reference_distance_m": reference_dist,
            "anchor_ego_distance_m": ego_dist,
        }
        scene_json_out = scene_dir / f"{token}.json"
        with scene_json_out.open("w") as f:
            json.dump(scene, f, indent=2)

        scene_pos, scene_len = pos_by_token[token]
        tag_rows.append(
            {
                "sample_token": token,
                "scene_name": sample["scene_name"],
                "scene_pos": scene_pos,
                "scene_len": scene_len,
                "boundary_left_id": "left",
                "boundary_right_id": "right",
                "diverge_boundary_id": tag,
                "reference_boundary_id": "right" if tag == "left" else "left",
                "diverge_side": tag,
                "asymmetry_score": abs(left_score - right_score),
                "tag_confidence": confidence,
                "left_score": left_score,
                "right_score": right_score,
                "tag_source": "geometry_heuristic",
                "anchor_idx": int(anchor_idx),
                "anchor_x": float(anchor[0]),
                "anchor_y": float(anchor[1]),
                "distance_to_reference_boundary_m": reference_dist,
                "distance_to_ego_m": ego_dist,
                "scene_json": str(scene_json_out),
            }
        )
        asset_rows.append(
            {
                "sample_token": token,
                "scene_name": sample["scene_name"],
                "scene_pos": scene_pos,
                "scene_len": scene_len,
                "is_temporal_eligible": True,
                "is_phase1_selected": False,
                "is_primary_scene_sample": False,
                "has_scene_json": True,
                "scene_json": str(scene_json_out),
                "has_centerline_json": False,
                "centerline_json": "",
                "has_blind_rsa_loc": False,
                "blind_rsa_x": "",
                "blind_rsa_y": "",
                "blind_rsa_z": "",
                "has_blind_eta_loc": True,
                "blind_eta_x": float(anchor[0]),
                "blind_eta_y": float(anchor[1]),
                "blind_eta_z": args.light_z,
                "has_patch_rsa_file": False,
                "patch_rsa_file": "",
                "has_patch_eta_file": False,
                "patch_eta_file": "",
                "mapecho_loc_method": "diverge_boundary_anchor_heuristic",
                "mapecho_tag_confidence": confidence,
                "mapecho_distance_to_reference_boundary_m": reference_dist,
            }
        )

    tag_fields = [
        "sample_token",
        "scene_name",
        "scene_pos",
        "scene_len",
        "boundary_left_id",
        "boundary_right_id",
        "diverge_boundary_id",
        "reference_boundary_id",
        "diverge_side",
        "asymmetry_score",
        "tag_confidence",
        "left_score",
        "right_score",
        "tag_source",
        "anchor_idx",
        "anchor_x",
        "anchor_y",
        "distance_to_reference_boundary_m",
        "distance_to_ego_m",
        "scene_json",
    ]
    asset_fields = [
        "sample_token",
        "scene_name",
        "scene_pos",
        "scene_len",
        "is_temporal_eligible",
        "is_phase1_selected",
        "is_primary_scene_sample",
        "has_scene_json",
        "scene_json",
        "has_centerline_json",
        "centerline_json",
        "has_blind_rsa_loc",
        "blind_rsa_x",
        "blind_rsa_y",
        "blind_rsa_z",
        "has_blind_eta_loc",
        "blind_eta_x",
        "blind_eta_y",
        "blind_eta_z",
        "has_patch_rsa_file",
        "patch_rsa_file",
        "has_patch_eta_file",
        "patch_eta_file",
        "mapecho_loc_method",
        "mapecho_tag_confidence",
        "mapecho_distance_to_reference_boundary_m",
    ]
    write_csv(out_dir / "asymmetric_dist_boundary_tags.csv", tag_rows, tag_fields)
    write_csv(out_dir / "phase1_1_asymmetric_dist_eta_like_assets.csv", asset_rows, asset_fields)

    summary = {
        "source_stage": args.source_stage,
        "eligibility_key": args.eligibility_key,
        "num_assets": len(asset_rows),
        "num_scenes": len({row["scene_name"] for row in asset_rows}),
        "boundary_tags_csv": str(out_dir / "asymmetric_dist_boundary_tags.csv"),
        "asset_csv": str(out_dir / "phase1_1_asymmetric_dist_eta_like_assets.csv"),
        "scene_json_dir": str(scene_dir),
        "median_tag_confidence": float(np.median([row["tag_confidence"] for row in tag_rows])) if tag_rows else None,
        "median_distance_to_reference_boundary_m": float(np.median([row["distance_to_reference_boundary_m"] for row in tag_rows])) if tag_rows else None,
    }
    (out_dir / "phase1_1_asymmetric_dist_asset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
