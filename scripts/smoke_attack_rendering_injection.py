#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transform_matrix(rotation_quat, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def invert_rigid(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inv


def homogeneous(point3):
    return np.array([point3[0], point3[1], point3[2], 1.0], dtype=np.float64)


def point3_str(point):
    return " ".join(f"{float(value):.6f}" for value in point)


def parse_offsets(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def glare_radius(distance, img_h, img_w):
    base_radius = min(img_h, img_w) // 2
    min_radius = min(img_h, img_w) // 8
    normalized_distance = min(max(distance, 1.0), 30.0)
    log_factor = 1.0 - (math.log(normalized_distance) / math.log(30.0))
    return int(min_radius + (base_radius - min_radius) * max(0.0, log_factor))


def glare_intensity(distance, power):
    effective_distance = max(distance, 1.0)
    intensity = power / (effective_distance ** 1.5)
    intensity = min(1.0, intensity * 0.02)
    return max(float(intensity), 0.6)


def project_ego_to_camera(point_ego, cam_info):
    ego2cam = np.asarray(cam_info["extrinsics"], dtype=np.float64)
    intrinsic = np.asarray(cam_info["intrinsics"], dtype=np.float64)
    point_cam = (ego2cam @ homogeneous(point_ego))[:3]
    depth = float(point_cam[2])
    if depth <= 1e-6:
        return point_cam, math.nan, math.nan, depth
    uvw = intrinsic @ point_cam
    return point_cam, float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), depth


def choose_attack_camera(sample, point_ego):
    candidates = []
    for cam_name in CAMERA_ORDER:
        cam_info = sample["cams"][cam_name]
        img = cv2.imread(cam_info["img_fpath"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        point_cam, u, v, depth = project_ego_to_camera(point_ego, cam_info)
        in_image = (
            depth > 1e-6
            and math.isfinite(u)
            and math.isfinite(v)
            and 0 <= u < img_w
            and 0 <= v < img_h
        )
        if not in_image:
            continue
        center_distance = math.hypot(u - img_w / 2.0, v - img_h / 2.0)
        distance_to_camera = float(np.linalg.norm(point_cam))
        candidates.append(
            {
                "camera": cam_name,
                "cam_info": cam_info,
                "img_h": img_h,
                "img_w": img_w,
                "point_cam": point_cam,
                "u": u,
                "v": v,
                "depth": depth,
                "distance_to_camera": distance_to_camera,
                "center_distance_px": center_distance,
                "glare_radius_px": glare_radius(distance_to_camera, img_h, img_w),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["center_distance_px"], row["distance_to_camera"]))


def resample_polyline(points, interval):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return points
    sampled = [points[0]]
    carry = 0.0
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        direction = segment / length
        distance = interval - carry
        while distance <= length:
            sampled.append(start + direction * distance)
            distance += interval
        carry = length - (distance - interval)
    sampled.append(points[-1])
    return np.asarray(sampled, dtype=np.float64)


def extract_scene_boundaries(scene_json):
    with open(scene_json) as f:
        scene = json.load(f)
    left = right = None
    for element in scene["map_elements"]:
        if element["tag"] == "left":
            left = np.asarray(element["coordinates"], dtype=np.float64)
        elif element["tag"] == "right":
            right = np.asarray(element["coordinates"], dtype=np.float64)
    diverge_tag = scene["diverge_boundary_tag"][0]
    if diverge_tag == "left":
        return diverge_tag, left, right
    return diverge_tag, right, left


def lidar_polyline_to_global(polyline_xy, z, target_lidar2ego, target_ego2global):
    points_global = []
    for x, y in polyline_xy:
        point_ego = (target_lidar2ego @ homogeneous([x, y, z]))[:3]
        point_global = (target_ego2global @ homogeneous(point_ego))[:3]
        points_global.append(point_global)
    return np.asarray(points_global, dtype=np.float64)


def global_polyline_to_ego(polyline_global, current_global2ego):
    return np.asarray(
        [(current_global2ego @ homogeneous(point))[:3] for point in polyline_global],
        dtype=np.float64,
    )


def project_polyline_ego(polyline_ego, cam_info, img_w, img_h):
    uv = []
    for point in polyline_ego:
        _, u, v, depth = project_ego_to_camera(point, cam_info)
        if (
            depth > 1e-6
            and math.isfinite(u)
            and math.isfinite(v)
            and -0.5 * img_w <= u < 1.5 * img_w
            and -0.5 * img_h <= v < 1.5 * img_h
        ):
            uv.append([u, v])
    return np.asarray(uv, dtype=np.float64)


def render_glare(image_bgr, u, v, radius, intensity):
    h, w = image_bgr.shape[:2]
    x0 = int(max(0, math.floor(u - radius)))
    x1 = int(min(w, math.ceil(u + radius + 1)))
    y0 = int(max(0, math.floor(v - radius)))
    y1 = int(min(h, math.ceil(v + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return image_bgr.copy()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - u) ** 2 + (yy - v) ** 2
    sigma = max(radius / 2.4, 1.0)
    alpha = np.exp(-dist2 / (2.0 * sigma * sigma)) * intensity
    alpha = np.clip(alpha, 0.0, 0.92)[..., None]
    tint = np.array([255.0, 245.0, 235.0], dtype=np.float32)
    out = image_bgr.astype(np.float32)
    region = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = region * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_overlay(image_bgr, attack, diverge_uv, reference_uv, label):
    out = image_bgr.copy()
    if len(diverge_uv) >= 2:
        pts = np.round(diverge_uv).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    if len(reference_uv) >= 2:
        pts = np.round(reference_uv).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (0, 180, 0), 4, cv2.LINE_AA)
    if attack:
        center = (int(round(attack["u"])), int(round(attack["v"])))
        cv2.circle(out, center, int(attack["radius"]), (0, 220, 255), 3)
        cv2.circle(out, center, 9, (255, 0, 0), -1)
    cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def vpa_coverage(target_uv, u, v, radius):
    if len(target_uv) == 0:
        return 0.0
    dists = np.linalg.norm(target_uv - np.array([u, v]), axis=1)
    return float(np.mean(dists <= radius))


def main():
    parser = argparse.ArgumentParser(
        description="Render raw-image attack injection smoke test artifacts."
    )
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attack-objective", choices=["eta", "rsa"], default="eta")
    parser.add_argument("--source-frame", choices=["lidar", "ego"], default="lidar")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--offsets", default="-2,-1,0,1,2")
    parser.add_argument("--boundary-z", type=float, default=-1.84)
    parser.add_argument("--sample-interval-m", type=float, default=0.25)
    parser.add_argument("--power", type=float, default=3000.0)
    args = parser.parse_args()

    stream_samples = load_pickle(args.stream_ann)
    asset_rows = load_csv(args.asset_csv)
    offsets = parse_offsets(args.offsets)
    out_dir = Path(args.out_dir)

    sample_by_token = {sample["token"]: sample for sample in stream_samples}
    scenes = defaultdict(list)
    for sample in stream_samples:
        scenes[sample["scene_name"]].append(sample)
    for samples in scenes.values():
        samples.sort(key=lambda sample: sample["sample_idx"])

    loc_prefix = f"blind_{args.attack_objective}"
    has_loc_key = f"has_blind_{args.attack_objective}_loc"
    selected_assets = [
        row for row in asset_rows if str(row.get(has_loc_key, "")).lower() == "true"
    ][: args.max_samples]

    frame_rows = []
    sample_summaries = []
    for asset in selected_assets:
        token = asset["sample_token"]
        target = sample_by_token[token]
        scene_samples = scenes[target["scene_name"]]
        target_scene_pos = next(
            i for i, sample in enumerate(scene_samples) if sample["token"] == token
        )
        sample_dir = out_dir / args.attack_objective / token
        sample_dir.mkdir(parents=True, exist_ok=True)

        point_source = np.array(
            [
                float(asset[f"{loc_prefix}_x"]),
                float(asset[f"{loc_prefix}_y"]),
                float(asset[f"{loc_prefix}_z"]),
            ],
            dtype=np.float64,
        )
        target_lidar2ego = transform_matrix(
            target["lidar2ego_rotation"], target["lidar2ego_translation"]
        )
        target_ego2global = transform_matrix(target["e2g_rotation"], target["e2g_translation"])
        if args.source_frame == "lidar":
            point_ego_t = (target_lidar2ego @ homogeneous(point_source))[:3]
        else:
            point_ego_t = point_source
        point_global = (target_ego2global @ homogeneous(point_ego_t))[:3]

        diverge_tag, diverge_xy, reference_xy = extract_scene_boundaries(asset["scene_json"])
        diverge_xy = resample_polyline(diverge_xy, args.sample_interval_m)
        reference_xy = resample_polyline(reference_xy, args.sample_interval_m)
        diverge_global = lidar_polyline_to_global(
            diverge_xy, args.boundary_z, target_lidar2ego, target_ego2global
        )
        reference_global = lidar_polyline_to_global(
            reference_xy, args.boundary_z, target_lidar2ego, target_ego2global
        )

        attack_schedule = {
            "warmup": "clean",
            "target_frame_t": "attacked",
            "recovery": "clean",
            "N_attack": 1,
            "attacked_offsets": [0],
            "injection_order": "raw image -> attack rendering -> resize/pad/normalization -> model input",
        }
        (sample_dir / "attack_schedule.json").write_text(json.dumps(attack_schedule, indent=2))

        distances = []
        radii = []
        attacked_count = 0
        for offset in offsets:
            frame_pos = target_scene_pos + offset
            if frame_pos < 0 or frame_pos >= len(scene_samples):
                continue
            frame = scene_samples[frame_pos]
            ego2global = transform_matrix(frame["e2g_rotation"], frame["e2g_translation"])
            global2ego = invert_rigid(ego2global)
            point_ego = (global2ego @ homogeneous(point_global))[:3]
            camera = choose_attack_camera(frame, point_ego)
            if camera is None:
                continue

            clean = cv2.imread(camera["cam_info"]["img_fpath"], cv2.IMREAD_COLOR)
            if clean is None:
                continue
            img_h, img_w = clean.shape[:2]
            is_attacked = offset == 0
            intensity = glare_intensity(camera["distance_to_camera"], args.power)
            scheduled = (
                render_glare(clean, camera["u"], camera["v"], camera["glare_radius_px"], intensity)
                if is_attacked
                else clean.copy()
            )
            attacked_count += int(is_attacked)

            diverge_ego = global_polyline_to_ego(diverge_global, global2ego)
            reference_ego = global_polyline_to_ego(reference_global, global2ego)
            diverge_uv = project_polyline_ego(diverge_ego, camera["cam_info"], img_w, img_h)
            reference_uv = project_polyline_ego(reference_ego, camera["cam_info"], img_w, img_h)
            coverage = vpa_coverage(
                diverge_uv, camera["u"], camera["v"], camera["glare_radius_px"]
            )

            frame_dir = sample_dir / f"offset_{offset:+03d}_{camera['camera']}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            clean_path = frame_dir / "clean_raw.jpg"
            scheduled_path = frame_dir / ("attacked_raw.jpg" if is_attacked else "scheduled_clean_raw.jpg")
            overlay_path = frame_dir / "overlay.jpg"
            shutil.copyfile(camera["cam_info"]["img_fpath"], clean_path)
            cv2.imwrite(str(scheduled_path), scheduled)
            overlay_base = scheduled if is_attacked else clean
            label = (
                f"{token[:8]} off {offset:+d} {camera['camera']} "
                f"{'ATTACK' if is_attacked else 'clean'} cov={coverage:.2f}"
            )
            overlay = draw_overlay(
                overlay_base,
                {
                    "u": camera["u"],
                    "v": camera["v"],
                    "radius": camera["glare_radius_px"],
                },
                diverge_uv,
                reference_uv,
                label,
            )
            cv2.imwrite(str(overlay_path), overlay)

            row = {
                "sample_token": token,
                "frame_offset": offset,
                "frame_token": frame["token"],
                "scene_name": frame["scene_name"],
                "scene_pos": frame_pos,
                "camera": camera["camera"],
                "is_attacked": is_attacked,
                "p_global": point3_str(point_global),
                "p_ego": point3_str(point_ego),
                "p_cam": point3_str(camera["point_cam"]),
                "u": camera["u"],
                "v": camera["v"],
                "depth": camera["depth"],
                "distance_to_camera": camera["distance_to_camera"],
                "distance_to_ego_xy": float(np.linalg.norm(point_ego[:2])),
                "intensity": intensity,
                "glare_radius_px": camera["glare_radius_px"],
                "diverge_vpa_coverage": coverage,
                "clean_shape": f"{img_h} {img_w} {clean.shape[2]}",
                "clean_dtype": str(clean.dtype),
                "scheduled_shape": f"{scheduled.shape[0]} {scheduled.shape[1]} {scheduled.shape[2]}",
                "scheduled_dtype": str(scheduled.dtype),
                "scheduled_min": int(scheduled.min()),
                "scheduled_max": int(scheduled.max()),
                "clean_raw_path": str(clean_path),
                "scheduled_raw_path": str(scheduled_path),
                "overlay_path": str(overlay_path),
            }
            frame_rows.append(row)
            distances.append(camera["distance_to_camera"])
            radii.append(camera["glare_radius_px"])

            frame_meta = dict(row)
            frame_meta["attack_schedule"] = attack_schedule
            frame_meta["diverge_boundary_tag"] = diverge_tag
            frame_meta["source_frame"] = args.source_frame
            (frame_dir / "metadata.json").write_text(json.dumps(frame_meta, indent=2))

        sample_summaries.append(
            {
                "sample_token": token,
                "scene_name": target["scene_name"],
                "target_scene_pos": target_scene_pos,
                "frames_written": sum(1 for row in frame_rows if row["sample_token"] == token),
                "attacked_frames_written": attacked_count,
                "p_global": point3_str(point_global),
                "min_distance_to_camera": min(distances) if distances else "",
                "max_distance_to_camera": max(distances) if distances else "",
                "min_glare_radius_px": min(radii) if radii else "",
                "max_glare_radius_px": max(radii) if radii else "",
                "schedule_path": str(sample_dir / "attack_schedule.json"),
            }
        )

    fieldnames = [
        "sample_token",
        "frame_offset",
        "frame_token",
        "scene_name",
        "scene_pos",
        "camera",
        "is_attacked",
        "p_global",
        "p_ego",
        "p_cam",
        "u",
        "v",
        "depth",
        "distance_to_camera",
        "distance_to_ego_xy",
        "intensity",
        "glare_radius_px",
        "diverge_vpa_coverage",
        "clean_shape",
        "clean_dtype",
        "scheduled_shape",
        "scheduled_dtype",
        "scheduled_min",
        "scheduled_max",
        "clean_raw_path",
        "scheduled_raw_path",
        "overlay_path",
    ]
    frame_csv = out_dir / f"{args.attack_objective}_injection_smoke_frames.csv"
    write_csv(frame_csv, frame_rows, fieldnames)

    summary_csv = out_dir / f"{args.attack_objective}_injection_smoke_samples.csv"
    write_csv(
        summary_csv,
        sample_summaries,
        [
            "sample_token",
            "scene_name",
            "target_scene_pos",
            "frames_written",
            "attacked_frames_written",
            "p_global",
            "min_distance_to_camera",
            "max_distance_to_camera",
            "min_glare_radius_px",
            "max_glare_radius_px",
            "schedule_path",
        ],
    )

    summary = {
        "attack_objective": args.attack_objective,
        "samples_checked": len(sample_summaries),
        "frame_rows": len(frame_rows),
        "attacked_frame_rows": sum(row["is_attacked"] for row in frame_rows),
        "frame_csv": str(frame_csv),
        "sample_csv": str(summary_csv),
        "out_dir": str(out_dir),
        "pass_schedule_n_attack_1": all(
            row["attacked_frames_written"] == 1 for row in sample_summaries
        ),
        "pass_raw_uint8": all(
            row["clean_dtype"] == "uint8" and row["scheduled_dtype"] == "uint8"
            for row in frame_rows
        ),
        "pass_shape_unchanged": all(
            row["clean_shape"] == row["scheduled_shape"] for row in frame_rows
        ),
    }
    summary_json = out_dir / f"{args.attack_objective}_injection_smoke_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
