#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

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


def parse_offsets(offset_text):
    return [int(value.strip()) for value in offset_text.split(",") if value.strip()]


def get_image_shape(cam_info):
    try:
        import cv2

        img = cv2.imread(cam_info["img_fpath"], cv2.IMREAD_UNCHANGED)
        if img is not None:
            return int(img.shape[0]), int(img.shape[1])
    except Exception:
        pass
    return 900, 1600


def glare_radius(distance, img_h, img_w):
    base_radius = min(img_h, img_w) // 2
    min_radius = min(img_h, img_w) // 8
    normalized_distance = min(max(distance, 1.0), 30.0)
    log_factor = 1.0 - (math.log(normalized_distance) / math.log(30.0))
    return int(min_radius + (base_radius - min_radius) * max(0.0, log_factor))


def render_overlay(image_path, out_path, u, v, radius, label):
    import cv2

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return False
    h, w = image.shape[:2]
    center = (int(round(np.clip(u, 0, w - 1))), int(round(np.clip(v, 0, h - 1))))
    overlay = image.copy()
    cv2.circle(overlay, center, int(radius), (0, 255, 255), -1)
    image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
    cv2.circle(image, center, 10, (0, 0, 255), -1)
    cv2.circle(image, center, int(radius), (0, 200, 255), 3)
    cv2.drawMarker(
        image,
        center,
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=36,
        thickness=3,
    )
    cv2.putText(
        image,
        label,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), image))


def project_ego_to_camera(point_ego, cam_info):
    ego2cam = np.asarray(cam_info["extrinsics"], dtype=np.float64)
    intrinsic = np.asarray(cam_info["intrinsics"], dtype=np.float64)
    point_cam = (ego2cam @ homogeneous(point_ego))[:3]
    depth = float(point_cam[2])
    if depth <= 1e-6:
        return point_cam, math.nan, math.nan, depth
    uvw = intrinsic @ point_cam
    return point_cam, float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), depth


def choose_visible_camera(camera_rows):
    visible = [row for row in camera_rows if row["in_image"]]
    if not visible:
        return None
    return min(
        visible,
        key=lambda row: (
            row["center_distance_px"],
            row["distance_to_camera"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check CCS'25 attack-point coordinate conversion and projection."
    )
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attack-objective", choices=["eta", "rsa"], default="eta")
    parser.add_argument("--source-frame", choices=["lidar", "ego"], default="lidar")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--offsets", default="-2,-1,0,1,2,5")
    parser.add_argument("--render-overlays", action="store_true")
    parser.add_argument("--render-max-samples", type=int, default=1)
    args = parser.parse_args()

    stream_samples = load_pickle(args.stream_ann)
    asset_rows = load_csv(args.asset_csv)
    offsets = parse_offsets(args.offsets)

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
    ]
    selected_assets = selected_assets[: args.max_samples]

    summary_rows = []
    projection_rows = []
    overlay_rows = []
    for asset_index, asset in enumerate(selected_assets):
        token = asset["sample_token"]
        target = sample_by_token[token]
        scene_samples = scenes[target["scene_name"]]
        scene_pos = next(
            i for i, sample in enumerate(scene_samples) if sample["token"] == token
        )

        point_source = np.array(
            [
                float(asset[f"{loc_prefix}_x"]),
                float(asset[f"{loc_prefix}_y"]),
                float(asset[f"{loc_prefix}_z"]),
            ],
            dtype=np.float64,
        )
        lidar2ego = transform_matrix(
            target["lidar2ego_rotation"], target["lidar2ego_translation"]
        )
        target_ego2global = transform_matrix(
            target["e2g_rotation"], target["e2g_translation"]
        )
        if args.source_frame == "lidar":
            point_ego_target = (lidar2ego @ homogeneous(point_source))[:3]
        else:
            point_ego_target = point_source
        point_global = (target_ego2global @ homogeneous(point_ego_target))[:3]

        offset_distance = {}
        offset_best_camera = {}
        offset_best_uv = {}

        for offset in offsets:
            frame_pos = scene_pos + offset
            if frame_pos < 0 or frame_pos >= len(scene_samples):
                continue
            frame = scene_samples[frame_pos]
            ego2global = transform_matrix(frame["e2g_rotation"], frame["e2g_translation"])
            global2ego = invert_rigid(ego2global)
            point_ego_frame = (global2ego @ homogeneous(point_global))[:3]
            distance_to_ego = float(np.linalg.norm(point_ego_frame[:2]))
            offset_distance[offset] = distance_to_ego

            camera_rows_for_frame = []
            for cam_name in CAMERA_ORDER:
                cam_info = frame["cams"][cam_name]
                img_h, img_w = get_image_shape(cam_info)
                point_cam, u, v, depth = project_ego_to_camera(point_ego_frame, cam_info)
                in_image = (
                    depth > 1e-6
                    and math.isfinite(u)
                    and math.isfinite(v)
                    and 0 <= u < img_w
                    and 0 <= v < img_h
                )
                center_distance_px = (
                    math.hypot(u - img_w / 2.0, v - img_h / 2.0)
                    if math.isfinite(u) and math.isfinite(v)
                    else math.inf
                )
                distance_to_camera = float(np.linalg.norm(point_cam))
                radius = (
                    glare_radius(distance_to_camera, img_h, img_w)
                    if depth > 1e-6 and math.isfinite(u) and math.isfinite(v)
                    else ""
                )
                row = {
                    "sample_token": token,
                    "attack_objective": args.attack_objective,
                    "source_frame": args.source_frame,
                    "frame_offset": offset,
                    "frame_token": frame["token"],
                    "scene_name": frame["scene_name"],
                    "scene_pos": frame_pos,
                    "camera": cam_name,
                    "point_ego_frame": point3_str(point_ego_frame),
                    "distance_to_ego_xy": distance_to_ego,
                    "point_cam": point3_str(point_cam),
                    "distance_to_camera": distance_to_camera,
                    "u": "" if not math.isfinite(u) else u,
                    "v": "" if not math.isfinite(v) else v,
                    "depth": depth,
                    "img_w": img_w,
                    "img_h": img_h,
                    "in_image": in_image,
                    "center_distance_px": (
                        "" if center_distance_px == math.inf else center_distance_px
                    ),
                    "glare_radius_px": radius,
                }
                camera_rows_for_frame.append(row)
                projection_rows.append(row)

            best = choose_visible_camera(camera_rows_for_frame)
            if best is not None:
                offset_best_camera[offset] = best["camera"]
                offset_best_uv[offset] = f"{float(best['u']):.2f} {float(best['v']):.2f}"
                if args.render_overlays and asset_index < args.render_max_samples:
                    frame = scene_samples[frame_pos]
                    cam_info = frame["cams"][best["camera"]]
                    overlay_dir = (
                        Path(args.out_dir)
                        / "overlays"
                        / args.attack_objective
                        / token
                    )
                    overlay_path = overlay_dir / (
                        f"offset_{offset:+03d}_{best['camera']}.jpg"
                    )
                    label = (
                        f"{token[:8]} offset {offset:+d} {best['camera']} "
                        f"uv=({float(best['u']):.1f},{float(best['v']):.1f}) "
                        f"r={int(best['glare_radius_px'])}"
                    )
                    wrote = render_overlay(
                        cam_info["img_fpath"],
                        overlay_path,
                        float(best["u"]),
                        float(best["v"]),
                        int(best["glare_radius_px"]),
                        label,
                    )
                    overlay_rows.append(
                        {
                            "sample_token": token,
                            "attack_objective": args.attack_objective,
                            "frame_offset": offset,
                            "frame_token": frame["token"],
                            "camera": best["camera"],
                            "u": best["u"],
                            "v": best["v"],
                            "distance_to_camera": best["distance_to_camera"],
                            "glare_radius_px": best["glare_radius_px"],
                            "overlay_path": str(overlay_path) if wrote else "",
                        }
                    )

        summary_rows.append(
            {
                "sample_token": token,
                "scene_name": target["scene_name"],
                "scene_pos": scene_pos,
                "attack_objective": args.attack_objective,
                "source_frame": args.source_frame,
                "p_source": point3_str(point_source),
                "p_ego_target": point3_str(point_ego_target),
                "p_global": point3_str(point_global),
                "distance_to_ego_t": offset_distance.get(0, ""),
                "distance_to_ego_t_plus_1": offset_distance.get(1, ""),
                "distance_to_ego_t_plus_5": offset_distance.get(5, ""),
                "visible_camera_at_t": offset_best_camera.get(0, ""),
                "projected_uv_at_t": offset_best_uv.get(0, ""),
                "visible_camera_at_t_plus_1": offset_best_camera.get(1, ""),
                "projected_uv_at_t_plus_1": offset_best_uv.get(1, ""),
                "visible_camera_at_t_plus_5": offset_best_camera.get(5, ""),
                "projected_uv_at_t_plus_5": offset_best_uv.get(5, ""),
            }
        )

    out_dir = Path(args.out_dir)
    summary_csv = out_dir / f"{args.attack_objective}_attack_point_coordinate_sanity_summary.csv"
    projection_csv = out_dir / f"{args.attack_objective}_attack_point_frame_projection_sanity.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "sample_token",
            "scene_name",
            "scene_pos",
            "attack_objective",
            "source_frame",
            "p_source",
            "p_ego_target",
            "p_global",
            "distance_to_ego_t",
            "distance_to_ego_t_plus_1",
            "distance_to_ego_t_plus_5",
            "visible_camera_at_t",
            "projected_uv_at_t",
            "visible_camera_at_t_plus_1",
            "projected_uv_at_t_plus_1",
            "visible_camera_at_t_plus_5",
            "projected_uv_at_t_plus_5",
        ],
    )
    write_csv(
        projection_csv,
        projection_rows,
        [
            "sample_token",
            "attack_objective",
            "source_frame",
            "frame_offset",
            "frame_token",
            "scene_name",
            "scene_pos",
            "camera",
            "point_ego_frame",
            "distance_to_ego_xy",
            "point_cam",
            "distance_to_camera",
            "u",
            "v",
            "depth",
            "img_w",
            "img_h",
            "in_image",
            "center_distance_px",
            "glare_radius_px",
        ],
    )
    overlay_csv = out_dir / f"{args.attack_objective}_attack_point_overlay_index.csv"
    if args.render_overlays:
        write_csv(
            overlay_csv,
            overlay_rows,
            [
                "sample_token",
                "attack_objective",
                "frame_offset",
                "frame_token",
                "camera",
                "u",
                "v",
                "distance_to_camera",
                "glare_radius_px",
                "overlay_path",
            ],
        )
    summary = {
        "attack_objective": args.attack_objective,
        "source_frame": args.source_frame,
        "samples_checked": len(summary_rows),
        "offsets": offsets,
        "summary_csv": str(summary_csv),
        "projection_csv": str(projection_csv),
        "visible_at_t": sum(bool(row["visible_camera_at_t"]) for row in summary_rows),
        "visible_at_t_plus_1": sum(
            bool(row["visible_camera_at_t_plus_1"]) for row in summary_rows
        ),
        "visible_at_t_plus_5": sum(
            bool(row["visible_camera_at_t_plus_5"]) for row in summary_rows
        ),
        "overlay_csv": str(overlay_csv) if args.render_overlays else "",
        "overlays_written": sum(bool(row["overlay_path"]) for row in overlay_rows),
    }
    summary_json = out_dir / f"{args.attack_objective}_attack_point_coordinate_sanity_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
