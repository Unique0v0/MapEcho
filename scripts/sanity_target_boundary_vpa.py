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


def points2_str(points):
    return ";".join(f"{float(x):.3f} {float(y):.3f}" for x, y in points)


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


def project_ego_to_camera(point_ego, cam_info):
    ego2cam = np.asarray(cam_info["extrinsics"], dtype=np.float64)
    intrinsic = np.asarray(cam_info["intrinsics"], dtype=np.float64)
    point_cam = (ego2cam @ homogeneous(point_ego))[:3]
    depth = float(point_cam[2])
    if depth <= 1e-6:
        return point_cam, math.nan, math.nan, depth
    uvw = intrinsic @ point_cam
    return point_cam, float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), depth


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


def point_to_polyline_distance(point_xy, polyline_xy):
    point = np.asarray(point_xy, dtype=np.float64)
    polyline = np.asarray(polyline_xy, dtype=np.float64)
    best = math.inf
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            dist = float(np.linalg.norm(point - start))
        else:
            t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
            projection = start + t * segment
            dist = float(np.linalg.norm(point - projection))
        best = min(best, dist)
    return best


def choose_attack_camera(sample, point_ego):
    candidates = []
    for cam_name in CAMERA_ORDER:
        cam_info = sample["cams"][cam_name]
        img_h, img_w = get_image_shape(cam_info)
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


def project_polyline_to_image(polyline_xy, z, cam_info):
    uv_points = []
    for x, y in polyline_xy:
        _, u, v, depth = project_ego_to_camera(np.array([x, y, z]), cam_info)
        if depth > 1e-6 and math.isfinite(u) and math.isfinite(v):
            uv_points.append([u, v])
    return np.asarray(uv_points, dtype=np.float64)


def extract_scene_targets(scene_json, centerline_json):
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
        diverge = left
        reference = right
    else:
        diverge = right
        reference = left

    targets = {
        "diverge_boundary": diverge,
        "reference_boundary": reference,
    }
    if centerline_json:
        with open(centerline_json) as f:
            targets["eta_centerline"] = np.asarray(json.load(f), dtype=np.float64)
    return diverge_tag, targets


def render_vpa_overlay(image_path, out_path, attack_u, attack_v, radius, target_uv, label):
    import cv2

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return False
    h, w = image.shape[:2]
    center = (
        int(round(np.clip(attack_u, 0, w - 1))),
        int(round(np.clip(attack_v, 0, h - 1))),
    )
    overlay = image.copy()
    cv2.circle(overlay, center, int(radius), (0, 255, 255), -1)
    image = cv2.addWeighted(overlay, 0.26, image, 0.74, 0)
    if len(target_uv) >= 2:
        pts = np.round(target_uv).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.polylines(image, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(image, center, 9, (255, 0, 0), -1)
    cv2.circle(image, center, int(radius), (0, 200, 255), 3)
    cv2.putText(
        image,
        label,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), image))


def main():
    parser = argparse.ArgumentParser(
        description="Compute target-boundary VPA sanity for CCS'25 attack locations."
    )
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attack-objective", choices=["eta", "rsa"], default="eta")
    parser.add_argument("--source-frame", choices=["lidar", "ego"], default="lidar")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--sample-interval-m", type=float, default=0.25)
    parser.add_argument("--boundary-z", type=float, default=-1.84)
    parser.add_argument("--pixel-threshold", type=float, default=10.0)
    parser.add_argument("--meter-threshold", type=float, default=2.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.05)
    parser.add_argument("--render-overlays", action="store_true")
    parser.add_argument("--render-max-samples", type=int, default=5)
    args = parser.parse_args()

    stream_samples = load_pickle(args.stream_ann)
    asset_rows = load_csv(args.asset_csv)
    sample_by_token = {sample["token"]: sample for sample in stream_samples}

    loc_prefix = f"blind_{args.attack_objective}"
    has_loc_key = f"has_blind_{args.attack_objective}_loc"
    selected_assets = [
        row for row in asset_rows if str(row.get(has_loc_key, "")).lower() == "true"
    ][: args.max_samples]

    rows = []
    overlay_rows = []
    for asset_index, asset in enumerate(selected_assets):
        token = asset["sample_token"]
        sample = sample_by_token[token]
        point_source = np.array(
            [
                float(asset[f"{loc_prefix}_x"]),
                float(asset[f"{loc_prefix}_y"]),
                float(asset[f"{loc_prefix}_z"]),
            ],
            dtype=np.float64,
        )
        lidar2ego = transform_matrix(sample["lidar2ego_rotation"], sample["lidar2ego_translation"])
        if args.source_frame == "lidar":
            point_ego = (lidar2ego @ homogeneous(point_source))[:3]
        else:
            point_ego = point_source

        camera = choose_attack_camera(sample, point_ego)
        scene_json = asset["scene_json"]
        centerline_json = asset.get("centerline_json", "")
        diverge_tag, targets = extract_scene_targets(scene_json, centerline_json)

        for target_type, target_lidar_xy in targets.items():
            target_lidar_xy = resample_polyline(target_lidar_xy, args.sample_interval_m)
            target_ego_xyz_ground = np.asarray(
                [
                    (lidar2ego @ homogeneous([x, y, args.boundary_z]))[:3]
                    for x, y in target_lidar_xy
                ],
                dtype=np.float64,
            )
            target_ego_xyz_attack_height = np.asarray(
                [
                    (lidar2ego @ homogeneous([x, y, point_source[2]]))[:3]
                    for x, y in target_lidar_xy
                ],
                dtype=np.float64,
            )
            target_ego_xy = target_ego_xyz_ground[:, :2]
            dist_m = point_to_polyline_distance(point_ego[:2], target_ego_xy)

            visible_cam = ""
            attack_u = attack_v = attack_depth = ""
            radius = ""
            dist_px_ground = ""
            dist_px_attack_height = ""
            coverage = 0.0
            overlay_on_target = False
            ground_center_pass = False
            height_aligned_center_pass = False
            vpa_pass = False
            projected_points = 0
            overlay_path = ""

            if camera is not None:
                cam_info = camera["cam_info"]
                target_uv_ground = []
                for point in target_ego_xyz_ground:
                    _, u, v, depth = project_ego_to_camera(point, cam_info)
                    if (
                        depth > 1e-6
                        and math.isfinite(u)
                        and math.isfinite(v)
                        and -0.5 * camera["img_w"] <= u < 1.5 * camera["img_w"]
                        and -0.5 * camera["img_h"] <= v < 1.5 * camera["img_h"]
                    ):
                        target_uv_ground.append([u, v])
                target_uv_ground = np.asarray(target_uv_ground, dtype=np.float64)
                projected_points = len(target_uv_ground)
                visible_cam = camera["camera"]
                attack_u = camera["u"]
                attack_v = camera["v"]
                attack_depth = camera["depth"]
                radius = camera["glare_radius_px"]
                if projected_points > 0:
                    dists = np.linalg.norm(target_uv_ground - np.array([attack_u, attack_v]), axis=1)
                    dist_px_ground = float(dists.min())
                    coverage = float(np.mean(dists <= radius))
                    overlay_on_target = bool(coverage > 0.0)

                    target_uv_attack_height = []
                    for point in target_ego_xyz_attack_height:
                        _, u, v, depth = project_ego_to_camera(point, cam_info)
                        if (
                            depth > 1e-6
                            and math.isfinite(u)
                            and math.isfinite(v)
                            and -0.5 * camera["img_w"] <= u < 1.5 * camera["img_w"]
                            and -0.5 * camera["img_h"] <= v < 1.5 * camera["img_h"]
                        ):
                            target_uv_attack_height.append([u, v])
                    if target_uv_attack_height:
                        target_uv_attack_height = np.asarray(
                            target_uv_attack_height, dtype=np.float64
                        )
                        dists_attack_height = np.linalg.norm(
                            target_uv_attack_height - np.array([attack_u, attack_v]), axis=1
                        )
                        dist_px_attack_height = float(dists_attack_height.min())

                    ground_center_pass = bool(
                        dist_m <= args.meter_threshold
                        and dist_px_ground <= args.pixel_threshold
                    )
                    height_aligned_center_pass = bool(
                        dist_m <= args.meter_threshold
                        and dist_px_attack_height != ""
                        and dist_px_attack_height <= args.pixel_threshold
                    )
                    vpa_pass = bool(
                        dist_m <= args.meter_threshold
                        and overlay_on_target
                        and coverage >= args.coverage_threshold
                    )
                    if args.render_overlays and asset_index < args.render_max_samples:
                        overlay_dir = (
                            Path(args.out_dir)
                            / "overlays"
                            / args.attack_objective
                            / token
                        )
                        overlay_file = overlay_dir / f"{target_type}_{visible_cam}.jpg"
                        label = (
                            f"{token[:8]} {target_type} "
                            f"dpx={dist_px_ground:.1f} dm={dist_m:.2f} cov={coverage:.2f}"
                        )
                        if render_vpa_overlay(
                            cam_info["img_fpath"],
                            overlay_file,
                            attack_u,
                            attack_v,
                            radius,
                            target_uv_ground,
                            label,
                        ):
                            overlay_path = str(overlay_file)
                            overlay_rows.append(
                                {
                                    "sample_token": token,
                                    "target_type": target_type,
                                    "camera": visible_cam,
                                    "overlay_path": overlay_path,
                                }
                            )

            rows.append(
                {
                    "sample_id": token,
                    "attack_objective": args.attack_objective,
                    "target_type": target_type,
                    "diverge_boundary_tag": diverge_tag,
                    "dist_to_boundary_px": dist_px_ground,
                    "dist_to_boundary_px_attack_height": dist_px_attack_height,
                    "dist_to_boundary_m": dist_m,
                    "on_boundary": dist_m <= args.meter_threshold,
                    "visible_cam": visible_cam,
                    "attack_u": attack_u,
                    "attack_v": attack_v,
                    "attack_depth": attack_depth,
                    "glare_radius_px": radius,
                    "target_projected_points": projected_points,
                    "vpa_point_coverage": coverage,
                    "attack_overlay_on_target": overlay_on_target,
                    "ground_center_pass": ground_center_pass,
                    "height_aligned_center_pass": height_aligned_center_pass,
                    "vpa_pass": vpa_pass,
                    "pass_sanity": vpa_pass,
                    "p_source": point3_str(point_source),
                    "p_ego": point3_str(point_ego),
                    "target_points_ego_xy_preview": points2_str(target_ego_xy[:8]),
                    "overlay_path": overlay_path,
                }
            )

    out_dir = Path(args.out_dir)
    out_csv = out_dir / f"{args.attack_objective}_target_boundary_vpa_sanity.csv"
    write_csv(
        out_csv,
        rows,
        [
            "sample_id",
            "attack_objective",
            "target_type",
            "diverge_boundary_tag",
            "dist_to_boundary_px",
            "dist_to_boundary_px_attack_height",
            "dist_to_boundary_m",
            "on_boundary",
            "visible_cam",
            "attack_u",
            "attack_v",
            "attack_depth",
            "glare_radius_px",
            "target_projected_points",
            "vpa_point_coverage",
            "attack_overlay_on_target",
            "ground_center_pass",
            "height_aligned_center_pass",
            "vpa_pass",
            "pass_sanity",
            "p_source",
            "p_ego",
            "target_points_ego_xy_preview",
            "overlay_path",
        ],
    )
    overlay_csv = out_dir / f"{args.attack_objective}_target_boundary_vpa_overlay_index.csv"
    if args.render_overlays:
        write_csv(
            overlay_csv,
            overlay_rows,
            ["sample_token", "target_type", "camera", "overlay_path"],
        )

    by_target = defaultdict(list)
    for row in rows:
        by_target[row["target_type"]].append(row)
    summary = {
        "attack_objective": args.attack_objective,
        "samples_checked": len(selected_assets),
        "csv": str(out_csv),
        "overlay_csv": str(overlay_csv) if args.render_overlays else "",
        "pixel_threshold": args.pixel_threshold,
        "meter_threshold": args.meter_threshold,
        "coverage_threshold": args.coverage_threshold,
        "targets": {
            target_type: {
                "rows": len(target_rows),
                "visible": sum(bool(row["visible_cam"]) for row in target_rows),
                "on_boundary": sum(row["on_boundary"] for row in target_rows),
                "overlay_on_target": sum(row["attack_overlay_on_target"] for row in target_rows),
                "ground_center_pass": sum(row["ground_center_pass"] for row in target_rows),
                "height_aligned_center_pass": sum(
                    row["height_aligned_center_pass"] for row in target_rows
                ),
                "vpa_pass": sum(row["vpa_pass"] for row in target_rows),
                "pass_sanity": sum(row["pass_sanity"] for row in target_rows),
                "median_dist_m": float(
                    np.median([row["dist_to_boundary_m"] for row in target_rows])
                ),
                "median_dist_px": float(
                    np.median(
                        [
                            row["dist_to_boundary_px"]
                            for row in target_rows
                            if row["dist_to_boundary_px"] != ""
                        ]
                    )
                )
                if any(row["dist_to_boundary_px"] != "" for row in target_rows)
                else None,
                "median_vpa_point_coverage": float(
                    np.median([row["vpa_point_coverage"] for row in target_rows])
                ),
            }
            for target_type, target_rows in by_target.items()
        },
    }
    summary_json = out_dir / f"{args.attack_objective}_target_boundary_vpa_sanity_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
