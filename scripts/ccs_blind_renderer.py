#!/usr/bin/env python3
import math

import cv2
import numpy as np

from smoke_attack_rendering_injection import (
    CAMERA_ORDER,
    homogeneous,
    invert_rigid,
    project_ego_to_camera,
    transform_matrix,
)


def camera_pose_in_lidar(sample, cam_name):
    """Return CCS-style camera position/direction in the current LiDAR frame."""
    ego2cam = np.asarray(sample["cams"][cam_name]["extrinsics"], dtype=np.float64)
    cam2ego = invert_rigid(ego2cam)
    lidar2ego = transform_matrix(
        sample["lidar2ego_rotation"], sample["lidar2ego_translation"]
    )
    ego2lidar = invert_rigid(lidar2ego)

    camera_pos_ego = cam2ego[:3, 3]
    camera_pos_lidar = (ego2lidar @ homogeneous(camera_pos_ego))[:3]

    camera_direction_ego = -cam2ego[:3, 2]
    camera_direction_lidar = (ego2lidar[:3, :3] @ camera_direction_ego)
    norm = np.linalg.norm(camera_direction_lidar)
    if norm > 1e-9:
        camera_direction_lidar = camera_direction_lidar / norm

    return {
        "position": camera_pos_lidar,
        "direction": camera_direction_lidar,
    }


def calculate_light_direction(attacker_position, camera_position):
    direction = camera_position - attacker_position
    norm = np.linalg.norm(direction)
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return direction / norm


def is_camera_affected(camera_params, light_source_position, angle_threshold=math.pi / 3):
    light_direction = calculate_light_direction(
        np.asarray(light_source_position, dtype=np.float64),
        camera_params["position"],
    )
    if np.linalg.norm(light_direction) <= 1e-9:
        return False

    dot = float(np.clip(np.dot(camera_params["direction"], light_direction), -1.0, 1.0))
    angle = math.acos(dot)
    if angle > angle_threshold:
        return False

    cam_to_center = camera_params["position"]
    light_to_center = np.asarray(light_source_position, dtype=np.float64)
    if (
        np.sign(cam_to_center[0]) != np.sign(light_to_center[0])
        and np.sign(cam_to_center[1]) != np.sign(light_to_center[1])
    ):
        return False

    return True


def ccs_flare_params(distance, image_shape, power):
    ori_h, ori_w = image_shape[:2]
    base_radius = min(ori_h, ori_w) // 2
    min_radius = min(ori_h, ori_w) // 8
    normalized_distance = min(max(distance, 1.0), 30.0)
    log_factor = 1.0 - (math.log(normalized_distance) / math.log(30.0))
    blur_radius = int(min_radius + (base_radius - min_radius) * max(0.0, log_factor))

    intensity = power / (max(distance, 1.0) ** 1.5)
    intensity = min(1.0, intensity * 0.02)
    intensity = max(float(intensity), 0.6)
    return blur_radius, intensity


def render_ccs_lens_flare_bgr(image_bgr, u, v, radius, intensity):
    """Raw-image equivalent of CCS generate_lens_flare for BGR StreamMapNet input."""
    ori_h, ori_w = image_bgr.shape[:2]
    flare = np.zeros_like(image_bgr, dtype=np.float32)
    flare = np.ascontiguousarray(flare)
    cv2.circle(flare, (int(u), int(v)), int(radius), (255.0, 255.0, 255.0), -1)

    kernel_size = min(2 * int(radius) + 1, min(ori_h, ori_w) - 1)
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    flare = cv2.GaussianBlur(flare, (kernel_size, kernel_size), radius / 2)

    # StreamMapNet uses BGR images with to_rgb=False, so channel 0 is blue.
    flare[:, :, 0] *= 1.1
    flare = np.clip(flare, 0, 255)

    result = cv2.addWeighted(image_bgr.astype(np.float32), 1.0, flare, intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def render_ccs_camera(sample, cam_name, attack_loc_lidar, power):
    cam_info = sample["cams"][cam_name]
    clean_img = cv2.imread(cam_info["img_fpath"], cv2.IMREAD_COLOR)
    if clean_img is None:
        raise FileNotFoundError(cam_info["img_fpath"])

    lidar2ego = transform_matrix(
        sample["lidar2ego_rotation"], sample["lidar2ego_translation"]
    )
    attack_loc_ego = (lidar2ego @ homogeneous(attack_loc_lidar))[:3]
    _, u, v, depth = project_ego_to_camera(attack_loc_ego, cam_info)

    img_h, img_w = clean_img.shape[:2]
    if not math.isfinite(u):
        u = 0.0
    if not math.isfinite(v):
        v = 0.0
    is_visible = depth > 1e-6 and 0 <= u < img_w and 0 <= v < img_h
    u = float(np.clip(u, 0, img_w - 1))
    v = float(np.clip(v, 0, img_h - 1))

    camera_params = camera_pose_in_lidar(sample, cam_name)
    affected = is_camera_affected(camera_params, attack_loc_lidar)
    distance = float(np.linalg.norm(attack_loc_lidar[:2] - camera_params["position"][:2]))
    radius, intensity = ccs_flare_params(distance, clean_img.shape, power)

    if affected:
        attacked_img = render_ccs_lens_flare_bgr(clean_img, u, v, radius, intensity)
    else:
        attacked_img = clean_img.copy()

    return {
        "camera": cam_name,
        "clean_img": clean_img,
        "attacked_img": attacked_img,
        "clean_img_path": cam_info["img_fpath"],
        "affected": bool(affected),
        "is_visible": bool(is_visible),
        "u": u,
        "v": v,
        "depth": float(depth),
        "distance_to_camera_xy": distance,
        "glare_radius_px": int(radius),
        "intensity": float(intensity),
    }


def render_ccs_all_cameras(sample, attack_loc_lidar, power):
    return [
        render_ccs_camera(sample, cam_name, np.asarray(attack_loc_lidar, dtype=np.float64), power)
        for cam_name in CAMERA_ORDER
    ]

