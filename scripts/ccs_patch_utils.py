#!/usr/bin/env python3
import copy
import math

import numpy as np
import torch
from mmdet3d.core import LiDARInstance3DBoxes
from torchvision.transforms.functional import InterpolationMode, perspective


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def normalize_img(img, img_norm_cfg):
    mean = torch.tensor(img_norm_cfg["mean"], dtype=img.dtype, device=img.device).reshape(1, 1, 3, 1, 1)
    std = torch.tensor(img_norm_cfg["std"], dtype=img.dtype, device=img.device).reshape(1, 1, 3, 1, 1)
    return (img - mean) / std


def denormalize_img(img, img_norm_cfg):
    mean = torch.tensor(img_norm_cfg["mean"], dtype=img.dtype, device=img.device).reshape(1, 1, 3, 1, 1)
    std = torch.tensor(img_norm_cfg["std"], dtype=img.dtype, device=img.device).reshape(1, 1, 3, 1, 1)
    return img * std + mean


def normalize_patch(patch, img_norm_cfg):
    mean = torch.tensor(img_norm_cfg["mean"], dtype=patch.dtype, device=patch.device).reshape(3, 1, 1)
    std = torch.tensor(img_norm_cfg["std"], dtype=patch.dtype, device=patch.device).reshape(3, 1, 1)
    return (patch - mean) / std


def transform_matrix(rotation_quat, translation):
    from pyquaternion import Quaternion

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def invert_rigid(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inv


def sample_boundary_at_interval(boundary_pts, interval=0.5):
    from shapely.geometry import LineString

    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)
    line = LineString(boundary_pts)
    total_length = line.length
    num_points = max(2, int(total_length / interval) + 1)
    sampled = []
    for i in range(num_points):
        distance = i * interval
        if distance > total_length:
            break
        point = line.interpolate(distance)
        sampled.append((point.x, point.y))
    return np.asarray(sampled, dtype=np.float64)


def extend_boundary(reference_boundary, incomplete_boundary, step=5):
    from scipy.interpolate import interp1d

    reference_boundary = np.asarray(reference_boundary, dtype=np.float64)
    incomplete_boundary = np.asarray(incomplete_boundary, dtype=np.float64)
    f_reference = interp1d(
        reference_boundary[:, 1],
        reference_boundary[:, 0],
        kind="linear",
        fill_value="extrapolate",
    )
    f_incomplete = interp1d(
        incomplete_boundary[:, 1],
        incomplete_boundary[:, 0],
        kind="linear",
        fill_value="extrapolate",
    )
    measure_points = min(step, len(incomplete_boundary))
    y_samples = incomplete_boundary[:measure_points, 1]
    road_width_samples = f_incomplete(y_samples) - f_reference(y_samples)
    avg_road_width = np.mean(road_width_samples)
    mirrored_boundary = reference_boundary.copy()
    mirrored_boundary[:, 0] += avg_road_width
    return mirrored_boundary, avg_road_width


def stitch_boundaries(incomplete_boundary, mirrored_boundary, step=5):
    incomplete_boundary = np.asarray(incomplete_boundary, dtype=np.float64)
    mirrored_boundary = np.asarray(mirrored_boundary, dtype=np.float64)
    reliable_part = incomplete_boundary[:step].tolist()
    y_cutoff = incomplete_boundary[step - 1][1] if step > 0 else incomplete_boundary[0][1]
    mirrored_part = [point for point in mirrored_boundary if point[1] >= y_cutoff]
    return np.asarray(reliable_part + mirrored_part, dtype=np.float64)


def get_target_boundary_pts(diverge_boundary_pts, reference_boundary_pts, diverge_boundary_tag, dataset, step=5):
    if dataset != "asymmetric":
        raise ValueError("MapEcho patch_rsa currently expects the asymmetric CCS setting")
    extended_diverge_boundary, _ = extend_boundary(reference_boundary_pts, diverge_boundary_pts, step=step)
    return stitch_boundaries(diverge_boundary_pts, extended_diverge_boundary, step=step)


def torch_chamfer_distance(points_a, points_b):
    distances = torch.norm(points_a.unsqueeze(1) - points_b.unsqueeze(0), dim=2)
    return distances.min(dim=1)[0].mean() + distances.min(dim=0)[0].mean()


def find_best_matching_boundary_torch(polylines, target_polyline):
    if polylines.numel() == 0 or polylines.shape[0] == 0:
        return None
    best_polyline = None
    best_dist = None
    for polyline in polylines:
        valid = polyline[polyline[:, 0] != -10000]
        if valid.numel() == 0:
            continue
        dist = torch_chamfer_distance(valid, target_polyline)
        if best_dist is None or dist.detach().item() < best_dist.detach().item():
            best_dist = dist
            best_polyline = valid
    return best_polyline


def denormalize_streammapnet_lines(lines, roi_size=(60.0, 30.0)):
    roi = torch.tensor(roi_size, dtype=lines.dtype, device=lines.device)
    origin = torch.tensor([-roi_size[0] / 2.0, -roi_size[1] / 2.0], dtype=lines.dtype, device=lines.device)
    return lines * (roi + 1e-5) + origin


def generate_sampled_points(center, grid_size=1.0, num_points=1, mode="random"):
    center = np.asarray(center, dtype=np.float64)
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    if mode == "left":
        x = np.random.rand(num_points) * grid_size + (center[0] - grid_size)
        y = np.random.rand(num_points) * grid_size - grid_size / 2.0 + center[1]
    elif mode == "right":
        x = np.random.rand(num_points) * grid_size + center[0]
        y = np.random.rand(num_points) * grid_size - grid_size / 2.0 + center[1]
    elif mode == "random":
        x = np.random.rand(num_points) * grid_size - grid_size / 2.0 + center[0]
        y = np.random.rand(num_points) * grid_size - grid_size / 2.0 + center[1]
    else:
        raise ValueError(f"invalid sample mode: {mode}")
    return np.stack([x, y], axis=1)


def get_patch_heading_facing_ego(point_xy):
    x, y = float(point_xy[0]), float(point_xy[1])
    bev_heading = np.arctan2(y, x)
    bev_heading_perpendicular = bev_heading - np.pi / 2.0
    lidar_heading = -bev_heading_perpendicular
    return float((lidar_heading + np.pi) % (2 * np.pi) - np.pi)


def get_proj_scale(lat_dist, long_dist, ori_img_width, camera_height=1.51):
    focal_length_mm = 5.5
    sensor_width_mm = 7.2
    distance = math.sqrt(camera_height**2 + long_dist**2 + lat_dist**2)
    pixel_size_width = sensor_width_mm / ori_img_width
    focal_length_px = focal_length_mm / pixel_size_width
    return focal_length_px / distance


def create_pseudo_area(patch_cfg, ori_img_shape, proj_scale=50):
    patch_h, patch_w = patch_cfg["height"], patch_cfg["width"]
    _, img_h, img_w = ori_img_shape
    if patch_h > img_h // proj_scale or patch_w > img_w // proj_scale:
        return None
    pseudo_area = (
        img_h - patch_h * proj_scale - 1,
        (img_w - patch_w * proj_scale) // 2,
        patch_h * proj_scale,
        patch_w * proj_scale,
    )
    return tuple(int(v) for v in pseudo_area)


def init_patch_mask(ori_img_shape, device, pseudo_area, mode="random"):
    if mode == "random":
        patch = torch.randn(ori_img_shape, device=device) * 255.0
    elif mode == "zero":
        patch = torch.zeros(ori_img_shape, device=device)
    else:
        raise ValueError(f"unsupported patch init mode: {mode}")
    patch.requires_grad_(True)

    _, img_h, img_w = ori_img_shape
    mask = torch.zeros((1, img_h, img_w), device=device)
    top, left, height, width = pseudo_area
    mask[:, top : top + height, left : left + width] = 1.0
    return patch, mask


def sample_to_lidar2global(sample):
    lidar2ego = transform_matrix(sample["lidar2ego_rotation"], sample["lidar2ego_translation"])
    ego2global = transform_matrix(sample["e2g_rotation"], sample["e2g_translation"])
    return ego2global @ lidar2ego


def sample_to_global2img(sample):
    ego2global = transform_matrix(sample["e2g_rotation"], sample["e2g_translation"])
    global2ego = invert_rigid(ego2global)
    matrices = []
    for cam_name in CAMERA_ORDER:
        cam = sample["cams"][cam_name]
        ego2cam = np.asarray(cam["extrinsics"], dtype=np.float64)
        intrinsic = np.asarray(cam["intrinsics"], dtype=np.float64)
        intrinsic4 = np.eye(4, dtype=np.float64)
        intrinsic4[:3, :3] = intrinsic
        matrices.append(intrinsic4 @ ego2cam @ global2ego)
    return np.stack(matrices, axis=0)


def get_patch_corners_on_img(patch_cfg, lidar2global, global2img, img_shape):
    img_h, img_w = img_shape
    if patch_cfg["type"] == "vertical":
        box = LiDARInstance3DBoxes(
            tensor=[
                [
                    patch_cfg["lat"],
                    patch_cfg["long"],
                    -patch_cfg["lidar2ground"],
                    patch_cfg["width"],
                    0,
                    patch_cfg["height"],
                    patch_cfg["heading"],
                ]
            ],
            origin=(0.5, 0.5, 0.5),
        )
    elif patch_cfg["type"] == "ground":
        box = LiDARInstance3DBoxes(
            tensor=[
                [
                    patch_cfg["lat"],
                    patch_cfg["long"],
                    -patch_cfg["lidar2ground"],
                    patch_cfg["width"],
                    patch_cfg["height"],
                    0,
                    patch_cfg["heading"],
                ]
            ],
            origin=(0.5, 0.5, 0.5),
        )
    else:
        raise ValueError(f"invalid patch type: {patch_cfg['type']}")

    corners = box.corners.reshape(-1, 3).detach().cpu().numpy()
    lidar2global = np.asarray(lidar2global, dtype=np.float64)
    corners_global = corners @ lidar2global[:3, :3].T + lidar2global[:3, 3]
    corners_global[[0, 3, 4, 7], 2] = 0
    corners_global[[1, 2, 5, 6], 2] = float(box.height.item())
    pts_4d = np.concatenate([corners_global, np.ones((8, 1), dtype=np.float64)], axis=1)

    visible = []
    for cam_idx, matrix in enumerate(np.asarray(global2img, dtype=np.float64)):
        pts_2d = pts_4d @ matrix.T
        depth = pts_2d[:, 2]
        if not (depth > 1e-5).any():
            continue
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], 1e-5, 1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        in_img = (
            (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 0] < img_w)
            & (pts_2d[:, 1] >= 0)
            & (pts_2d[:, 1] < img_h)
            & (depth > 1e-5)
        )
        if in_img.sum() <= 0:
            continue
        coords = pts_2d[:, :2]
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        coords = coords[unique_indices]
        center = coords.mean(axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        ordered = coords[np.argsort(angles)]
        visible.append((cam_idx, ordered))
    return visible


def get_phy_patch_mask(patch, mask, patch_cfg, pseudo_area, lidar2global, global2img, img_shape):
    top, left, height, width = pseudo_area
    start = [[left, top], [left + width, top], [left + width, top + height], [left, top + height]]
    visible = get_patch_corners_on_img(patch_cfg, lidar2global, global2img, img_shape)
    if not visible:
        return None, None, None

    patch_list = []
    mask_list = []
    cam_indices = []
    for cam_idx, end in visible:
        patch_trans = perspective(
            patch,
            start,
            end.tolist(),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask_trans = perspective(
            mask,
            start,
            end.tolist(),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        patch_list.append(patch_trans.unsqueeze(0).unsqueeze(0))
        mask_list.append(mask_trans.unsqueeze(0).unsqueeze(0))
        cam_indices.append(cam_idx)
    return patch_list, mask_list, cam_indices


def apply_patch(imgs_adv, patch_trans_list, mask_trans_list, img_norm_cfg, visible_cam_indices):
    for cam_idx, patch_trans, mask_trans in zip(visible_cam_indices, patch_trans_list, mask_trans_list):
        if patch_trans.shape[-2] < imgs_adv.shape[-2]:
            patch_trans = torch.nn.functional.pad(
                patch_trans,
                (0, 0, 0, imgs_adv.shape[-2] - patch_trans.shape[-2], 0, 0),
                "constant",
                0,
            )
        if mask_trans.shape[-2] < imgs_adv.shape[-2]:
            mask_trans = torch.nn.functional.pad(
                mask_trans,
                (0, 0, 0, imgs_adv.shape[-2] - mask_trans.shape[-2], 0, 0),
                "constant",
                0,
            )
        patch_norm = normalize_patch(patch_trans.squeeze(0).squeeze(0), img_norm_cfg)
        patch_norm = patch_norm.unsqueeze(0).unsqueeze(0)
        imgs_adv[:, [cam_idx], ...] = torch.where(mask_trans > 0, patch_norm, imgs_adv[:, [cam_idx], ...])
    return imgs_adv


def snapshot_stream_memory(model):
    def clone_memory(memory):
        return {
            "test_memory_list": [
                item.clone().detach() if isinstance(item, torch.Tensor) else None
                for item in memory.test_memory_list
            ],
            "test_img_metas_memory": copy.deepcopy(memory.test_img_metas_memory),
        }

    snap = {}
    if hasattr(model, "bev_memory"):
        snap["bev_memory"] = clone_memory(model.bev_memory)
    for name in ["query_memory", "reference_points_memory", "target_memory"]:
        if hasattr(model.head, name):
            snap[name] = clone_memory(getattr(model.head, name))
    return snap


def restore_stream_memory(model, snap):
    def restore(memory, payload):
        memory.test_memory_list = [
            item.clone().detach() if isinstance(item, torch.Tensor) else None
            for item in payload["test_memory_list"]
        ]
        memory.test_img_metas_memory = copy.deepcopy(payload["test_img_metas_memory"])

    if "bev_memory" in snap and hasattr(model, "bev_memory"):
        restore(model.bev_memory, snap["bev_memory"])
    for name in ["query_memory", "reference_points_memory", "target_memory"]:
        if name in snap and hasattr(model.head, name):
            restore(getattr(model.head, name), snap[name])
