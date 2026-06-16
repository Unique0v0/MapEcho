#!/usr/bin/env python3
"""Build a sequence annotation with CCS-style six-camera rendering on N frames.

The glare source is defined in the target frame and converted to a fixed global
point once. For every perturbed frame, that fixed world point is transformed
back into the current LiDAR frame before calling the CCS renderer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from ccs_blind_renderer import render_ccs_all_cameras
from smoke_attack_rendering_injection import homogeneous, invert_rigid, transform_matrix


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def point3_str(point):
    return " ".join(f"{float(value):.6f}" for value in point)


def write_overlay(path, image, cam_result):
    overlay = image.copy()
    center = (int(round(cam_result["u"])), int(round(cam_result["v"])))
    color = (0, 220, 255) if cam_result.get("affected") else (160, 160, 160)
    cv2.circle(overlay, center, int(cam_result["glare_radius_px"]), color, 3)
    cv2.circle(overlay, center, 9, (255, 0, 0), -1)
    label = (
        f"{cam_result['camera']} affected={cam_result.get('affected')} "
        f"visible={cam_result.get('is_visible')}"
    )
    cv2.putText(
        overlay,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), overlay)


def write_contact_sheet(path, cam_results):
    thumbs = []
    for result in cam_results:
        img = result["attacked_img"]
        thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
        label = (
            f"{result['camera']} affected={result['affected']} "
            f"visible={result['is_visible']}"
        )
        cv2.putText(
            thumb,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumb,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    row1 = np.hstack(thumbs[:3])
    row2 = np.hstack(thumbs[3:])
    cv2.imwrite(str(path), np.vstack([row1, row2]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build StreamMapNet sequence ann with CCS six-camera rendering on consecutive frames."
    )
    parser.add_argument("--clean-ann", required=True)
    parser.add_argument("--asset-csv", default="")
    parser.add_argument("--out-ann", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attack-objective", choices=["eta", "rsa"], default="eta")
    parser.add_argument("--source-frame", choices=["lidar", "ego"], default="lidar")
    parser.add_argument("--loc-x", type=float, default=None)
    parser.add_argument("--loc-y", type=float, default=None)
    parser.add_argument("--loc-z", type=float, default=None)
    parser.add_argument("--loc-method", default="")
    parser.add_argument("--power", type=float, default=3000.0)
    parser.add_argument("--attack-start-offset", type=int, default=0)
    parser.add_argument("--num-attack-frames", type=int, default=3)
    parser.add_argument(
        "--renderer",
        choices=["ccs"],
        default="ccs",
        help="Use the CCS-style lens-flare renderer. Simplified renderers are disabled.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["all"],
        default="all",
        help="Replace all six camera files on every perturbed frame.",
    )
    return parser.parse_args()


def resolve_source_point(args, target_token):
    explicit_loc_values = [args.loc_x, args.loc_y, args.loc_z]
    if any(value is not None for value in explicit_loc_values):
        if not all(value is not None for value in explicit_loc_values):
            raise ValueError("--loc-x, --loc-y, and --loc-z must be provided together")
        return np.asarray(explicit_loc_values, dtype=np.float64), (
            args.loc_method or "explicit_location"
        )

    if not args.asset_csv:
        raise ValueError("--asset-csv is required unless explicit location is provided")
    asset_rows = load_csv(args.asset_csv)
    loc_prefix = f"blind_{args.attack_objective}"
    has_loc_key = f"has_blind_{args.attack_objective}_loc"
    matching_assets = [
        row
        for row in asset_rows
        if row["sample_token"] == target_token
        and str(row.get(has_loc_key, "")).lower() == "true"
    ]
    if not matching_assets:
        raise KeyError(f"no {args.attack_objective} location for {target_token}")
    asset = matching_assets[0]
    point_source = np.array(
        [
            float(asset[f"{loc_prefix}_x"]),
            float(asset[f"{loc_prefix}_y"]),
            float(asset[f"{loc_prefix}_z"]),
        ],
        dtype=np.float64,
    )
    return point_source, asset.get("mapecho_loc_method", f"{args.attack_objective}_asset_csv")


def current_lidar_point_from_global(sample, point_global):
    current_lidar2ego = transform_matrix(
        sample["lidar2ego_rotation"], sample["lidar2ego_translation"]
    )
    current_ego2lidar = invert_rigid(current_lidar2ego)
    current_ego2global = transform_matrix(sample["e2g_rotation"], sample["e2g_translation"])
    current_global2ego = invert_rigid(current_ego2global)
    point_ego = (current_global2ego @ homogeneous(point_global))[:3]
    point_lidar = (current_ego2lidar @ homogeneous(point_ego))[:3]
    return point_lidar, point_ego


def main():
    args = parse_args()
    if args.num_attack_frames < 1:
        raise ValueError("--num-attack-frames must be >= 1")

    samples = [copy.deepcopy(sample) for sample in load_pickle(args.clean_ann)]
    target_tokens = {sample.get("mapecho_target_token") for sample in samples}
    if len(target_tokens) != 1:
        raise ValueError(f"expected one target token in clean ann, got {target_tokens}")
    target_token = next(iter(target_tokens))

    target = next(sample for sample in samples if sample["token"] == target_token)
    if target.get("mapecho_frame_offset") != 0:
        raise ValueError(
            f"target frame offset must be 0, got {target.get('mapecho_frame_offset')}"
        )

    point_source, loc_method = resolve_source_point(args, target_token)
    target_lidar2ego = transform_matrix(
        target["lidar2ego_rotation"], target["lidar2ego_translation"]
    )
    target_ego2lidar = invert_rigid(target_lidar2ego)
    target_ego2global = transform_matrix(target["e2g_rotation"], target["e2g_translation"])
    if args.source_frame == "lidar":
        point_lidar_t = point_source
        point_ego_t = (target_lidar2ego @ homogeneous(point_source))[:3]
    else:
        point_ego_t = point_source
        point_lidar_t = (target_ego2lidar @ homogeneous(point_source))[:3]
    point_global = (target_ego2global @ homogeneous(point_ego_t))[:3]

    attack_offsets = list(
        range(args.attack_start_offset, args.attack_start_offset + args.num_attack_frames)
    )
    samples_by_offset = {int(sample["mapecho_frame_offset"]): sample for sample in samples}
    missing_offsets = [offset for offset in attack_offsets if offset not in samples_by_offset]
    if missing_offsets:
        raise KeyError(f"missing perturbation offsets in clean ann: {missing_offsets}")

    out_dir = Path(args.out_dir)
    attacked_sample_count = 0
    frame_summaries = []
    for sample in samples:
        sample["mapecho_attack_schedule"] = "clean"
        sample["mapecho_num_attack_frames"] = args.num_attack_frames
        sample["mapecho_attack_start_offset"] = args.attack_start_offset
        sample["mapecho_attack_offsets"] = attack_offsets

    for offset in attack_offsets:
        sample = samples_by_offset[offset]
        sample_token = sample["token"]
        point_lidar_current, point_ego_current = current_lidar_point_from_global(
            sample, point_global
        )
        cam_results = render_ccs_all_cameras(sample, point_lidar_current, args.power)

        image_dir = out_dir / "images" / f"offset_{offset:+03d}_{sample_token}"
        image_dir.mkdir(parents=True, exist_ok=True)
        camera_artifacts = {}
        for cam_result in cam_results:
            cam_name = cam_result["camera"]
            attacked_img_path = image_dir / f"{cam_name}_attacked.png"
            overlay_path = image_dir / f"{cam_name}_overlay.png"
            cv2.imwrite(str(attacked_img_path), cam_result["attacked_img"])
            write_overlay(overlay_path, cam_result["attacked_img"], cam_result)
            cam_result["attacked_img_path"] = str(attacked_img_path)
            cam_result["overlay_path"] = str(overlay_path)
            camera_artifacts[cam_name] = cam_result

        contact_sheet_path = image_dir / "six_camera_attacked_contact_sheet.png"
        if len(cam_results) == 6:
            write_contact_sheet(contact_sheet_path, cam_results)

        replaced_camera_names = sorted(camera_artifacts)
        for cam_name, cam_result in camera_artifacts.items():
            sample["cams"][cam_name]["mapecho_clean_img_fpath"] = cam_result["clean_img_path"]
            sample["cams"][cam_name]["img_fpath"] = cam_result["attacked_img_path"]
        sample["mapecho_attack_schedule"] = "attacked"
        sample["mapecho_attack_renderer"] = args.renderer
        sample["mapecho_attack_camera_mode"] = args.camera_mode
        sample["mapecho_attack_cameras_replaced"] = replaced_camera_names
        sample["mapecho_attack_cameras_affected"] = [
            cam_name for cam_name, result in camera_artifacts.items() if result["affected"]
        ]
        sample["mapecho_attack_fixed_global_xyz"] = point3_str(point_global)
        sample["mapecho_attack_lidar_current_xyz"] = point3_str(point_lidar_current)
        sample["mapecho_attack_ego_current_xyz"] = point3_str(point_ego_current)
        attacked_sample_count += 1

        frame_summaries.append(
            {
                "offset": offset,
                "sample_token": sample_token,
                "point_lidar_current": point3_str(point_lidar_current),
                "point_ego_current": point3_str(point_ego_current),
                "affected_camera_count": sum(result["affected"] for result in cam_results),
                "affected_cameras": [
                    result["camera"] for result in cam_results if result["affected"]
                ],
                "visible_cameras": [
                    result["camera"] for result in cam_results if result["is_visible"]
                ],
                "contact_sheet_path": str(contact_sheet_path) if len(cam_results) == 6 else "",
                "camera_artifacts": {
                    cam_name: {
                        "clean_img_path": result["clean_img_path"],
                        "attacked_img_path": result["attacked_img_path"],
                        "overlay_path": result["overlay_path"],
                        "affected": result["affected"],
                        "is_visible": result["is_visible"],
                        "u": result["u"],
                        "v": result["v"],
                        "depth": result["depth"],
                        "distance_to_camera_xy": result["distance_to_camera_xy"],
                        "glare_radius_px": result["glare_radius_px"],
                        "intensity": result["intensity"],
                    }
                    for cam_name, result in camera_artifacts.items()
                },
            }
        )

    out_ann = Path(args.out_ann)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    with out_ann.open("wb") as f:
        pickle.dump(samples, f)

    summary = {
        "clean_ann": str(Path(args.clean_ann).resolve()),
        "out_ann": str(out_ann),
        "out_dir": str(out_dir),
        "attack_objective": args.attack_objective,
        "source_frame": args.source_frame,
        "target_token": target_token,
        "target_scene": target["scene_name"],
        "target_sample_idx": target["sample_idx"],
        "renderer": args.renderer,
        "camera_mode": args.camera_mode,
        "attack_power": args.power,
        "loc_method": loc_method,
        "attack_start_offset": args.attack_start_offset,
        "num_attack_frames": args.num_attack_frames,
        "attack_offsets": attack_offsets,
        "attacked_sample_count": attacked_sample_count,
        "num_frames": len(samples),
        "p_source": point3_str(point_source),
        "p_lidar_t": point3_str(point_lidar_t),
        "p_global": point3_str(point_global),
        "fixed_global_point": True,
        "six_camera_replacement_per_attacked_frame": True,
        "pass_n_attack_frames": attacked_sample_count == args.num_attack_frames,
        "frame_summaries": frame_summaries,
    }
    summary_path = out_dir / "multi_frame_glare_ann_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
