#!/usr/bin/env python3
import argparse
import csv
import copy
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from smoke_attack_rendering_injection import (
    choose_attack_camera,
    glare_intensity,
    homogeneous,
    invert_rigid,
    render_glare,
    transform_matrix,
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def point3_str(point):
    return " ".join(f"{float(value):.6f}" for value in point)


def main():
    parser = argparse.ArgumentParser(
        description="Build a StreamMapNet sequence ann with ETA camera-blinding injected only at target frame t."
    )
    parser.add_argument("--clean-ann", required=True)
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--out-ann", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attack-objective", choices=["eta", "rsa"], default="eta")
    parser.add_argument("--source-frame", choices=["lidar", "ego"], default="lidar")
    parser.add_argument("--power", type=float, default=3000.0)
    args = parser.parse_args()

    samples = [copy.deepcopy(sample) for sample in load_pickle(args.clean_ann)]
    asset_rows = load_csv(args.asset_csv)
    target_tokens = {sample.get("mapecho_target_token") for sample in samples}
    if len(target_tokens) != 1:
        raise ValueError(f"expected one target token in clean ann, got {target_tokens}")
    target_token = next(iter(target_tokens))

    target = next(sample for sample in samples if sample["token"] == target_token)
    target_offset = target.get("mapecho_frame_offset")
    if target_offset != 0:
        raise ValueError(f"target frame offset must be 0, got {target_offset}")

    loc_prefix = f"blind_{args.attack_objective}"
    has_loc_key = f"has_blind_{args.attack_objective}_loc"
    matching_assets = [
        row
        for row in asset_rows
        if row["sample_token"] == target_token
        and str(row.get(has_loc_key, "")).lower() == "true"
    ]
    if not matching_assets:
        raise KeyError(f"no {args.attack_objective} attack location for {target_token}")
    asset = matching_assets[0]

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

    target_global2ego = invert_rigid(target_ego2global)
    point_ego = (target_global2ego @ homogeneous(point_global))[:3]
    attack_camera = choose_attack_camera(target, point_ego)
    if attack_camera is None:
        raise RuntimeError(f"attack point is not visible in target frame {target_token}")

    clean_img_path = attack_camera["cam_info"]["img_fpath"]
    clean = cv2.imread(clean_img_path, cv2.IMREAD_COLOR)
    if clean is None:
        raise FileNotFoundError(clean_img_path)
    intensity = glare_intensity(attack_camera["distance_to_camera"], args.power)
    attacked = render_glare(
        clean,
        attack_camera["u"],
        attack_camera["v"],
        attack_camera["glare_radius_px"],
        intensity,
    )

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images" / target_token
    image_dir.mkdir(parents=True, exist_ok=True)
    attacked_img_path = image_dir / f"{attack_camera['camera']}_attacked.png"
    overlay_path = image_dir / f"{attack_camera['camera']}_overlay.png"
    cv2.imwrite(str(attacked_img_path), attacked)

    overlay = attacked.copy()
    center = (int(round(attack_camera["u"])), int(round(attack_camera["v"])))
    cv2.circle(overlay, center, int(attack_camera["glare_radius_px"]), (0, 220, 255), 3)
    cv2.circle(overlay, center, 9, (255, 0, 0), -1)
    cv2.imwrite(str(overlay_path), overlay)

    attacked_sample_count = 0
    attacked_camera_name = attack_camera["camera"]
    for sample in samples:
        sample["mapecho_attack_schedule"] = "clean"
        if sample["token"] != target_token:
            continue
        sample["cams"][attacked_camera_name]["mapecho_clean_img_fpath"] = clean_img_path
        sample["cams"][attacked_camera_name]["img_fpath"] = str(attacked_img_path)
        sample["mapecho_attack_schedule"] = "attacked"
        sample["mapecho_attack_camera"] = attacked_camera_name
        attacked_sample_count += 1

    out_ann = Path(args.out_ann)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann, "wb") as f:
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
        "target_frame_offset": target_offset,
        "attacked_sample_count": attacked_sample_count,
        "num_frames": len(samples),
        "attack_camera": attacked_camera_name,
        "clean_img_path": clean_img_path,
        "attacked_img_path": str(attacked_img_path),
        "overlay_path": str(overlay_path),
        "p_source": point3_str(point_source),
        "p_global": point3_str(point_global),
        "u": attack_camera["u"],
        "v": attack_camera["v"],
        "distance_to_camera": attack_camera["distance_to_camera"],
        "glare_radius_px": attack_camera["glare_radius_px"],
        "intensity": intensity,
        "pass_n_attack_1": attacked_sample_count == 1,
        "pass_raw_uint8": clean.dtype == np.uint8 and attacked.dtype == np.uint8,
        "pass_shape_unchanged": clean.shape == attacked.shape,
    }
    summary_path = out_dir / "attack_at_t_ann_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
