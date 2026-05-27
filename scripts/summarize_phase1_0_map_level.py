#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyquaternion import Quaternion


LABEL_BOUNDARY = 2


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


def chamfer(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return math.inf
    diff = a[:, None, :] - b[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return float((dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2.0)


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


def global_polyline_to_ego_xy(polyline_global, current_global2ego):
    points = [(current_global2ego @ homogeneous(point))[:2] for point in polyline_global]
    return np.asarray(points, dtype=np.float64)


def denormalize_vectors(vectors, roi_size):
    vectors = np.asarray(vectors, dtype=np.float64)
    origin = -np.array([roi_size[0] / 2.0, roi_size[1] / 2.0], dtype=np.float64)
    return vectors * (np.asarray(roi_size, dtype=np.float64) + 1e-5) + origin


def load_outputs_records(condition_dir, roi_size):
    outputs_path = condition_dir / "outputs.pkl"
    if outputs_path.exists():
        rows = load_pickle(outputs_path)
        records = {}
        for pred in rows:
            records[pred["token"]] = {
                "vectors": denormalize_vectors(pred["vectors"], roi_size),
                "scores": np.asarray(pred["scores"], dtype=np.float64),
                "labels": np.asarray(pred["labels"], dtype=np.int64),
                "prop": np.asarray(pred.get("prop_mask", []), dtype=bool),
                "source": str(outputs_path),
            }
        return records

    query_root = condition_dir / "query_memory"
    if query_root.exists():
        records = {}
        for path in sorted(query_root.glob("scene-*/*.pt")):
            payload = torch.load(path, map_location="cpu")
            scores = payload["all_query_scores_sigmoid"].max(dim=1).values.numpy()
            records[payload["token"]] = {
                "vectors": denormalize_vectors(payload["all_query_pred_vectors"].numpy(), roi_size),
                "scores": scores,
                "labels": payload["all_query_labels"].numpy().astype(np.int64),
                "prop": payload["propagated_query_mask"].numpy().astype(bool),
                "source": str(path),
            }
        return records

    raise FileNotFoundError(f"No outputs.pkl or query_memory found under {condition_dir}")


def best_boundary_metrics(record, diverge_xy, reference_xy, score_thr, sample_interval):
    labels = record["labels"]
    scores = record["scores"]
    vectors = record["vectors"]
    mask = (labels == LABEL_BOUNDARY) & (scores >= score_thr)
    indices = np.flatnonzero(mask)
    candidates = []
    for idx in indices:
        pred = resample_polyline(vectors[idx], sample_interval)
        cd_diverge = chamfer(pred, diverge_xy)
        cd_reference = chamfer(pred, reference_xy)
        candidates.append(
            {
                "idx": int(idx),
                "score": float(scores[idx]),
                "prop": bool(record["prop"][idx]) if len(record["prop"]) > idx else False,
                "cd_to_diverge_m": cd_diverge,
                "cd_to_reference_m": cd_reference,
                "wrong_reference_preference_m": cd_diverge - cd_reference,
            }
        )
    if not candidates:
        return {
            "num_boundary_preds": 0,
            "best_idx": -1,
            "best_score": math.nan,
            "best_prop": False,
            "cd_to_diverge_m": math.inf,
            "cd_to_reference_m": math.inf,
            "wrong_reference_preference_m": math.inf,
        }
    best = min(candidates, key=lambda item: item["cd_to_diverge_m"])
    return {
        "num_boundary_preds": int(len(indices)),
        "best_idx": best["idx"],
        "best_score": best["score"],
        "best_prop": best["prop"],
        "cd_to_diverge_m": best["cd_to_diverge_m"],
        "cd_to_reference_m": best["cd_to_reference_m"],
        "wrong_reference_preference_m": best["wrong_reference_preference_m"],
    }


def plot_frame(out_path, title, diverge_xy, reference_xy, condition_records, token, score_thr):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(condition_records)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (condition, record) in zip(axes, condition_records.items()):
        ax.plot(diverge_xy[:, 0], diverge_xy[:, 1], color="crimson", linewidth=2.4, label="diverge GT")
        ax.plot(reference_xy[:, 0], reference_xy[:, 1], color="seagreen", linewidth=2.0, label="reference")
        mask = (record["labels"] == LABEL_BOUNDARY) & (record["scores"] >= score_thr)
        boundary_indices = np.flatnonzero(mask)
        order = boundary_indices[np.argsort(record["scores"][boundary_indices])[::-1]][:12]
        for idx in order:
            pred = record["vectors"][idx]
            ax.plot(pred[:, 0], pred[:, 1], color="royalblue", alpha=0.35, linewidth=1.2)
        ax.set_title(condition, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-15, 15)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title} | token={token[:8]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def finite(value):
    return value if math.isfinite(float(value)) else ""


def main():
    parser = argparse.ArgumentParser(description="Summarize single-sequence map-level boundary residue.")
    parser.add_argument("--ann-file", default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann.pkl")
    parser.add_argument("--asset-csv", default="/data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv")
    parser.add_argument("--hook-root", default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_map_level")
    parser.add_argument("--target-token", default="")
    parser.add_argument("--offsets", default="0,1,2")
    parser.add_argument("--roi-size", default="60,30")
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--boundary-z", type=float, default=-1.84)
    parser.add_argument("--sample-interval-m", type=float, default=0.25)
    args = parser.parse_args()

    ann = load_pickle(args.ann_file)
    target_token = args.target_token or ann[0]["mapecho_target_token"]
    offsets = [int(item.strip()) for item in args.offsets.split(",") if item.strip()]
    roi_size = [float(item.strip()) for item in args.roi_size.split(",")]
    out_dir = Path(args.out_dir)

    asset_rows = {row["sample_token"]: row for row in load_csv(args.asset_csv)}
    if target_token not in asset_rows:
        raise KeyError(f"Missing target token in asset CSV: {target_token}")
    asset = asset_rows[target_token]
    target_sample = next(sample for sample in ann if sample["token"] == target_token)
    samples_by_offset = {int(sample["mapecho_frame_offset"]): sample for sample in ann}

    _, diverge_lidar, reference_lidar = extract_scene_boundaries(asset["scene_json"])
    target_lidar2ego = transform_matrix(
        target_sample["lidar2ego_rotation"], target_sample["lidar2ego_translation"]
    )
    target_ego2global = transform_matrix(
        target_sample["e2g_rotation"], target_sample["e2g_translation"]
    )
    diverge_global = lidar_polyline_to_global(
        resample_polyline(diverge_lidar, args.sample_interval_m),
        args.boundary_z,
        target_lidar2ego,
        target_ego2global,
    )
    reference_global = lidar_polyline_to_global(
        resample_polyline(reference_lidar, args.sample_interval_m),
        args.boundary_z,
        target_lidar2ego,
        target_ego2global,
    )

    hook_root = Path(args.hook_root)
    condition_dirs = {
        "clean_keep": hook_root / "phase1_0_clean_keep",
        "clean_reset_all": hook_root / "phase1_0_reset_sanity" / "reset_all",
        "clean_reset_query": hook_root / "phase1_0_reset_sanity" / "reset_query",
        "clean_reset_bev": hook_root / "phase1_0_reset_sanity" / "reset_bev",
        "attack_keep": hook_root / "phase1_0_attack_reset_ablation" / "attack_keep",
        "attack_reset_all": hook_root / "phase1_0_attack_reset_ablation" / "attack_reset_all",
        "attack_reset_query": hook_root / "phase1_0_attack_reset_ablation" / "attack_reset_query",
        "attack_reset_bev": hook_root / "phase1_0_attack_reset_ablation" / "attack_reset_bev",
    }
    condition_records = {
        name: load_outputs_records(path, roi_size) for name, path in condition_dirs.items()
    }

    rows = []
    for offset in offsets:
        sample = samples_by_offset[offset]
        current_global2ego = invert_rigid(
            transform_matrix(sample["e2g_rotation"], sample["e2g_translation"])
        )
        diverge_xy = resample_polyline(
            global_polyline_to_ego_xy(diverge_global, current_global2ego),
            args.sample_interval_m,
        )
        reference_xy = resample_polyline(
            global_polyline_to_ego_xy(reference_global, current_global2ego),
            args.sample_interval_m,
        )
        for condition, records in condition_records.items():
            metrics = best_boundary_metrics(
                records[sample["token"]],
                diverge_xy,
                reference_xy,
                args.score_thr,
                args.sample_interval_m,
            )
            rows.append(
                {
                    "condition": condition,
                    "frame_offset": offset,
                    "sample_idx": sample["sample_idx"],
                    "sample_token": sample["token"],
                    "num_boundary_preds": metrics["num_boundary_preds"],
                    "best_idx": metrics["best_idx"],
                    "best_score": finite(metrics["best_score"]),
                    "best_prop": metrics["best_prop"],
                    "cd_to_diverge_m": finite(metrics["cd_to_diverge_m"]),
                    "cd_to_reference_m": finite(metrics["cd_to_reference_m"]),
                    "wrong_reference_preference_m": finite(metrics["wrong_reference_preference_m"]),
                }
            )
        plot_conditions = {
            key: condition_records[key][sample["token"]]
            for key in [
                "clean_keep",
                "attack_keep",
                "attack_reset_all",
                "attack_reset_query",
                "attack_reset_bev",
            ]
        }
        plot_frame(
            out_dir / "figures" / f"offset_{offset:+d}_boundary_overlay.png",
            f"Phase1.0 map-level boundary offset {offset:+d}",
            diverge_xy,
            reference_xy,
            plot_conditions,
            sample["token"],
            args.score_thr,
        )

    row_by_key = {(row["condition"], int(row["frame_offset"])): row for row in rows}
    pairs = {
        "attack_keep": "clean_keep",
        "attack_reset_all": "clean_reset_all",
        "attack_reset_query": "clean_reset_query",
        "attack_reset_bev": "clean_reset_bev",
    }
    delta_rows = []
    for attack_cond, clean_cond in pairs.items():
        for offset in offsets:
            attack = row_by_key[(attack_cond, offset)]
            clean = row_by_key[(clean_cond, offset)]
            def diff(field):
                if attack[field] == "" or clean[field] == "":
                    return ""
                return float(attack[field]) - float(clean[field])
            delta_rows.append(
                {
                    "attack_condition": attack_cond,
                    "matched_clean_condition": clean_cond,
                    "frame_offset": offset,
                    "sample_idx": attack["sample_idx"],
                    "sample_token": attack["sample_token"],
                    "delta_cd_to_diverge_m": diff("cd_to_diverge_m"),
                    "delta_cd_to_reference_m": diff("cd_to_reference_m"),
                    "delta_wrong_reference_preference_m": diff("wrong_reference_preference_m"),
                    "attack_cd_to_diverge_m": attack["cd_to_diverge_m"],
                    "clean_cd_to_diverge_m": clean["cd_to_diverge_m"],
                    "attack_wrong_reference_preference_m": attack["wrong_reference_preference_m"],
                    "clean_wrong_reference_preference_m": clean["wrong_reference_preference_m"],
                }
            )

    metrics_fields = [
        "condition",
        "frame_offset",
        "sample_idx",
        "sample_token",
        "num_boundary_preds",
        "best_idx",
        "best_score",
        "best_prop",
        "cd_to_diverge_m",
        "cd_to_reference_m",
        "wrong_reference_preference_m",
    ]
    delta_fields = [
        "attack_condition",
        "matched_clean_condition",
        "frame_offset",
        "sample_idx",
        "sample_token",
        "delta_cd_to_diverge_m",
        "delta_cd_to_reference_m",
        "delta_wrong_reference_preference_m",
        "attack_cd_to_diverge_m",
        "clean_cd_to_diverge_m",
        "attack_wrong_reference_preference_m",
        "clean_wrong_reference_preference_m",
    ]
    write_csv(out_dir / "phase1_0_single_sequence_map_metrics.csv", rows, metrics_fields)
    write_csv(out_dir / "phase1_0_single_sequence_map_matched_deltas.csv", delta_rows, delta_fields)

    summary = {
        "target_token": target_token,
        "scene_name": target_sample["scene_name"],
        "offsets": offsets,
        "score_thr": args.score_thr,
        "metrics_csv": str(out_dir / "phase1_0_single_sequence_map_metrics.csv"),
        "matched_delta_csv": str(out_dir / "phase1_0_single_sequence_map_matched_deltas.csv"),
        "figures_dir": str(out_dir / "figures"),
    }
    (out_dir / "phase1_0_single_sequence_map_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
