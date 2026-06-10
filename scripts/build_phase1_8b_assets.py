#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from build_phase1_1_asymmetric_dist_assets import (
    choose_anchor,
    identify_diverging_boundary,
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


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


def load_scene_json(scene_root, token):
    for subdir in ("scenes_asymmetric_dist", "scenes_candidate"):
        path = Path(scene_root) / subdir / f"{token}.json"
        if path.exists():
            return path
    raise FileNotFoundError(f"no scene json for {token} under {scene_root}")


def temporal_flags(scene_pos, scene_len, warmup, recovery):
    return {
        "scene_pos": scene_pos,
        "scene_len": scene_len,
        "has_warmup": scene_pos >= warmup,
        "has_recovery": scene_pos + recovery < scene_len,
        "is_temporal_eligible": scene_pos >= warmup and scene_pos + recovery < scene_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Build Phase 1.8-B assets from CCS rule-based newsplit-val rebuild.")
    parser.add_argument("--rule-dir", default="/data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild")
    parser.add_argument("--tokens", default="")
    parser.add_argument("--newsplit-val-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_8b_assets")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=9)
    parser.add_argument("--light-z", type=float, default=-0.6133333333333333)
    parser.add_argument("--min-anchor-y", type=float, default=3.0)
    parser.add_argument("--max-per-scene", type=int, default=5)
    parser.add_argument("--target-max", type=int, default=120)
    args = parser.parse_args()

    rule_dir = Path(args.rule_dir)
    token_path = Path(args.tokens) if args.tokens else rule_dir / "sample_tokens_asymmetric_curvature.txt"
    tokens = read_tokens(token_path)
    infos = load_pickle(args.newsplit_val_ann)
    by_token, pos_by_token = scene_index(infos)

    out_dir = Path(args.out_dir)
    scene_dir = out_dir / "scene_json"
    scene_dir.mkdir(parents=True, exist_ok=True)

    tag_rows = []
    asset_rows = []
    metadata_rows = []
    for token in tokens:
        if token not in by_token or token not in pos_by_token:
            continue
        sample = by_token[token]
        scene_pos, scene_len = pos_by_token[token]
        flags = temporal_flags(scene_pos, scene_len, args.warmup, args.recovery)

        scene_json_in = load_scene_json(rule_dir, token)
        with scene_json_in.open() as f:
            scene = json.load(f)
        boundaries = {
            element["tag"]: np.asarray(element["coordinates"], dtype=np.float64)
            for element in scene["map_elements"]
        }
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
            "source_stage": "phase1_8b_ccs_rule_asymmetric_curvature",
            "tag_source": "ccs_rule_rebuild_geometry",
            "anchor_idx": int(anchor_idx),
            "anchor_reference_distance_m": reference_dist,
            "anchor_ego_distance_m": ego_dist,
            "source_scene_json": str(scene_json_in),
        }
        scene_json_out = scene_dir / f"{token}.json"
        with scene_json_out.open("w") as f:
            json.dump(scene, f, indent=2)

        base = {
            "sample_token": token,
            "scene_name": sample["scene_name"],
            **flags,
            "scene_json": str(scene_json_out),
        }
        metadata_rows.append(base)
        tag_rows.append(
            {
                **base,
                "boundary_left_id": "left",
                "boundary_right_id": "right",
                "diverge_boundary_id": tag,
                "reference_boundary_id": "right" if tag == "left" else "left",
                "diverge_side": tag,
                "asymmetry_score": abs(left_score - right_score),
                "tag_confidence": confidence,
                "left_score": left_score,
                "right_score": right_score,
                "tag_source": "ccs_rule_rebuild_geometry",
                "anchor_idx": int(anchor_idx),
                "anchor_x": float(anchor[0]),
                "anchor_y": float(anchor[1]),
                "distance_to_reference_boundary_m": reference_dist,
                "distance_to_ego_m": ego_dist,
            }
        )
        asset_rows.append(
            {
                "sample_token": token,
                "scene_name": sample["scene_name"],
                "scene_pos": scene_pos,
                "scene_len": scene_len,
                "is_temporal_eligible": flags["is_temporal_eligible"],
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
                "mapecho_loc_method": "ccs_rule_rebuild_anchor",
                "mapecho_tag_confidence": confidence,
                "mapecho_distance_to_reference_boundary_m": reference_dist,
            }
        )

    eligible_assets = [row for row in asset_rows if str(row["is_temporal_eligible"]).lower() == "true"]

    by_scene = {}
    for row in eligible_assets:
        by_scene.setdefault(row["scene_name"], []).append(row)
    selected = []
    for scene_name, rows in sorted(by_scene.items()):
        rows.sort(
            key=lambda row: (
                -float(row["mapecho_tag_confidence"]),
                abs(int(row["scene_pos"]) - (int(row["scene_len"]) - 1) / 2.0),
                row["sample_token"],
            )
        )
        selected.extend(rows[: args.max_per_scene])
    selected.sort(
        key=lambda row: (
            -float(row["mapecho_tag_confidence"]),
            row["scene_name"],
            int(row["scene_pos"]),
            row["sample_token"],
        )
    )
    if args.target_max > 0:
        selected = selected[: args.target_max]
    selected_tokens = {row["sample_token"] for row in selected}
    for row in asset_rows:
        if row["sample_token"] in selected_tokens:
            row["is_phase1_selected"] = True
    primary_by_scene = set()
    for row in selected:
        if row["scene_name"] not in primary_by_scene:
            row["is_primary_scene_sample"] = True
            primary_by_scene.add(row["scene_name"])
    for row in asset_rows:
        if row["sample_token"] in {r["sample_token"] for r in selected if r["is_primary_scene_sample"]}:
            row["is_primary_scene_sample"] = True

    write_csv(out_dir / "phase1_8b_temporal_metadata_W10_L9.csv", metadata_rows)
    write_csv(out_dir / "phase1_8b_boundary_tags.csv", tag_rows)
    write_csv(out_dir / "phase1_8b_all_assets.csv", asset_rows)
    write_csv(out_dir / "phase1_8b_temporal_eligible_assets.csv", eligible_assets)
    write_csv(out_dir / "phase1_8b_selected_assets.csv", selected)
    (out_dir / "phase1_8b_temporal_eligible_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in eligible_assets) + ("\n" if eligible_assets else "")
    )
    (out_dir / "phase1_8b_selected_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in selected) + ("\n" if selected else "")
    )
    scene_rows = []
    for scene_name, rows in sorted(by_scene.items()):
        scene_rows.append(
            {
                "scene_name": scene_name,
                "eligible_frames": len(rows),
                "selected_frames": sum(row["sample_token"] in selected_tokens for row in rows),
            }
        )
    write_csv(out_dir / "phase1_8b_scene_coverage.csv", scene_rows)

    summary = {
        "rule_dir": str(rule_dir),
        "tokens": str(token_path),
        "warmup": args.warmup,
        "recovery": args.recovery,
        "all_frames": len(asset_rows),
        "all_scenes": len({row["scene_name"] for row in asset_rows}),
        "temporal_eligible_frames": len(eligible_assets),
        "temporal_eligible_scenes": len(by_scene),
        "selected_frames": len(selected),
        "selected_scenes": len({row["scene_name"] for row in selected}),
        "outputs": {
            "all_assets": str(out_dir / "phase1_8b_all_assets.csv"),
            "temporal_eligible_assets": str(out_dir / "phase1_8b_temporal_eligible_assets.csv"),
            "selected_assets": str(out_dir / "phase1_8b_selected_assets.csv"),
            "selected_tokens": str(out_dir / "phase1_8b_selected_tokens.txt"),
        },
    }
    (out_dir / "phase1_8b_asset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
