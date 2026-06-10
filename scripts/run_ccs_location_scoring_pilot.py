#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapecho_matplotlib")

from summarize_phase1_0_map_level import (
    best_boundary_metrics,
    extract_scene_boundaries,
    finite,
    global_polyline_to_ego_xy,
    invert_rigid,
    lidar_polyline_to_global,
    load_outputs_records,
    resample_polyline,
    transform_matrix,
)


ROOT = Path("/home/dj/MapEcho")
STREAMMAPNET_ROOT = ROOT / "src" / "StreamMapNet"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_cmd(cmd, cwd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("[MapEcho]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=str(cwd), env=merged_env, check=True)


def stream_env():
    env = {
        "MPLCONFIGDIR": "/tmp/mapecho_matplotlib",
        "PYTHONPATH": (
            "/home/dj/physical-online-map-attack:"
            "/home/dj/MapEcho/src/StreamMapNet:"
            "/home/dj/MapEcho:"
            f"{os.environ.get('PYTHONPATH', '')}"
        ),
        "LD_LIBRARY_PATH": (
            "/home/dj/.conda/envs/maptr4090/lib:"
            f"{os.environ.get('LD_LIBRARY_PATH', '')}"
        ),
    }
    return env


def build_clean_sequence(args, token, out_dir):
    token_file = out_dir / "target_token.txt"
    token_file.write_text(f"{token}\n")
    clean_ann = out_dir / "anns" / "clean_sequence_ann.pkl"
    if clean_ann.exists() and args.skip_completed:
        return clean_ann
    run_cmd(
        [
            args.python_maptr,
            ROOT / "scripts" / "build_sequence_ann_subset.py",
            "--stream-ann",
            args.stream_ann,
            "--tokens",
            token_file,
            "--target-token",
            token,
            "--out",
            clean_ann,
            "--summary-out",
            out_dir / "anns" / "clean_sequence_ann_summary.json",
            "--warmup",
            args.warmup,
            "--recovery",
            args.recovery,
        ],
        cwd=ROOT,
    )
    return clean_ann


def run_condition(args, ann_file, out_dir, condition):
    outputs = out_dir / "outputs.pkl"
    if outputs.exists() and args.skip_completed:
        return
    run_cmd(
        [
            args.python_stream,
            ROOT / "scripts" / "run_streammapnet_sequence_condition.py",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--ann-file",
            ann_file,
            "--out-dir",
            out_dir,
            "--condition",
            condition,
            "--reset-mode",
            "none",
        ],
        cwd=STREAMMAPNET_ROOT,
        env=stream_env(),
    )


def build_candidate_ann(args, clean_ann, candidate, out_dir):
    rank = int(candidate["rank"])
    ann_file = out_dir / "anns" / f"candidate_rank_{rank:03d}_sequence_ann.pkl"
    if ann_file.exists() and args.skip_completed:
        return ann_file
    run_cmd(
        [
            args.python_maptr,
            ROOT / "scripts" / "build_attack_at_t_sequence_ann.py",
            "--clean-ann",
            clean_ann,
            "--out-ann",
            ann_file,
            "--out-dir",
            out_dir / "rendered" / f"rank_{rank:03d}",
            "--attack-objective",
            "eta",
            "--source-frame",
            "lidar",
            "--power",
            args.power,
            "--renderer",
            "ccs",
            "--camera-mode",
            "all",
            "--loc-x",
            candidate["x"],
            "--loc-y",
            candidate["y"],
            "--loc-z",
            candidate["z"],
            "--loc-method",
            "ccs_dense_model_scoring_candidate",
        ],
        cwd=ROOT,
    )
    return ann_file


def score_candidate(clean_ann, scene_json, clean_dir, candidate_dir, args):
    ann = load_pickle(clean_ann)
    target_token = ann[0]["mapecho_target_token"]
    target_sample = next(sample for sample in ann if sample["token"] == target_token)
    current_sample = target_sample
    roi_size = [float(item.strip()) for item in args.roi_size.split(",")]

    _, diverge_lidar, reference_lidar = extract_scene_boundaries(scene_json)
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
    current_global2ego = invert_rigid(
        transform_matrix(current_sample["e2g_rotation"], current_sample["e2g_translation"])
    )
    diverge_xy = resample_polyline(
        global_polyline_to_ego_xy(diverge_global, current_global2ego),
        args.sample_interval_m,
    )
    reference_xy = resample_polyline(
        global_polyline_to_ego_xy(reference_global, current_global2ego),
        args.sample_interval_m,
    )

    clean_records = load_outputs_records(clean_dir, roi_size)
    candidate_records = load_outputs_records(candidate_dir, roi_size)
    clean_metrics = best_boundary_metrics(
        clean_records[target_token],
        diverge_xy,
        reference_xy,
        args.score_thr,
        args.sample_interval_m,
    )
    candidate_metrics = best_boundary_metrics(
        candidate_records[target_token],
        diverge_xy,
        reference_xy,
        args.score_thr,
        args.sample_interval_m,
    )

    def diff(field):
        return float(candidate_metrics[field]) - float(clean_metrics[field])

    return {
        "clean_cd_to_diverge_m": finite(clean_metrics["cd_to_diverge_m"]),
        "candidate_cd_to_diverge_m": finite(candidate_metrics["cd_to_diverge_m"]),
        "delta_cd_to_diverge_m": diff("cd_to_diverge_m"),
        "clean_cd_to_reference_m": finite(clean_metrics["cd_to_reference_m"]),
        "candidate_cd_to_reference_m": finite(candidate_metrics["cd_to_reference_m"]),
        "delta_cd_to_reference_m": diff("cd_to_reference_m"),
        "clean_wrong_reference_preference_m": finite(clean_metrics["wrong_reference_preference_m"]),
        "candidate_wrong_reference_preference_m": finite(candidate_metrics["wrong_reference_preference_m"]),
        "delta_wrong_reference_preference_m": diff("wrong_reference_preference_m"),
        "candidate_num_boundary_preds": candidate_metrics["num_boundary_preds"],
        "candidate_best_idx": candidate_metrics["best_idx"],
        "candidate_best_score": finite(candidate_metrics["best_score"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pilot CCS-style glare-source model scoring for StreamMapNet. "
            "Each dense candidate is rendered on all six target-frame cameras, "
            "run through StreamMapNet, and ranked by frame-t target-boundary CD delta."
        )
    )
    parser.add_argument("--target-token", required=True)
    parser.add_argument("--dense-candidates-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--stream-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl")
    parser.add_argument("--config", default="/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py")
    parser.add_argument("--checkpoint", default="/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth")
    parser.add_argument("--python-maptr", default="/home/dj/.conda/envs/maptr/bin/python")
    parser.add_argument("--python-stream", default="/home/dj/.conda/envs/maptr4090/bin/python")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=0)
    parser.add_argument("--power", type=float, default=3000.0)
    parser.add_argument("--roi-size", default="60,30")
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--boundary-z", type=float, default=-1.84)
    parser.add_argument("--sample-interval-m", type=float, default=0.25)
    parser.add_argument("--skip-completed", action="store_true", default=False)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    token_root = out_root / args.target_token
    token_root.mkdir(parents=True, exist_ok=True)

    all_candidates = [
        row for row in read_csv(args.dense_candidates_csv)
        if row["sample_token"] == args.target_token
    ]
    all_candidates.sort(key=lambda row: int(row["rank"]))
    candidates = all_candidates[: args.max_candidates]
    if not candidates:
        raise ValueError(f"No dense candidates found for {args.target_token}")

    clean_ann = build_clean_sequence(args, args.target_token, token_root)
    clean_dir = token_root / "clean_keep"
    run_condition(args, clean_ann, clean_dir, "clean_keep")

    score_rows = []
    for candidate in candidates:
        rank = int(candidate["rank"])
        candidate_root = token_root / "candidates" / f"rank_{rank:03d}"
        candidate_ann = build_candidate_ann(args, clean_ann, candidate, candidate_root)
        condition_dir = candidate_root / "streammapnet_frame_t"
        run_condition(args, candidate_ann, condition_dir, f"candidate_rank_{rank:03d}")
        metrics = score_candidate(
            clean_ann,
            candidate["scene_json"],
            clean_dir,
            condition_dir,
            args,
        )
        row = dict(candidate)
        row.update(metrics)
        score_rows.append(row)
        write_csv(token_root / "candidate_model_scores.csv", score_rows)

    score_rows.sort(
        key=lambda row: (
            float(row["delta_cd_to_diverge_m"]),
            float(row["geometric_score"]),
        ),
        reverse=True,
    )
    best = score_rows[0]
    best_asset = {
        "sample_token": best["sample_token"],
        "scene_name": best["scene_name"],
        "scene_pos": best.get("scene_pos", ""),
        "has_blind_eta_loc": True,
        "blind_eta_x": best["x"],
        "blind_eta_y": best["y"],
        "blind_eta_z": best["z"],
        "scene_json": best["scene_json"],
        "mapecho_loc_method": "ccs_dense_streammapnet_model_scored",
        "ccs_dense_rank": best["rank"],
        "ccs_dense_geometric_score": best["geometric_score"],
        "streammapnet_score_delta_cd_to_diverge_m": best["delta_cd_to_diverge_m"],
        "streammapnet_score_delta_wrong_reference_preference_m": best[
            "delta_wrong_reference_preference_m"
        ],
        "streammapnet_score_power": args.power,
        "streammapnet_score_num_candidates": len(score_rows),
    }
    write_csv(token_root / "ccs_model_scored_best_location_asset.csv", [best_asset])

    summary = {
        "target_token": args.target_token,
        "out_root": str(token_root),
        "dense_candidates_csv": args.dense_candidates_csv,
        "num_candidates_scored": len(score_rows),
        "power": args.power,
        "warmup": args.warmup,
        "recovery": args.recovery,
        "best_rank": best["rank"],
        "best_xyz_lidar": [best["x"], best["y"], best["z"]],
        "best_delta_cd_to_diverge_m": best["delta_cd_to_diverge_m"],
        "scores_csv": str(token_root / "candidate_model_scores.csv"),
        "best_asset_csv": str(token_root / "ccs_model_scored_best_location_asset.csv"),
        "note": (
            "Candidate generation and six-camera rendering follow the CCS-style migrated path. "
            "The final ranking is StreamMapNet-aware and uses frame-t target-boundary CD delta."
        ),
    }
    (token_root / "ccs_location_scoring_pilot_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
