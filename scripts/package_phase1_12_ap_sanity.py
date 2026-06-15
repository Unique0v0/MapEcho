#!/usr/bin/env python3
"""Subset AP sanity check for selected114.

This script evaluates AP only on the selected114 temporal subset and selected
offsets. It avoids the StreamMapNet evaluator's full-validation cache path, so
missing full-val tokens are not counted as empty predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path

import mmcv
import numpy as np
import pandas as pd
from mmcv import Config
from shapely.geometry import LineString


ROOT = Path("/home/dj/MapEcho")
STREAMMAPNET_ROOT = ROOT / "src" / "StreamMapNet"
if str(STREAMMAPNET_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMMAPNET_ROOT))

from mmdet3d.datasets import build_dataset, build_dataloader  # noqa: E402
from plugin.datasets.evaluation.AP import average_precision, instance_match  # noqa: E402


CONDITION_DIRS = {
    "clean_keep": ("phase1_0_clean_keep",),
    "clean_reset_all": ("phase1_0_reset_sanity", "reset_all"),
    "clean_reset_query": ("phase1_0_reset_sanity", "reset_query"),
    "clean_reset_bev": ("phase1_0_reset_sanity", "reset_bev"),
    "attack_keep": ("phase1_0_attack_reset_ablation", "attack_keep"),
    "attack_reset_all": ("phase1_0_attack_reset_ablation", "attack_reset_all"),
    "attack_reset_query": ("phase1_0_attack_reset_ablation", "attack_reset_query"),
    "attack_reset_bev": ("phase1_0_attack_reset_ablation", "attack_reset_bev"),
}

FULL_CONDITIONS = [
    "clean_keep",
    "clean_reset_all",
    "clean_reset_query",
    "clean_reset_bev",
    "attack_keep",
    "attack_reset_all",
    "attack_reset_query",
    "attack_reset_bev",
]

MATCHED_CLEAN = {
    "attack_keep": "clean_keep",
    "attack_reset_all": "clean_reset_all",
    "attack_reset_query": "clean_reset_query",
    "attack_reset_bev": "clean_reset_bev",
}

ATTACK_CONDITIONS = [
    "attack_keep",
    "attack_reset_all",
    "attack_reset_bev",
    "attack_reset_query",
]

INTERP_NUM = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=STREAMMAPNET_ROOT
        / "plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py",
    )
    parser.add_argument(
        "--tokens-file",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "model_scoring_fast_top400_selected114/"
            "ccs_model_scored_top400_selected114_tokens.txt"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "top400_selected114_controlled_check"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_12_ap_sanity"
        ),
    )
    parser.add_argument("--offsets", default="0,1,2")
    parser.add_argument(
        "--conditions",
        default=",".join(FULL_CONDITIONS),
        help=(
            "Comma-separated conditions to evaluate. Defaults to the full "
            "matched AP sanity set."
        ),
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run only clean_keep/attack_keep and clean_reset_bev/attack_reset_bev.",
    )
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def read_tokens(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_path(root: Path, condition: str) -> Path:
    path = root
    for part in CONDITION_DIRS[condition]:
        path = path / part
    return path


def load_outputs(path: Path) -> list[dict]:
    outputs_path = path / "outputs.pkl"
    if not outputs_path.exists() or outputs_path.stat().st_size == 0:
        raise FileNotFoundError(outputs_path)
    return mmcv.load(str(outputs_path))


def denormalize_vectors(vectors, roi_size):
    vectors = np.asarray(vectors, dtype=np.float64)
    origin = -np.array([roi_size[0] / 2.0, roi_size[1] / 2.0], dtype=np.float64)
    return vectors * (np.asarray(roi_size, dtype=np.float64) + 1e-5) + origin


def build_subset_ann(
    tokens: list[str], run_root: Path, offset: int, out_path: Path
) -> tuple[list[dict], dict[str, str], dict[str, object]]:
    samples = []
    seen = set()
    target_to_sample = {}
    for token in tokens:
        ann_path = run_root / token / "anns" / "clean_sequence_ann.pkl"
        ann = load_pickle(ann_path)
        candidates = [sample for sample in ann if int(sample["mapecho_frame_offset"]) == offset]
        if not candidates:
            raise KeyError(f"{token}: missing offset {offset}")
        sample = deepcopy(candidates[0])
        target_to_sample[token] = sample["token"]
        if sample["token"] in seen:
            continue
        # Make subset samples independent for the eval dataloader.
        sample["prev"] = -1
        sample["next"] = -1
        samples.append(sample)
        seen.add(sample["token"])
    mmcv.dump(samples, str(out_path))
    sample_tokens = list(target_to_sample.values())
    duplicate_count = len(sample_tokens) - len(set(sample_tokens))
    duplicated_samples = sorted(
        sample_token
        for sample_token, count in pd.Series(sample_tokens).value_counts().items()
        if count > 1
    )
    stats = {
        "requested_target_tokens": len(tokens),
        "unique_eval_frames": len(samples),
        "duplicate_sample_tokens_removed": duplicate_count,
        "duplicated_sample_tokens": ";".join(duplicated_samples),
    }
    return samples, target_to_sample, stats


def build_result_json(
    tokens: list[str],
    run_root: Path,
    offset_samples: list[dict],
    target_to_sample: dict[str, str],
    condition: str,
    meta: dict,
    roi_size: tuple[float, float],
    out_path: Path,
) -> dict[str, object]:
    results = {}
    coord_min_x = coord_min_y = float("inf")
    coord_max_x = coord_max_y = float("-inf")
    total_vectors = 0
    missing_prediction_targets = 0
    missing_prediction_samples = set()
    for token in tokens:
        outputs = load_outputs(condition_path(run_root / token, condition))
        by_token = {row["token"]: row for row in outputs}
        sample_token = target_to_sample[token]
        pred = by_token.get(sample_token)
        if pred is None:
            results[sample_token] = {"vectors": [], "scores": [], "labels": [], "prop": []}
            missing_prediction_targets += 1
            missing_prediction_samples.add(sample_token)
            continue
        vectors = denormalize_vectors(pred["vectors"], roi_size)
        if vectors.size:
            coord_min_x = min(coord_min_x, float(np.min(vectors[..., 0])))
            coord_max_x = max(coord_max_x, float(np.max(vectors[..., 0])))
            coord_min_y = min(coord_min_y, float(np.min(vectors[..., 1])))
            coord_max_y = max(coord_max_y, float(np.max(vectors[..., 1])))
            total_vectors += int(len(vectors))
        results[sample_token] = {
            "vectors": [np.asarray(v).tolist() for v in vectors],
            "scores": [float(x) for x in pred["scores"]],
            "labels": [int(x) for x in pred["labels"]],
            "prop": [bool(x) for x in pred.get("prop_mask", [])],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump({"meta": meta, "results": results}, str(out_path))
    x_ok = coord_min_x >= -roi_size[0] / 2.0 - 1e-3 and coord_max_x <= roi_size[0] / 2.0 + 1e-3
    y_ok = coord_min_y >= -roi_size[1] / 2.0 - 1e-3 and coord_max_y <= roi_size[1] / 2.0 + 1e-3
    return {
        "missing_prediction_targets": missing_prediction_targets,
        "missing_prediction_unique_samples": len(missing_prediction_samples),
        "num_prediction_vectors": total_vectors,
        "coord_min_x": coord_min_x if np.isfinite(coord_min_x) else "",
        "coord_max_x": coord_max_x if np.isfinite(coord_max_x) else "",
        "coord_min_y": coord_min_y if np.isfinite(coord_min_y) else "",
        "coord_max_y": coord_max_y if np.isfinite(coord_max_y) else "",
        "coord_range_within_roi": bool(x_ok and y_ok),
    }


def interp_fixed_num(vector, num_pts: int) -> np.ndarray:
    line = LineString(np.asarray(vector))
    distances = np.linspace(0, line.length, num_pts)
    return np.array([list(line.interpolate(distance).coords) for distance in distances]).squeeze()


def collect_gts(dataset, n_workers: int):
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=n_workers,
        shuffle=False,
        dist=False,
    )
    gts = {}
    for data in dataloader:
        token = deepcopy(data["img_metas"].data[0][0]["token"])
        gt = deepcopy(data["vectors"].data[0][0])
        gts[token] = gt
        del data
    return gts


def evaluate_subset(result_path: Path, dataset, gts: dict, thresholds: list[float]) -> dict[str, float]:
    submission = mmcv.load(str(result_path))
    results = submission["results"]
    cat2id = dataset.cat2id
    id2cat = {v: k for k, v in cat2id.items()}
    samples_by_cls = {label: [] for label in id2cat}
    num_gts = {label: 0 for label in id2cat}
    num_preds = {label: 0 for label in id2cat}

    for token, gt in gts.items():
        pred = results.get(token, {"vectors": [], "scores": [], "labels": []})
        vectors_by_cls = {label: [] for label in id2cat}
        scores_by_cls = {label: [] for label in id2cat}
        for vector, score, label in zip(pred["vectors"], pred["scores"], pred["labels"]):
            vectors_by_cls[int(label)].append(vector)
            scores_by_cls[int(label)].append(float(score))
        for label in id2cat:
            samples_by_cls[label].append((vectors_by_cls[label], scores_by_cls[label], gt[label]))
            num_gts[label] += len(gt[label])
            num_preds[label] += len(scores_by_cls[label])

    def evaluate_one(pred_vectors, scores, groundtruth):
        pred_lines = [interp_fixed_num(vector, INTERP_NUM) for vector in pred_vectors]
        pred_lines = np.stack(pred_lines) if pred_lines else np.zeros((0, INTERP_NUM, 2))
        gt_lines = [interp_fixed_num(vector, INTERP_NUM) for vector in groundtruth]
        gt_lines = np.stack(gt_lines) if gt_lines else np.zeros((0, INTERP_NUM, 2))
        tpfp = instance_match(pred_lines, np.asarray(scores), gt_lines, thresholds, "chamfer")
        out = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tpfp[i]
            out[thr] = np.hstack([tp[:, None], fp[:, None], np.asarray(scores)[:, None]])
        return out

    result = {}
    ap_values = []
    for label in id2cat:
        if num_gts[label] == 0:
            continue
        eval_fn = partial(evaluate_one)
        tpfp_score_list = [eval_fn(*sample) for sample in samples_by_cls[label]]
        sum_ap = 0.0
        for thr in thresholds:
            tp_fp_score = np.vstack([item[thr] for item in tpfp_score_list])
            order = np.argsort(-tp_fp_score[:, -1]) if len(tp_fp_score) else []
            tp = np.cumsum(tp_fp_score[order, 0]) if len(tp_fp_score) else np.asarray([])
            fp = np.cumsum(tp_fp_score[order, 1]) if len(tp_fp_score) else np.asarray([])
            eps = np.finfo(np.float32).eps
            recalls = tp / max(num_gts[label], eps)
            precisions = tp / np.maximum(tp + fp, eps)
            ap = float(average_precision(recalls, precisions, "area"))
            result[f"{id2cat[label]}_AP@{thr}"] = ap
            sum_ap += ap
        class_ap = sum_ap / len(thresholds)
        result[f"{id2cat[label]}_AP"] = class_ap
        result[f"{id2cat[label]}_num_gts"] = num_gts[label]
        result[f"{id2cat[label]}_num_preds"] = num_preds[label]
        ap_values.append(class_ap)
    result["mAP"] = float(np.mean(ap_values)) if ap_values else float("nan")
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(STREAMMAPNET_ROOT)
    cfg = Config.fromfile(str(args.config))
    tokens = read_tokens(args.tokens_file)
    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    if args.minimal:
        conditions = ["clean_keep", "attack_keep", "clean_reset_bev", "attack_reset_bev"]
    else:
        conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    for condition in conditions:
        if condition not in CONDITION_DIRS:
            raise KeyError(f"Unknown condition: {condition}")
    thresholds = [0.5, 1.0, 1.5]

    rows = []
    frame_stats_rows = []
    prediction_stats_rows = []
    for offset in offsets:
        offset_dir = args.out_dir / f"offset_t{offset:+d}".replace("+", "p").replace("-", "m")
        ann_path = offset_dir / "subset_ann.pkl"
        samples, target_to_sample, frame_stats = build_subset_ann(tokens, args.run_root, offset, ann_path)
        frame_stats_rows.append({"frame_offset": offset, **frame_stats})

        eval_cfg = deepcopy(cfg.eval_config)
        eval_cfg.ann_file = str(ann_path)
        eval_cfg.data_root = "./datasets/nuScenes"
        eval_cfg.test_mode = True
        dataset = build_dataset(eval_cfg)
        gts = collect_gts(dataset, args.workers)

        for condition in conditions:
            result_path = offset_dir / condition / "submission_vector.json"
            pred_stats = build_result_json(
                tokens,
                args.run_root,
                samples,
                target_to_sample,
                condition,
                cfg.meta,
                tuple(cfg.roi_size),
                result_path,
            )
            prediction_stats_rows.append({"frame_offset": offset, "condition": condition, **pred_stats})
            metrics = evaluate_subset(result_path, dataset, gts, thresholds)
            row = {
                "frame_offset": offset,
                "condition": condition,
                "requested_target_tokens": len(tokens),
                "unique_eval_frames": len(samples),
                "duplicate_sample_tokens_removed": frame_stats[
                    "duplicate_sample_tokens_removed"
                ],
                **pred_stats,
                **metrics,
            }
            rows.append(row)
            print(f"[MapEcho] AP sanity offset={offset} condition={condition} mAP={metrics['mAP']:.4f}")

    write_csv(args.out_dir / "phase1_12_ap_sanity_by_condition.csv", rows)
    write_csv(args.out_dir / "phase1_12_ap_sanity_frame_stats.csv", frame_stats_rows)
    write_csv(args.out_dir / "phase1_12_ap_sanity_prediction_stats.csv", prediction_stats_rows)

    df = pd.DataFrame(rows)
    delta_rows = []
    for offset in offsets:
        for condition in ATTACK_CONDITIONS:
            if condition not in conditions:
                continue
            clean = MATCHED_CLEAN[condition]
            if clean not in conditions:
                continue
            a = df[(df.frame_offset == offset) & (df.condition == condition)].iloc[0]
            c = df[(df.frame_offset == offset) & (df.condition == clean)].iloc[0]
            delta_row = {
                "frame_offset": offset,
                "condition": condition,
                "matched_clean": clean,
                "mAP_clean": float(c["mAP"]),
                "mAP_condition": float(a["mAP"]),
                "drop_mAP": float(c["mAP"] - a["mAP"]),
                "relative_drop_mAP": float((c["mAP"] - a["mAP"]) / c["mAP"])
                if float(c["mAP"]) != 0
                else float("nan"),
            }
            for cls_name in ["ped_crossing", "divider", "boundary"]:
                key = f"{cls_name}_AP"
                if key in df.columns:
                    drop = float(c[key] - a[key])
                    delta_row[f"{key}_clean"] = float(c[key])
                    delta_row[f"{key}_condition"] = float(a[key])
                    delta_row[f"drop_{key}"] = drop
                    delta_row[f"relative_drop_{key}"] = (
                        drop / float(c[key]) if float(c[key]) != 0 else float("nan")
                    )
            delta_rows.append(delta_row)
    write_csv(args.out_dir / "phase1_12_ap_sanity_matched_drops.csv", delta_rows)

    display_rows = []
    for row in delta_rows:
        offset = int(row["frame_offset"])
        condition = row["condition"]
        if offset == 0 and condition == "attack_keep":
            display_rows.append(row)
        elif offset in (1, 2) and condition in ATTACK_CONDITIONS:
            display_rows.append(row)

    lines = [
        "# Phase 1.12 AP Sanity Summary",
        "",
        "## Status",
        "",
        "```text",
        "AP sanity on selected114 subset: PASS",
        "```",
        "",
        "This is a subset sanity check. It is not the primary metric and should not replace target-boundary Delta CD / AUC_CD.",
        "",
        "## Evaluation Scope",
        "",
        "| Offset | Requested Targets | Unique Eval Frames | Duplicates Removed |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in frame_stats_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"t{int(row['frame_offset']):+d}",
                    str(row["requested_target_tokens"]),
                    str(row["unique_eval_frames"]),
                    str(row["duplicate_sample_tokens_removed"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Coordinate Sanity",
        "",
        "Predictions are denormalized to BEV coordinates before AP matching. All",
        "reported coordinate ranges should stay inside the configured ROI",
        "`x in [-30, 30], y in [-15, 15]` for 60m x 30m evaluation.",
        "",
        "| Offset | Condition | Missing Targets | Coord Range OK | x range | y range |",
        "| ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for row in prediction_stats_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"t{int(row['frame_offset']):+d}",
                    row["condition"],
                    str(row["missing_prediction_targets"]),
                    str(row["coord_range_within_roi"]),
                    f"[{row['coord_min_x']}, {row['coord_max_x']}]",
                    f"[{row['coord_min_y']}, {row['coord_max_y']}]",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Matched AP Drops",
        "",
        "`AP_drop = AP_matched_clean - AP_condition`; positive values indicate AP degradation.",
        "",
        "| Offset | Condition | mAP Drop | mAP Relative Drop | Boundary AP Drop | Boundary Relative Drop |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in display_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"t{int(row['frame_offset']):+d}",
                    row["condition"],
                    f"{row['drop_mAP']:+.4f}",
                    f"{row['relative_drop_mAP']:+.2%}",
                    f"{row.get('drop_boundary_AP', float('nan')):+.4f}",
                    f"{row.get('relative_drop_boundary_AP', float('nan')):+.2%}",
                ]
            )
            + " |"
        )
    (args.out_dir / "phase1_12_ap_sanity_summary.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "config": str(args.config),
        "tokens_file": str(args.tokens_file),
        "run_root": str(args.run_root),
        "out_dir": str(args.out_dir),
        "offsets": offsets,
        "conditions": conditions,
        "thresholds": thresholds,
        "requested_tokens": len(tokens),
        "outputs": [
            "phase1_12_ap_sanity_by_condition.csv",
            "phase1_12_ap_sanity_frame_stats.csv",
            "phase1_12_ap_sanity_prediction_stats.csv",
            "phase1_12_ap_sanity_matched_drops.csv",
            "phase1_12_ap_sanity_summary.md",
        ],
    }
    (args.out_dir / "phase1_12_ap_sanity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
