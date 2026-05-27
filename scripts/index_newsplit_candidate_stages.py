#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path


STAGES = {
    "ccs_candidate": ("dir", "scenes_candidate"),
    "ccs_asymmetric_dist": ("dir", "scenes_asymmetric_dist"),
    "ccs_asymmetric_curvature_selected": ("dir", "scenes_asymmetric_curvature_selected"),
    "ccs_final_asymmetric_json": ("dir", "scenes_asymmetric"),
    "ccs_final_asymmetric_100": ("txt", "sample_tokens_asymmetric.txt"),
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_split_index(infos):
    by_token = {row["token"]: row for row in infos}
    by_scene = {}
    for row in infos:
        by_scene.setdefault(row["scene_name"], []).append(row)
    for rows in by_scene.values():
        rows.sort(key=lambda row: row["sample_idx"])
    scene_pos = {}
    for scene_name, rows in by_scene.items():
        for i, row in enumerate(rows):
            scene_pos[row["token"]] = (i, len(rows))
    return by_token, scene_pos


def read_stage_tokens(dataset_root, stage_name, stage_spec):
    kind, rel = stage_spec
    path = dataset_root / rel
    if kind == "dir":
        tokens = sorted(p.stem for p in path.glob("*.json"))
        asset_path = lambda token: str(path / f"{token}.json")
    elif kind == "txt":
        tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        asset_path = lambda token: ""
    else:
        raise ValueError(kind)
    return tokens, asset_path


def temporal_flags(pos, scene_len, windows):
    values = {}
    for warmup, recovery in windows:
        key = f"W{warmup}_L{recovery}"
        values[f"eligible_{key}"] = pos >= warmup and pos + recovery < scene_len
    return values


def load_diverge_info(path):
    if not path:
        return "", "", "", "", ""
    p = Path(path)
    if not p.exists():
        return False, "", "", "", ""
    with p.open() as f:
        data = json.load(f)
    diverge = data.get("diverge_boundary_tag")
    if not diverge:
        return False, "", "", "", ""
    tag = diverge[0]
    confidence = diverge[1] if len(diverge) > 1 else ""
    left_score = diverge[2] if len(diverge) > 2 else ""
    right_score = diverge[3] if len(diverge) > 3 else ""
    return True, tag, confidence, left_score, right_score


def main():
    parser = argparse.ArgumentParser(
        description="Index CCS candidate stages under StreamMapNet newsplit val/train."
    )
    parser.add_argument("--newsplit-val-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl")
    parser.add_argument("--newsplit-train-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_train_newsplit.pkl")
    parser.add_argument("--ccs-dataset-root", default="/home/dj/physical-online-map-attack/dataset")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/newsplit_candidates")
    parser.add_argument("--windows", default="10:19,10:9,5:9")
    args = parser.parse_args()

    windows = []
    for item in args.windows.split(","):
        warmup, recovery = item.split(":")
        windows.append((int(warmup), int(recovery)))

    val_infos = load_pickle(args.newsplit_val_ann)
    train_infos = load_pickle(args.newsplit_train_ann)
    val_by_token, val_scene_pos = build_split_index(val_infos)
    train_by_token, train_scene_pos = build_split_index(train_infos)

    dataset_root = Path(args.ccs_dataset_root)
    out_dir = Path(args.out_dir)
    rows = []
    for stage_name, stage_spec in STAGES.items():
        tokens, asset_path_fn = read_stage_tokens(dataset_root, stage_name, stage_spec)
        for token in tokens:
            if token in val_by_token:
                split = "new_val"
                sample = val_by_token[token]
                pos, scene_len = val_scene_pos[token]
            elif token in train_by_token:
                split = "new_train"
                sample = train_by_token[token]
                pos, scene_len = train_scene_pos[token]
            else:
                split = "missing"
                sample = {}
                pos = scene_len = ""
            scene_json = asset_path_fn(token)
            has_diverge, diverge_tag, confidence, left_score, right_score = load_diverge_info(scene_json)
            row = {
                "stage": stage_name,
                "sample_token": token,
                "newsplit_split": split,
                "scene_name": sample.get("scene_name", ""),
                "location": sample.get("location", ""),
                "sample_idx": sample.get("sample_idx", ""),
                "scene_pos": pos,
                "scene_len": scene_len,
                "scene_json": scene_json,
                "has_diverge_tag": has_diverge,
                "diverge_boundary_tag": diverge_tag,
                "diverge_confidence": confidence,
                "left_score": left_score,
                "right_score": right_score,
            }
            if split != "missing":
                row.update(temporal_flags(pos, scene_len, windows))
            else:
                for warmup, recovery in windows:
                    row[f"eligible_W{warmup}_L{recovery}"] = False
            rows.append(row)

    fieldnames = [
        "stage",
        "sample_token",
        "newsplit_split",
        "scene_name",
        "location",
        "sample_idx",
        "scene_pos",
        "scene_len",
        "scene_json",
        "has_diverge_tag",
        "diverge_boundary_tag",
        "diverge_confidence",
        "left_score",
        "right_score",
    ] + [f"eligible_W{w}_L{l}" for w, l in windows]
    write_csv(out_dir / "ccs_stage_newsplit_membership.csv", rows, fieldnames)

    summary_rows = []
    for stage_name in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage_name]
        item = {
            "stage": stage_name,
            "total": len(stage_rows),
            "new_val_frames": sum(row["newsplit_split"] == "new_val" for row in stage_rows),
            "new_val_scenes": len({row["scene_name"] for row in stage_rows if row["newsplit_split"] == "new_val"}),
            "new_train_frames": sum(row["newsplit_split"] == "new_train" for row in stage_rows),
            "missing_frames": sum(row["newsplit_split"] == "missing" for row in stage_rows),
            "new_val_with_diverge_tag": sum(
                row["newsplit_split"] == "new_val" and str(row["has_diverge_tag"]).lower() == "true"
                for row in stage_rows
            ),
        }
        for warmup, recovery in windows:
            key = f"eligible_W{warmup}_L{recovery}"
            eligible = [row for row in stage_rows if row["newsplit_split"] == "new_val" and row[key]]
            item[f"{key}_frames"] = len(eligible)
            item[f"{key}_scenes"] = len({row["scene_name"] for row in eligible})
        summary_rows.append(item)

    summary_fields = list(summary_rows[0].keys())
    write_csv(out_dir / "ccs_stage_newsplit_summary.csv", summary_rows, summary_fields)
    summary = {
        "windows": [{"warmup": w, "recovery": l} for w, l in windows],
        "membership_csv": str(out_dir / "ccs_stage_newsplit_membership.csv"),
        "summary_csv": str(out_dir / "ccs_stage_newsplit_summary.csv"),
        "summary": summary_rows,
    }
    (out_dir / "ccs_stage_newsplit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
