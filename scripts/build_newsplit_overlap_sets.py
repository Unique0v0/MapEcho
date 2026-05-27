#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_ann_index(infos):
    by_token = {row["token"]: row for row in infos}
    by_scene = {}
    for row in infos:
        by_scene.setdefault(row["scene_name"], []).append(row)
    for rows in by_scene.values():
        rows.sort(key=lambda row: row["sample_idx"])
    return by_token, by_scene


def temporal_row(token, split_name, by_token, by_scene, warmup, recovery):
    sample = by_token[token]
    scene_rows = by_scene[sample["scene_name"]]
    scene_pos = next(i for i, row in enumerate(scene_rows) if row["token"] == token)
    return {
        "sample_token": token,
        "scene_name": sample["scene_name"],
        "newsplit_split": split_name,
        "scene_pos": scene_pos,
        "scene_len": len(scene_rows),
        "has_warmup": scene_pos >= warmup,
        "has_recovery": scene_pos + recovery < len(scene_rows),
        "is_temporal_eligible": scene_pos >= warmup
        and scene_pos + recovery < len(scene_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build old CCS/Phase1 overlap sets under StreamMapNet newsplit."
    )
    parser.add_argument("--newsplit-train-ann", required=True)
    parser.add_argument("--newsplit-val-ann", required=True)
    parser.add_argument("--ccs-match-csv", required=True)
    parser.add_argument("--old-eligible-tokens", required=True)
    parser.add_argument("--old-phase1-tokens", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=19)
    args = parser.parse_args()

    train_infos = load_pickle(args.newsplit_train_ann)
    val_infos = load_pickle(args.newsplit_val_ann)
    train_by_token, train_by_scene = build_ann_index(train_infos)
    val_by_token, val_by_scene = build_ann_index(val_infos)

    ccs_rows = load_rows(args.ccs_match_csv)
    ccs_tokens = [row["token"] for row in ccs_rows]
    old_eligible_tokens = load_tokens(args.old_eligible_tokens)
    old_phase1_tokens = load_tokens(args.old_phase1_tokens)

    def classify(tokens):
        rows = []
        for token in tokens:
            if token in val_by_token:
                rows.append(
                    temporal_row(
                        token,
                        "new_val",
                        val_by_token,
                        val_by_scene,
                        args.warmup,
                        args.recovery,
                    )
                )
            elif token in train_by_token:
                rows.append(
                    temporal_row(
                        token,
                        "new_train",
                        train_by_token,
                        train_by_scene,
                        args.warmup,
                        args.recovery,
                    )
                )
            else:
                rows.append(
                    {
                        "sample_token": token,
                        "scene_name": "",
                        "newsplit_split": "missing",
                        "scene_pos": "",
                        "scene_len": "",
                        "has_warmup": False,
                        "has_recovery": False,
                        "is_temporal_eligible": False,
                    }
                )
        return rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_token",
        "scene_name",
        "newsplit_split",
        "scene_pos",
        "scene_len",
        "has_warmup",
        "has_recovery",
        "is_temporal_eligible",
    ]
    groups = {
        "ccs100": classify(ccs_tokens),
        "old_temporal_eligible": classify(old_eligible_tokens),
        "old_phase1": classify(old_phase1_tokens),
    }
    for name, rows in groups.items():
        write_csv(out_dir / f"{name}_newsplit_membership.csv", rows, fieldnames)

    phase1_val = [
        row
        for row in groups["old_phase1"]
        if row["newsplit_split"] == "new_val" and row["is_temporal_eligible"]
    ]
    write_csv(out_dir / "phase1_0_overlap_selection.csv", phase1_val, fieldnames)
    (out_dir / "phase1_0_overlap_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in phase1_val)
        + ("\n" if phase1_val else "")
    )

    summary = {"warmup": args.warmup, "recovery": args.recovery}
    for name, rows in groups.items():
        summary[name] = {
            "total": len(rows),
            "new_train_frames": sum(row["newsplit_split"] == "new_train" for row in rows),
            "new_train_scenes": len(
                {row["scene_name"] for row in rows if row["newsplit_split"] == "new_train"}
            ),
            "new_val_frames": sum(row["newsplit_split"] == "new_val" for row in rows),
            "new_val_scenes": len(
                {row["scene_name"] for row in rows if row["newsplit_split"] == "new_val"}
            ),
            "missing_frames": sum(row["newsplit_split"] == "missing" for row in rows),
            "new_val_temporal_eligible_frames": sum(
                row["newsplit_split"] == "new_val" and row["is_temporal_eligible"]
                for row in rows
            ),
            "new_val_temporal_eligible_scenes": len(
                {
                    row["scene_name"]
                    for row in rows
                    if row["newsplit_split"] == "new_val"
                    and row["is_temporal_eligible"]
                }
            ),
        }
    summary["phase1_0_overlap_tokens"] = len(phase1_val)
    summary["phase1_0_overlap_scenes"] = len({row["scene_name"] for row in phase1_val})
    (out_dir / "newsplit_overlap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
