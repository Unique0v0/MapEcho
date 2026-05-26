#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_match_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("seed_rank", "stream_global_idx", "sample_idx", "scene_pos", "scene_len"):
            if row.get(key) not in ("", None):
                row[key] = int(row[key])
        for key in ("matched_streammapnet", "present_in_ccs25_info", "temporal_eligible_W10_L19"):
            row[key] = row.get(key) == "True"
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def looks_like_attack_config(info):
    keys = {key.lower() for key in info.keys()}
    markers = ("attack", "rsa", "eta", "patch", "blind", "glare", "flash")
    return any(any(marker in key for marker in markers) for key in keys)


def main():
    parser = argparse.ArgumentParser(
        description="Build temporal eligible metadata and Phase 1 probe selection."
    )
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--ccs25-info", required=True)
    parser.add_argument("--match-csv", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--phase1-out-dir", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=19)
    parser.add_argument("--phase1-size", type=int, default=20)
    args = parser.parse_args()

    stream_samples = load_pickle(args.stream_ann)
    ccs25_obj = load_pickle(args.ccs25_info)
    ccs25_infos = ccs25_obj["infos"] if isinstance(ccs25_obj, dict) else ccs25_obj
    match_rows = load_match_rows(args.match_csv)

    sample_by_token = {sample["token"]: sample for sample in stream_samples}
    ccs25_by_token = {info["token"]: info for info in ccs25_infos}

    scene_samples = defaultdict(list)
    for sample in stream_samples:
        scene_samples[sample["scene_name"]].append(sample)
    for samples in scene_samples.values():
        samples.sort(key=lambda sample: sample["sample_idx"])

    metadata_rows = []
    for row in match_rows:
        if not row["matched_streammapnet"]:
            continue
        token = row["token"]
        scene_name = row["scene_name"]
        scene_pos = row["scene_pos"]
        scene_samples_sorted = scene_samples[scene_name]
        ccs25_info = ccs25_by_token.get(token, {})

        warmup_start = max(0, scene_pos - args.warmup)
        warmup_tokens = [
            sample["token"] for sample in scene_samples_sorted[warmup_start:scene_pos]
        ]
        recovery_end = min(len(scene_samples_sorted), scene_pos + args.recovery + 1)
        recovery_tokens = [
            sample["token"]
            for sample in scene_samples_sorted[scene_pos + 1:recovery_end]
        ]
        is_temporal_eligible = (
            len(warmup_tokens) == args.warmup
            and len(recovery_tokens) == args.recovery
        )
        post_recovery_margin = row["scene_len"] - 1 - (scene_pos + args.recovery)
        pre_warmup_margin = scene_pos - args.warmup
        center_margin = min(scene_pos, row["scene_len"] - 1 - scene_pos)
        eligibility_margin = min(pre_warmup_margin, post_recovery_margin)

        metadata_rows.append(
            {
                "sample_token": token,
                "scene_token": ccs25_info.get("scene_token", ""),
                "scene_name": scene_name,
                "scene_pos": scene_pos,
                "scene_len": row["scene_len"],
                "stream_global_idx": row["stream_global_idx"],
                "sample_idx": row["sample_idx"],
                "ccs25_seed_index": row["seed_rank"],
                "warmup_tokens": " ".join(warmup_tokens),
                "recovery_tokens": " ".join(recovery_tokens),
                "num_warmup": len(warmup_tokens),
                "num_recovery": len(recovery_tokens),
                "is_temporal_eligible": is_temporal_eligible,
                "pre_warmup_margin": pre_warmup_margin,
                "post_recovery_margin": post_recovery_margin,
                "eligibility_margin": eligibility_margin,
                "center_margin": center_margin,
                "has_attack_config": looks_like_attack_config(ccs25_info),
            }
        )

    metadata_fieldnames = [
        "sample_token",
        "scene_token",
        "scene_name",
        "scene_pos",
        "scene_len",
        "stream_global_idx",
        "sample_idx",
        "ccs25_seed_index",
        "warmup_tokens",
        "recovery_tokens",
        "num_warmup",
        "num_recovery",
        "is_temporal_eligible",
        "pre_warmup_margin",
        "post_recovery_margin",
        "eligibility_margin",
        "center_margin",
        "has_attack_config",
    ]
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(metadata_out, metadata_rows, metadata_fieldnames)

    eligible_rows = [
        row for row in metadata_rows if row["is_temporal_eligible"]
    ]
    by_scene = defaultdict(list)
    for row in eligible_rows:
        by_scene[row["scene_name"]].append(row)

    def primary_sort_key(row):
        return (
            -row["eligibility_margin"],
            -row["center_margin"],
            row["ccs25_seed_index"],
        )

    selected = []
    selected_tokens = set()
    for scene_name in sorted(by_scene):
        candidates = sorted(by_scene[scene_name], key=primary_sort_key)
        chosen = candidates[0]
        selected.append(
            {
                **chosen,
                "is_primary_scene_sample": True,
                "selection_reason": "primary_per_scene_max_margin",
            }
        )
        selected_tokens.add(chosen["sample_token"])

    remaining = [
        row for row in eligible_rows if row["sample_token"] not in selected_tokens
    ]
    remaining.sort(key=primary_sort_key)
    for row in remaining:
        if len(selected) >= args.phase1_size:
            break
        selected.append(
            {
                **row,
                "is_primary_scene_sample": False,
                "selection_reason": "extra_frame_to_reach_phase1_size",
            }
        )

    phase1_out_dir = Path(args.phase1_out_dir)
    phase1_out_dir.mkdir(parents=True, exist_ok=True)
    phase1_fieldnames = metadata_fieldnames + [
        "is_primary_scene_sample",
        "selection_reason",
    ]
    selection_csv = phase1_out_dir / "phase1_probe_selection.csv"
    write_csv(selection_csv, selected, phase1_fieldnames)
    (phase1_out_dir / "phase1_probe_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in selected) + ("\n" if selected else "")
    )

    summary = {
        "metadata_csv": str(metadata_out),
        "phase1_selection_csv": str(selection_csv),
        "warmup": args.warmup,
        "recovery": args.recovery,
        "all_seed_rows": len(metadata_rows),
        "temporal_eligible_frames": len(eligible_rows),
        "temporal_eligible_scenes": len(by_scene),
        "phase1_target_size": args.phase1_size,
        "phase1_selected_frames": len(selected),
        "phase1_selected_scenes": len({row["scene_name"] for row in selected}),
        "phase1_primary_frames": sum(row["is_primary_scene_sample"] for row in selected),
        "phase1_extra_frames": sum(not row["is_primary_scene_sample"] for row in selected),
        "has_attack_config_frames": sum(row["has_attack_config"] for row in metadata_rows),
    }
    (phase1_out_dir / "phase1_probe_selection_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
