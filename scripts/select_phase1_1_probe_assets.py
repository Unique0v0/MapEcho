#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value):
    return str(value).lower() == "true"


def as_float(value, default=0.0):
    if value in ("", None):
        return default
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Select Phase 1.1 probe assets after VPA sanity.")
    parser.add_argument("--asset-csv", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_asymmetric_dist_eta_like_assets.csv")
    parser.add_argument("--tag-csv", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/asymmetric_dist_boundary_tags.csv")
    parser.add_argument("--vpa-csv", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/vpa_sanity/eta_target_boundary_vpa_sanity.csv")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--min-vpa", type=float, default=0.05)
    args = parser.parse_args()

    assets = {row["sample_token"]: row for row in read_csv(args.asset_csv)}
    tags = {row["sample_token"]: row for row in read_csv(args.tag_csv)}
    vpa_rows = read_csv(args.vpa_csv)
    vpa_by_token = defaultdict(dict)
    for row in vpa_rows:
        vpa_by_token[row["sample_id"]][row["target_type"]] = row

    candidates = []
    for token, asset in assets.items():
        diverge = vpa_by_token[token].get("diverge_boundary")
        reference = vpa_by_token[token].get("reference_boundary")
        if not diverge or not reference:
            continue
        if not as_bool(diverge["vpa_pass"]):
            continue
        if as_float(diverge["vpa_point_coverage"]) < args.min_vpa:
            continue
        if as_bool(reference["vpa_pass"]):
            continue
        tag = tags[token]
        scene_pos = int(asset["scene_pos"])
        scene_len = int(asset["scene_len"])
        centrality = 1.0 - abs(scene_pos - (scene_len - 1) / 2.0) / max(scene_len / 2.0, 1.0)
        score = (
            2.0 * as_float(diverge["vpa_point_coverage"])
            + as_float(tag["tag_confidence"])
            + 0.2 * centrality
        )
        candidates.append(
            {
                **asset,
                "is_vpa_pass": True,
                "diverge_vpa_coverage": as_float(diverge["vpa_point_coverage"]),
                "reference_vpa_coverage": as_float(reference["vpa_point_coverage"]),
                "visible_cam": diverge["visible_cam"],
                "tag_confidence": as_float(tag["tag_confidence"]),
                "asymmetry_score": as_float(tag["asymmetry_score"]),
                "centrality_score": centrality,
                "selection_score": score,
                "selection_reason": "",
            }
        )

    by_scene = defaultdict(list)
    for row in candidates:
        by_scene[row["scene_name"]].append(row)
    for rows in by_scene.values():
        rows.sort(key=lambda row: row["selection_score"], reverse=True)

    selected = []
    selected_tokens = set()
    for scene_name in sorted(by_scene):
        row = by_scene[scene_name][0]
        row["selection_reason"] = "primary_scene_sample"
        row["is_primary_scene_sample"] = True
        row["is_phase1_selected"] = True
        selected.append(row)
        selected_tokens.add(row["sample_token"])

    remaining = [
        row
        for row in sorted(candidates, key=lambda item: item["selection_score"], reverse=True)
        if row["sample_token"] not in selected_tokens
    ]
    for row in remaining:
        if len(selected) >= args.max_frames:
            break
        row["selection_reason"] = "score_fill"
        row["is_primary_scene_sample"] = False
        row["is_phase1_selected"] = True
        selected.append(row)
        selected_tokens.add(row["sample_token"])

    selected.sort(key=lambda row: (row["scene_name"], int(row["scene_pos"])))
    out_dir = Path(args.out_dir)
    out_csv = out_dir / "phase1_1_probe_assets.csv"
    fieldnames = list(selected[0].keys()) if selected else []
    write_csv(out_csv, selected, fieldnames)
    (out_dir / "phase1_1_probe_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in selected) + ("\n" if selected else "")
    )

    summary = {
        "source_assets": args.asset_csv,
        "vpa_csv": args.vpa_csv,
        "candidate_frames_after_vpa": len(candidates),
        "candidate_scenes_after_vpa": len(by_scene),
        "selected_frames": len(selected),
        "selected_scenes": len({row["scene_name"] for row in selected}),
        "primary_scene_samples": sum(str(row["selection_reason"]) == "primary_scene_sample" for row in selected),
        "median_diverge_vpa_coverage": sorted(row["diverge_vpa_coverage"] for row in selected)[len(selected) // 2] if selected else None,
        "out_csv": str(out_csv),
        "tokens_txt": str(out_dir / "phase1_1_probe_tokens.txt"),
    }
    (out_dir / "phase1_1_probe_selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
