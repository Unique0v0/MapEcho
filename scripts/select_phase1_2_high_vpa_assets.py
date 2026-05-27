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


def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def enrich_candidates(assets, tags, vpa_rows, min_vpa):
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
        if as_bool(reference["vpa_pass"]):
            continue
        diverge_vpa = as_float(diverge["vpa_point_coverage"])
        if diverge_vpa < min_vpa:
            continue
        tag = tags[token]
        scene_pos = int(asset["scene_pos"])
        scene_len = int(asset["scene_len"])
        centrality = 1.0 - abs(scene_pos - (scene_len - 1) / 2.0) / max(scene_len / 2.0, 1.0)
        score = 3.0 * diverge_vpa + 0.5 * as_float(tag["tag_confidence"]) + 0.2 * centrality
        candidates.append(
            {
                **asset,
                "is_vpa_pass": True,
                "diverge_vpa_coverage": diverge_vpa,
                "reference_vpa_coverage": as_float(reference["vpa_point_coverage"]),
                "visible_cam": diverge["visible_cam"],
                "tag_confidence": as_float(tag["tag_confidence"]),
                "asymmetry_score": as_float(tag["asymmetry_score"]),
                "centrality_score": centrality,
                "selection_score": score,
                "selection_reason": "",
            }
        )
    return candidates


def select_balanced(candidates, max_frames):
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
        if len(selected) >= max_frames:
            break
        row["selection_reason"] = "score_fill"
        row["is_primary_scene_sample"] = False
        row["is_phase1_selected"] = True
        selected.append(row)
        selected_tokens.add(row["sample_token"])

    selected.sort(key=lambda row: (row["scene_name"], int(row["scene_pos"])))
    return selected


def summarize(rows):
    return {
        "frames": len(rows),
        "scenes": len({row["scene_name"] for row in rows}),
        "primary_scene_samples": sum(row.get("selection_reason") == "primary_scene_sample" for row in rows),
        "median_diverge_vpa_coverage": median([as_float(row["diverge_vpa_coverage"]) for row in rows]),
        "min_diverge_vpa_coverage": min([as_float(row["diverge_vpa_coverage"]) for row in rows]) if rows else None,
        "max_diverge_vpa_coverage": max([as_float(row["diverge_vpa_coverage"]) for row in rows]) if rows else None,
        "median_tag_confidence": median([as_float(row["tag_confidence"]) for row in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description="Select Phase 1.2 high-VPA assets from the ccs_candidate expansion pool.")
    parser.add_argument("--asset-csv", default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv")
    parser.add_argument("--tag-csv", default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/asymmetric_dist_boundary_tags.csv")
    parser.add_argument("--vpa-csv", default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa")
    parser.add_argument("--high-vpa", type=float, default=0.25)
    parser.add_argument("--medium-vpa", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()

    assets = {row["sample_token"]: row for row in read_csv(args.asset_csv)}
    tags = {row["sample_token"]: row for row in read_csv(args.tag_csv)}
    vpa_rows = read_csv(args.vpa_csv)

    high_candidates = enrich_candidates(assets, tags, vpa_rows, args.high_vpa)
    medium_plus_candidates = enrich_candidates(assets, tags, vpa_rows, args.medium_vpa)
    selected = select_balanced(high_candidates, args.max_frames)

    out_dir = Path(args.out_dir)
    selected_csv = out_dir / "phase1_2_high_vpa_assets.csv"
    token_txt = out_dir / "phase1_2_high_vpa_tokens.txt"
    reserve_csv = out_dir / "phase1_2_medium_vpa_reserve_assets.csv"
    fieldnames = list(selected[0].keys()) if selected else list(medium_plus_candidates[0].keys()) if medium_plus_candidates else []
    if fieldnames:
        write_csv(selected_csv, selected, fieldnames)
        reserve = [row for row in medium_plus_candidates if row["sample_token"] not in {item["sample_token"] for item in selected}]
        write_csv(reserve_csv, reserve, fieldnames)
    token_txt.write_text("\n".join(row["sample_token"] for row in selected) + ("\n" if selected else ""))

    summary = {
        "source_assets": args.asset_csv,
        "vpa_csv": args.vpa_csv,
        "high_vpa_threshold": args.high_vpa,
        "medium_vpa_threshold": args.medium_vpa,
        "candidate_frames_high_vpa": len(high_candidates),
        "candidate_scenes_high_vpa": len({row["scene_name"] for row in high_candidates}),
        "candidate_frames_medium_plus": len(medium_plus_candidates),
        "candidate_scenes_medium_plus": len({row["scene_name"] for row in medium_plus_candidates}),
        "selected": summarize(selected),
        "medium_reserve": summarize([row for row in medium_plus_candidates if row["sample_token"] not in {item["sample_token"] for item in selected}]),
        "selected_csv": str(selected_csv),
        "tokens_txt": str(token_txt),
        "reserve_csv": str(reserve_csv),
    }
    (out_dir / "phase1_2_high_vpa_selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
