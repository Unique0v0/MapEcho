#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


BLIND_RESULTS = {
    "rsa": "maptr-bevpool/train_blind_rsa_asymmetric/results/map/attack/best_attack_locs.json",
    "eta": "maptr-bevpool/train_blind_eta_asymmetric/results/map/attack/best_attack_locs.json",
}

PATCH_RESULTS = {
    "rsa": "maptr-bevpool/train_patch_rsa_asymmetric/results/map/attack/best_patches.pkl",
    "eta": "maptr-bevpool/train_patch_eta_asymmetric/results/map/attack/best_patches.pkl",
}


def read_tokens_from_match_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [row["token"] for row in rows], rows


def read_csv_by_token(path, token_field):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {row[token_field]: row for row in rows}


def read_best_locs(path):
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {token: [float(value) for value in loc] for token, loc in data.items()}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def loc_fields(prefix, locs, token):
    loc = locs.get(token)
    if loc is None:
        return {
            f"has_blind_{prefix}_loc": False,
            f"blind_{prefix}_x": "",
            f"blind_{prefix}_y": "",
            f"blind_{prefix}_z": "",
        }
    return {
        f"has_blind_{prefix}_loc": True,
        f"blind_{prefix}_x": loc[0],
        f"blind_{prefix}_y": loc[1],
        f"blind_{prefix}_z": loc[2],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Index reusable CCS'25 asymmetric attack assets for MapEcho."
    )
    parser.add_argument("--ccs25-root", required=True)
    parser.add_argument("--match-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temporal-metadata-csv")
    parser.add_argument("--phase1-selection-csv")
    args = parser.parse_args()

    ccs25_root = Path(args.ccs25_root).resolve()
    dataset_dir = ccs25_root / "dataset"
    out_dir = Path(args.out_dir)

    tokens, match_rows = read_tokens_from_match_csv(args.match_csv)
    match_by_token = {row["token"]: row for row in match_rows}
    temporal_by_token = (
        read_csv_by_token(args.temporal_metadata_csv, "sample_token")
        if args.temporal_metadata_csv
        else {}
    )
    phase1_by_token = (
        read_csv_by_token(args.phase1_selection_csv, "sample_token")
        if args.phase1_selection_csv
        else {}
    )

    blind_locs = {
        name: read_best_locs(dataset_dir / rel_path)
        for name, rel_path in BLIND_RESULTS.items()
    }

    patch_paths = {
        name: dataset_dir / rel_path for name, rel_path in PATCH_RESULTS.items()
    }

    rows = []
    for token in tokens:
        scene_json = dataset_dir / "scenes_asymmetric" / f"{token}.json"
        centerline_json = dataset_dir / "diverge_route_centerlines_asymmetric" / f"{token}.json"
        row = {
            "sample_token": token,
            "scene_name": match_by_token.get(token, {}).get("scene_name", ""),
            "scene_pos": match_by_token.get(token, {}).get("scene_pos", ""),
            "scene_len": match_by_token.get(token, {}).get("scene_len", ""),
            "is_temporal_eligible": temporal_by_token.get(token, {}).get(
                "is_temporal_eligible",
                match_by_token.get(token, {}).get("temporal_eligible_W10_L19", ""),
            ),
            "is_phase1_selected": token in phase1_by_token,
            "is_primary_scene_sample": phase1_by_token.get(token, {}).get(
                "is_primary_scene_sample", ""
            ),
            "has_scene_json": scene_json.exists(),
            "scene_json": str(scene_json) if scene_json.exists() else "",
            "has_centerline_json": centerline_json.exists(),
            "centerline_json": str(centerline_json) if centerline_json.exists() else "",
            "has_patch_rsa_file": patch_paths["rsa"].exists(),
            "patch_rsa_file": str(patch_paths["rsa"]) if patch_paths["rsa"].exists() else "",
            "has_patch_eta_file": patch_paths["eta"].exists(),
            "patch_eta_file": str(patch_paths["eta"]) if patch_paths["eta"].exists() else "",
        }
        row.update(loc_fields("rsa", blind_locs["rsa"], token))
        row.update(loc_fields("eta", blind_locs["eta"], token))
        rows.append(row)

    fieldnames = [
        "sample_token",
        "scene_name",
        "scene_pos",
        "scene_len",
        "is_temporal_eligible",
        "is_phase1_selected",
        "is_primary_scene_sample",
        "has_scene_json",
        "scene_json",
        "has_centerline_json",
        "centerline_json",
        "has_blind_rsa_loc",
        "blind_rsa_x",
        "blind_rsa_y",
        "blind_rsa_z",
        "has_blind_eta_loc",
        "blind_eta_x",
        "blind_eta_y",
        "blind_eta_z",
        "has_patch_rsa_file",
        "patch_rsa_file",
        "has_patch_eta_file",
        "patch_eta_file",
    ]
    index_csv = out_dir / "ccs25_attack_asset_index.csv"
    write_csv(index_csv, rows, fieldnames)

    temporal_rows = [
        row for row in rows if str(row["is_temporal_eligible"]).lower() == "true"
    ]
    temporal_csv = out_dir / "temporal_eligible_attack_assets_W10_L19.csv"
    write_csv(temporal_csv, temporal_rows, fieldnames)

    phase1_rows = [row for row in rows if row["is_phase1_selected"]]
    phase1_csv = out_dir / "phase1_attack_assets.csv"
    write_csv(phase1_csv, phase1_rows, fieldnames)

    summary = {
        "ccs25_root": str(ccs25_root),
        "asset_index_csv": str(index_csv),
        "temporal_eligible_asset_csv": str(temporal_csv),
        "phase1_asset_csv": str(phase1_csv),
        "seed_tokens": len(rows),
        "scene_json": sum(row["has_scene_json"] for row in rows),
        "centerline_json": sum(row["has_centerline_json"] for row in rows),
        "blind_rsa_locs": sum(row["has_blind_rsa_loc"] for row in rows),
        "blind_eta_locs": sum(row["has_blind_eta_loc"] for row in rows),
        "temporal_eligible_frames": len(temporal_rows),
        "temporal_eligible_with_blind_rsa_loc": sum(
            row["has_blind_rsa_loc"] for row in temporal_rows
        ),
        "temporal_eligible_with_blind_eta_loc": sum(
            row["has_blind_eta_loc"] for row in temporal_rows
        ),
        "phase1_selected_frames": len(phase1_rows),
        "phase1_with_blind_rsa_loc": sum(row["has_blind_rsa_loc"] for row in phase1_rows),
        "phase1_with_blind_eta_loc": sum(row["has_blind_eta_loc"] for row in phase1_rows),
        "has_patch_rsa_file": patch_paths["rsa"].exists(),
        "has_patch_eta_file": patch_paths["eta"].exists(),
    }
    summary_path = out_dir / "ccs25_attack_asset_index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
