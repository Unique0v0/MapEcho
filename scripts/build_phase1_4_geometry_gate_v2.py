#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


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


def write_tokens(path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + ("\n" if tokens else ""))


def yes(value):
    return str(value).strip().lower() == "true"


def as_float(value):
    if value in ("", None):
        return None
    return float(value)


def median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    n = len(values)
    if n % 2:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2.0


def score_component(value, lo, hi):
    if value is None:
        return 0.0
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / (hi - lo)


def summarize(name, rows, positive_threshold):
    t0 = [as_float(row["delta_cd_diverge_t"]) for row in rows]
    t1 = [as_float(row["delta_cd_diverge_t1_attack_keep"]) for row in rows]
    t2 = [as_float(row["delta_cd_diverge_t2_attack_keep"]) for row in rows]
    attack_effective = [row for row in rows if (as_float(row["delta_cd_diverge_t"]) or -999.0) > 0.01]
    ae_t1 = [as_float(row["delta_cd_diverge_t1_attack_keep"]) for row in attack_effective]
    ae_t2 = [as_float(row["delta_cd_diverge_t2_attack_keep"]) for row in attack_effective]
    return {
        "set_name": name,
        "frames": len(rows),
        "scenes": len({row["scene_name"] for row in rows}),
        "median_attack_frame_delta_cd": median(t0),
        "t1_median_delta_cd": median(t1),
        "t1_positive_count": sum(value is not None and value > positive_threshold for value in t1),
        "t1_positive_rate": (
            sum(value is not None and value > positive_threshold for value in t1) / len(t1)
            if t1
            else None
        ),
        "t2_median_delta_cd": median(t2),
        "t2_positive_count": sum(value is not None and value > positive_threshold for value in t2),
        "t2_positive_rate": (
            sum(value is not None and value > positive_threshold for value in t2) / len(t2)
            if t2
            else None
        ),
        "attack_effective_delta001_frames": len(attack_effective),
        "attack_effective_delta001_scenes": len({row["scene_name"] for row in attack_effective}),
        "attack_effective_delta001_t1_median_delta_cd": median(ae_t1),
        "attack_effective_delta001_t1_positive_count": sum(
            value is not None and value > positive_threshold for value in ae_t1
        ),
        "attack_effective_delta001_t2_median_delta_cd": median(ae_t2),
        "attack_effective_delta001_t2_positive_count": sum(
            value is not None and value > positive_threshold for value in ae_t2
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Build Phase 1.4 geometry-gate v2 tables and sets.")
    parser.add_argument(
        "--gate-table",
        default="/data/dj/MapEcho/artifacts/phase1_3_sample_gates/phase1_3_gate_table.csv",
    )
    parser.add_argument(
        "--asset-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv",
    )
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2")
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = read_csv(args.gate_table)
    asset_rows = read_csv(args.asset_csv)
    assets = {row["sample_token"]: row for row in asset_rows}

    out_rows = []
    for row in rows:
        asset = assets.get(row["sample_token"], {})
        tag = as_float(row["tag_confidence"]) or 0.0
        asym = as_float(row["asymmetry_score"]) or 0.0
        centrality = as_float(asset.get("centrality_score")) or 0.0
        selection_score = as_float(asset.get("selection_score")) or 0.0
        vpa = as_float(row["vpa_coverage"]) or 0.0
        clean_correct = yes(row["clean_correct_gate_pass"])
        clean_stable = yes(row["clean_stable_gate_pass"])
        old_geometry = yes(row["geometry_gate_pass"])
        cam = row.get("visible_cam", "")

        tag_component = score_component(tag, 0.15, 0.70)
        asym_component = score_component(asym, 0.05, 0.25)
        centrality_component = score_component(centrality, 0.55, 0.90)
        geometry_quality_v2_score = (
            0.35 * tag_component + 0.40 * asym_component + 0.25 * centrality_component
        )

        weak_geometry_signal = tag >= 0.15 or asym >= 0.10 or centrality >= 0.70
        moderate_geometry_signal = tag >= 0.20 or asym >= 0.12 or centrality >= 0.75
        camera_gate = cam != "CAM_FRONT_RIGHT"
        vpa_gate = vpa >= 0.15
        strict_pass = bool(vpa_gate and clean_correct and clean_stable and camera_gate and moderate_geometry_signal)
        relaxed_pass = bool(vpa_gate and (clean_correct or clean_stable) and camera_gate and weak_geometry_signal)

        attack_effective = (as_float(row["delta_cd_diverge_t"]) or -999.0) > 0.01
        recovery_positive = (
            (as_float(row["delta_cd_diverge_t1_attack_keep"]) or -999.0) > args.positive_threshold
            or (as_float(row["delta_cd_diverge_t2_attack_keep"]) or -999.0) > args.positive_threshold
        )
        geometry_false_negative = (not old_geometry) and attack_effective and recovery_positive
        geometry_false_positive = old_geometry and (not attack_effective) and (not recovery_positive)

        item = dict(row)
        item.update(
            {
                "centrality_score": centrality,
                "selection_score": selection_score,
                "geometry_quality_v2_score": geometry_quality_v2_score,
                "geometry_v2_tag_component": tag_component,
                "geometry_v2_asymmetry_component": asym_component,
                "geometry_v2_centrality_component": centrality_component,
                "geometry_v2_weak_signal": str(weak_geometry_signal).lower(),
                "geometry_v2_moderate_signal": str(moderate_geometry_signal).lower(),
                "geometry_v2_camera_gate": str(camera_gate).lower(),
                "geometry_v2_strict_pass": str(strict_pass).lower(),
                "geometry_v2_relaxed_pass": str(relaxed_pass).lower(),
                "geometry_v2_rule": (
                    "vpa>=0.15; camera!=CAM_FRONT_RIGHT; "
                    "strict uses clean_correct&clean_stable&moderate_geometry; "
                    "relaxed uses (clean_correct|clean_stable)&weak_geometry"
                ),
                "geometry_false_negative_case": str(geometry_false_negative).lower(),
                "geometry_false_positive_case": str(geometry_false_positive).lower(),
                "attack_effective_delta001": str(attack_effective).lower(),
                "recovery_positive_t1_or_t2": str(recovery_positive).lower(),
            }
        )
        out_rows.append(item)

    write_csv(out_dir / "geometry_quality_v2_table.csv", out_rows)

    asset_fieldnames = list(asset_rows[0].keys()) if asset_rows else []
    sets = {
        "high_quality_strict_v2": [row for row in out_rows if yes(row["geometry_v2_strict_pass"])],
        "high_quality_relaxed_v2": [row for row in out_rows if yes(row["geometry_v2_relaxed_pass"])],
    }
    for name, set_rows in sets.items():
        tokens = [row["sample_token"] for row in set_rows]
        write_tokens(out_dir / f"{name}_tokens.txt", tokens)
        write_csv(
            out_dir / f"{name}_assets.csv",
            [assets[token] for token in tokens if token in assets],
            asset_fieldnames,
        )

    false_negative_rows = [row for row in out_rows if yes(row["geometry_false_negative_case"])]
    false_positive_rows = [row for row in out_rows if yes(row["geometry_false_positive_case"])]
    write_csv(out_dir / "geometry_false_negative_cases.csv", false_negative_rows)
    write_csv(out_dir / "geometry_false_positive_cases.csv", false_positive_rows)

    summary_rows = [
        summarize("broad_report_set", out_rows, args.positive_threshold),
        summarize("high_quality_strict_v2", sets["high_quality_strict_v2"], args.positive_threshold),
        summarize("high_quality_relaxed_v2", sets["high_quality_relaxed_v2"], args.positive_threshold),
        summarize("geometry_false_negative_cases", false_negative_rows, args.positive_threshold),
        summarize("geometry_false_positive_cases", false_positive_rows, args.positive_threshold),
    ]
    write_csv(out_dir / "phase1_4_geometry_v2_set_summary.csv", summary_rows)

    report = {
        "out_dir": str(out_dir),
        "gate_table": args.gate_table,
        "asset_csv": args.asset_csv,
        "geometry_quality_v2_table": str(out_dir / "geometry_quality_v2_table.csv"),
        "high_quality_strict_v2": {
            "frames": len(sets["high_quality_strict_v2"]),
            "scenes": len({row["scene_name"] for row in sets["high_quality_strict_v2"]}),
            "tokens": str(out_dir / "high_quality_strict_v2_tokens.txt"),
            "assets": str(out_dir / "high_quality_strict_v2_assets.csv"),
        },
        "high_quality_relaxed_v2": {
            "frames": len(sets["high_quality_relaxed_v2"]),
            "scenes": len({row["scene_name"] for row in sets["high_quality_relaxed_v2"]}),
            "tokens": str(out_dir / "high_quality_relaxed_v2_tokens.txt"),
            "assets": str(out_dir / "high_quality_relaxed_v2_assets.csv"),
        },
        "geometry_false_negative_cases": {
            "frames": len(false_negative_rows),
            "scenes": len({row["scene_name"] for row in false_negative_rows}),
            "csv": str(out_dir / "geometry_false_negative_cases.csv"),
        },
        "geometry_false_positive_cases": {
            "frames": len(false_positive_rows),
            "scenes": len({row["scene_name"] for row in false_positive_rows}),
            "csv": str(out_dir / "geometry_false_positive_cases.csv"),
        },
        "set_summary_csv": str(out_dir / "phase1_4_geometry_v2_set_summary.csv"),
        "acceptance_relaxed_ge_25_frames_8_scenes": (
            len(sets["high_quality_relaxed_v2"]) >= 25
            and len({row["scene_name"] for row in sets["high_quality_relaxed_v2"]}) >= 8
        ),
    }
    (out_dir / "phase1_4_geometry_v2_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
