#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


CONDITIONS = [
    "attack_keep",
    "attack_reset_all",
    "attack_reset_query",
    "attack_reset_bev",
]


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


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def bool_text(value):
    return "true" if value else "false"


def percentile(values, q):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def std(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if len(values) < 2:
        return 0.0 if values else None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def summarize_values(values, positive_threshold):
    vals = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "median": percentile(vals, 0.5),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "positive_count": sum(value > positive_threshold for value in vals),
        "positive_rate": sum(value > positive_threshold for value in vals) / len(vals) if vals else None,
    }


def vpa_bin(vpa):
    if vpa is None:
        return "missing"
    if vpa >= 0.25:
        return "high_ge_0p25"
    if vpa >= 0.15:
        return "mid_0p15_0p25"
    return "low_lt_0p15"


def gate_reason(parts):
    failed = [name for name, passed in parts if not passed]
    return "pass" if not failed else "fail:" + ",".join(failed)


def rows_by_token_condition(rows):
    out = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        token = row["target_token"]
        condition = row["attack_condition"]
        offset = int(row["frame_offset"])
        out[token][condition][offset] = row
    return out


def get_delta(by_condition, condition, offset, field="delta_cd_to_diverge_m"):
    row = by_condition.get(condition, {}).get(offset, {})
    return to_float(row.get(field))


def get_row(by_condition, condition, offset):
    return by_condition.get(condition, {}).get(offset, {})


def main():
    parser = argparse.ArgumentParser(description="Build Phase 1.3 gate-based sample sets.")
    parser.add_argument(
        "--summary-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000/summary/phase1_1_map_matched_deltas_enriched.csv",
    )
    parser.add_argument(
        "--asset-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv",
    )
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_3_sample_gates")
    parser.add_argument("--source-stage", default="ccs_candidate_expanded_vpa015")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=9)
    parser.add_argument("--attack-power", type=int, default=6000)
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    parser.add_argument("--geometry-tag-threshold", type=float, default=0.4)
    parser.add_argument("--vpa-threshold", type=float, default=0.15)
    parser.add_argument("--clean-correct-quantile", type=float, default=0.75)
    parser.add_argument("--clean-stable-quantile", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    map_rows = read_csv(args.summary_csv)
    asset_rows = read_csv(args.asset_csv)
    assets = {row["sample_token"]: row for row in asset_rows}
    grouped = rows_by_token_condition(map_rows)
    tokens = sorted(grouped.keys(), key=lambda token: (assets.get(token, {}).get("scene_name", ""), assets.get(token, {}).get("scene_pos", ""), token))

    clean_cd_t0_values = []
    clean_std_values = []
    for token in tokens:
        keep_t0 = get_row(grouped[token], "attack_keep", 0)
        clean_cd_t0 = to_float(keep_t0.get("clean_cd_to_diverge_m"))
        clean_vals = [
            to_float(get_row(grouped[token], "attack_keep", offset).get("clean_cd_to_diverge_m"))
            for offset in [0, 1, 2]
        ]
        clean_cd_t0_values.append(clean_cd_t0)
        clean_std_values.append(std(clean_vals))
    clean_correct_threshold = percentile(clean_cd_t0_values, args.clean_correct_quantile)
    clean_stable_threshold = percentile(clean_std_values, args.clean_stable_quantile)

    gate_rows = []
    for token in tokens:
        asset = assets.get(token, {})
        by_condition = grouped[token]
        keep_t0 = get_row(by_condition, "attack_keep", 0)
        scene_name = asset.get("scene_name") or keep_t0.get("scene_name", "")
        tag_confidence = to_float(asset.get("tag_confidence") or asset.get("mapecho_tag_confidence") or keep_t0.get("tag_confidence"))
        asymmetry_score = to_float(asset.get("asymmetry_score"))
        vpa = to_float(asset.get("diverge_vpa_coverage") or keep_t0.get("diverge_vpa_coverage"))
        reference_vpa = to_float(asset.get("reference_vpa_coverage"))
        clean_vals = [
            to_float(get_row(by_condition, "attack_keep", offset).get("clean_cd_to_diverge_m"))
            for offset in [0, 1, 2]
        ]
        clean_cd_t = clean_vals[0]
        clean_recovery_std = std(clean_vals)

        geometry_parts = [
            ("tag_confidence", tag_confidence is not None and tag_confidence >= args.geometry_tag_threshold),
            ("asymmetry_score", asymmetry_score is not None),
            ("scene_json", bool(asset.get("scene_json"))),
        ]
        geometry_gate_pass = all(passed for _, passed in geometry_parts)
        clean_correct_gate_pass = (
            clean_correct_threshold is not None
            and clean_cd_t is not None
            and clean_cd_t <= clean_correct_threshold
        )
        clean_stable_gate_pass = (
            clean_stable_threshold is not None
            and clean_recovery_std is not None
            and clean_recovery_std <= clean_stable_threshold
        )
        diverge_vpa_pass = bool(vpa is not None and vpa > 0.0)
        reference_vpa_pass = bool(reference_vpa is not None and reference_vpa > 0.0)
        vpa_gate_015 = bool(vpa is not None and vpa >= 0.15)
        vpa_gate_020 = bool(vpa is not None and vpa >= 0.20)
        vpa_gate_025 = bool(vpa is not None and vpa >= 0.25)
        vpa_gate_pass = bool(vpa is not None and vpa >= args.vpa_threshold and not reference_vpa_pass)

        delta_t0 = get_delta(by_condition, "attack_keep", 0)
        attack_effective_0005 = bool(delta_t0 is not None and delta_t0 > 0.005)
        attack_effective_001 = bool(delta_t0 is not None and delta_t0 > 0.01)
        attack_effective_002 = bool(delta_t0 is not None and delta_t0 > 0.02)
        if delta_t0 is None:
            attack_effective_bin = "missing"
        elif delta_t0 > 0.02:
            attack_effective_bin = "strong_gt_0p02"
        elif delta_t0 > 0.01:
            attack_effective_bin = "medium_gt_0p01"
        elif delta_t0 > 0.005:
            attack_effective_bin = "weak_gt_0p005"
        else:
            attack_effective_bin = "not_effective"

        high_quality_parts = [
            ("geometry", geometry_gate_pass),
            ("clean_correct", clean_correct_gate_pass),
            ("clean_stable", clean_stable_gate_pass),
            ("vpa", vpa_gate_pass),
        ]
        high_quality_candidate_pass = all(passed for _, passed in high_quality_parts)

        row = {
            "sample_token": token,
            "scene_token": asset.get("scene_token", ""),
            "scene_name": scene_name,
            "scene_pos": asset.get("scene_pos", ""),
            "scene_len": asset.get("scene_len", ""),
            "source_stage": args.source_stage,
            "W": args.warmup,
            "L": args.recovery,
            "attack_power": args.attack_power,
            "asymmetry_score": asymmetry_score,
            "tag_confidence": tag_confidence,
            "diverge_boundary_id": asset.get("diverge_boundary_id", ""),
            "reference_boundary_id": asset.get("reference_boundary_id", ""),
            "diverge_side": asset.get("diverge_side", ""),
            "geometry_quality_score": tag_confidence,
            "geometry_gate_pass": bool_text(geometry_gate_pass),
            "geometry_gate_reason": gate_reason(geometry_parts),
            "clean_cd_diverge_t": clean_cd_t,
            "clean_cd_diverge_t1": clean_vals[1],
            "clean_cd_diverge_t2": clean_vals[2],
            "clean_recovery_cd_std": clean_recovery_std,
            "clean_correct_gate_pass": bool_text(clean_correct_gate_pass),
            "clean_stable_gate_pass": bool_text(clean_stable_gate_pass),
            "clean_gate_reason": gate_reason(
                [
                    ("clean_correct", clean_correct_gate_pass),
                    ("clean_stable", clean_stable_gate_pass),
                ]
            ),
            "clean_correct_threshold_p75": clean_correct_threshold,
            "clean_stable_threshold_p75": clean_stable_threshold,
            "vpa_coverage": vpa,
            "diverge_vpa_pass": bool_text(diverge_vpa_pass),
            "reference_vpa_pass": bool_text(reference_vpa_pass),
            "vpa_gate_015": bool_text(vpa_gate_015),
            "vpa_gate_020": bool_text(vpa_gate_020),
            "vpa_gate_025": bool_text(vpa_gate_025),
            "vpa_bin": vpa_bin(vpa),
            "attack_cd_diverge_t": to_float(keep_t0.get("attack_cd_to_diverge_m")),
            "delta_cd_diverge_t": delta_t0,
            "attack_effective_gate_0005": bool_text(attack_effective_0005),
            "attack_effective_gate_001": bool_text(attack_effective_001),
            "attack_effective_gate_002": bool_text(attack_effective_002),
            "attack_effective_bin": attack_effective_bin,
            "high_quality_candidate_pass": bool_text(high_quality_candidate_pass),
            "high_quality_candidate_reason": gate_reason(high_quality_parts),
            "is_primary_scene_sample": asset.get("is_primary_scene_sample", keep_t0.get("is_primary_scene_sample", "")),
            "visible_cam": asset.get("visible_cam", ""),
            "scene_json": asset.get("scene_json", ""),
        }
        for condition in CONDITIONS:
            for offset in [1, 2]:
                delta = get_delta(by_condition, condition, offset)
                suffix = f"t{offset}_{condition}"
                row[f"delta_cd_diverge_{suffix}"] = delta
                row[f"{suffix}_positive"] = bool_text(delta is not None and delta > args.positive_threshold)
        gate_rows.append(row)

    write_csv(out_dir / "phase1_3_gate_table.csv", gate_rows)

    asset_by_token = {row["sample_token"]: row for row in asset_rows}
    sets = {
        "broad_report_set": [row["sample_token"] for row in gate_rows],
        "attack_effective_set_delta0005": [
            row["sample_token"] for row in gate_rows if row["attack_effective_gate_0005"] == "true"
        ],
        "attack_effective_set_delta001": [
            row["sample_token"] for row in gate_rows if row["attack_effective_gate_001"] == "true"
        ],
        "attack_effective_set_delta002": [
            row["sample_token"] for row in gate_rows if row["attack_effective_gate_002"] == "true"
        ],
        "high_quality_candidate_set": [
            row["sample_token"] for row in gate_rows if row["high_quality_candidate_pass"] == "true"
        ],
    }
    for set_name, set_tokens in sets.items():
        write_tokens(out_dir / f"{set_name}_tokens.txt", set_tokens)
        write_csv(
            out_dir / f"{set_name}_assets.csv",
            [asset_by_token[token] for token in set_tokens if token in asset_by_token],
            list(asset_rows[0].keys()) if asset_rows else [],
        )

    set_rows = []
    reset_rows = []
    for set_name, set_tokens in sets.items():
        rows = [row for row in gate_rows if row["sample_token"] in set_tokens]
        item = {
            "set_name": set_name,
            "frames": len(rows),
            "scenes": len({row["scene_name"] for row in rows}),
        }
        for offset in [1, 2]:
            values = [to_float(row[f"delta_cd_diverge_t{offset}_attack_keep"]) for row in rows]
            summary = summarize_values(values, args.positive_threshold)
            item[f"t{offset}_median_delta_cd"] = summary["median"]
            item[f"t{offset}_p75_delta_cd"] = summary["p75"]
            item[f"t{offset}_positive_count"] = summary["positive_count"]
            item[f"t{offset}_positive_rate"] = summary["positive_rate"]
        set_rows.append(item)

        reset_item = {
            "set_name": set_name,
            "frames": len(rows),
            "scenes": len({row["scene_name"] for row in rows}),
        }
        for condition in ["attack_reset_all", "attack_reset_bev", "attack_reset_query"]:
            for offset in [1, 2]:
                values = [to_float(row[f"delta_cd_diverge_t{offset}_{condition}"]) for row in rows]
                summary = summarize_values(values, args.positive_threshold)
                reset_item[f"{condition}_t{offset}_positive_count"] = summary["positive_count"]
                reset_item[f"{condition}_t{offset}_positive_rate"] = summary["positive_rate"]
                reset_item[f"{condition}_t{offset}_median_delta_cd"] = summary["median"]
        reset_rows.append(reset_item)
    write_csv(out_dir / "phase1_3_set_summary.csv", set_rows)
    write_csv(out_dir / "phase1_3_reset_summary.csv", reset_rows)

    scene_rows = []
    for set_name, set_tokens in sets.items():
        rows = [row for row in gate_rows if row["sample_token"] in set_tokens]
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row["scene_name"]].append(row)
        for scene, scene_rows_for_set in sorted(by_scene.items()):
            scene_rows.append(
                {
                    "set_name": set_name,
                    "scene_name": scene,
                    "frames": len(scene_rows_for_set),
                    "primary_count": sum(str(row["is_primary_scene_sample"]).lower() == "true" for row in scene_rows_for_set),
                    "t1_positive_count": sum(
                        (to_float(row["delta_cd_diverge_t1_attack_keep"]) or -999) > args.positive_threshold
                        for row in scene_rows_for_set
                    ),
                    "t2_positive_count": sum(
                        (to_float(row["delta_cd_diverge_t2_attack_keep"]) or -999) > args.positive_threshold
                        for row in scene_rows_for_set
                    ),
                    "median_vpa": percentile([row["vpa_coverage"] for row in scene_rows_for_set], 0.5),
                    "median_attack_frame_delta_cd": percentile(
                        [row["delta_cd_diverge_t"] for row in scene_rows_for_set], 0.5
                    ),
                }
            )
    write_csv(out_dir / "phase1_3_scene_coverage_summary.csv", scene_rows)

    report = {
        "out_dir": str(out_dir),
        "summary_csv": args.summary_csv,
        "asset_csv": args.asset_csv,
        "clean_correct_threshold_p75": clean_correct_threshold,
        "clean_stable_threshold_p75": clean_stable_threshold,
        "geometry_tag_threshold": args.geometry_tag_threshold,
        "vpa_threshold": args.vpa_threshold,
        "positive_threshold": args.positive_threshold,
        "sets": {
            set_name: {
                "frames": len(set_tokens),
                "scenes": len({row["scene_name"] for row in gate_rows if row["sample_token"] in set_tokens}),
            }
            for set_name, set_tokens in sets.items()
        },
        "gate_table_csv": str(out_dir / "phase1_3_gate_table.csv"),
        "set_summary_csv": str(out_dir / "phase1_3_set_summary.csv"),
        "reset_summary_csv": str(out_dir / "phase1_3_reset_summary.csv"),
        "scene_summary_csv": str(out_dir / "phase1_3_scene_coverage_summary.csv"),
    }
    (out_dir / "phase1_3_gate_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
