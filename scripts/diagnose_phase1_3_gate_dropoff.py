#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


GATES = [
    ("geometry", "geometry_gate_pass"),
    ("clean_correct", "clean_correct_gate_pass"),
    ("clean_stable", "clean_stable_gate_pass"),
    ("vpa", "vpa_gate_015"),
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


def summarize_subset(name, rows, positive_threshold):
    t1 = [as_float(row["delta_cd_diverge_t1_attack_keep"]) for row in rows]
    t2 = [as_float(row["delta_cd_diverge_t2_attack_keep"]) for row in rows]
    t0 = [as_float(row["delta_cd_diverge_t"]) for row in rows]
    vpa = [as_float(row["vpa_coverage"]) for row in rows]
    return {
        "subset": name,
        "frames": len(rows),
        "scenes": len({row["scene_name"] for row in rows}),
        "median_vpa": median(vpa),
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
    }


def pass_all(row, gates):
    return all(yes(row[field]) for _, field in gates)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Phase 1.3 pre-attack gate drop-off.")
    parser.add_argument(
        "--gate-table",
        default="/data/dj/MapEcho/artifacts/phase1_3_sample_gates/phase1_3_gate_table.csv",
    )
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_3_sample_gates/dropoff")
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = read_csv(args.gate_table)

    single_rows = []
    for name, field in GATES:
        passed = [row for row in rows if yes(row[field])]
        failed = [row for row in rows if not yes(row[field])]
        single_rows.append(
            {
                "gate": name,
                "field": field,
                "pass_frames": len(passed),
                "pass_scenes": len({row["scene_name"] for row in passed}),
                "fail_frames": len(failed),
                "fail_scenes": len({row["scene_name"] for row in failed}),
                "pass_rate": len(passed) / len(rows) if rows else None,
            }
        )
    write_csv(out_dir / "phase1_3_single_gate_dropoff.csv", single_rows)

    cumulative = []
    active = []
    for gate in GATES:
        active.append(gate)
        subset = [row for row in rows if pass_all(row, active)]
        cumulative.append(
            {
                "step": "+".join(name for name, _ in active),
                "frames": len(subset),
                "scenes": len({row["scene_name"] for row in subset}),
                "dropped_from_previous": (
                    len(rows) - len(subset) if len(active) == 1 else cumulative[-1]["frames"] - len(subset)
                ),
            }
        )
    write_csv(out_dir / "phase1_3_cumulative_gate_dropoff.csv", cumulative)

    exact_rows = []
    buckets = defaultdict(list)
    for row in rows:
        failed = tuple(name for name, field in GATES if not yes(row[field]))
        key = "pass_all" if not failed else "fail:" + ",".join(failed)
        buckets[key].append(row)
    for key, subset in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        exact_rows.append(summarize_subset(key, subset, args.positive_threshold))
    write_csv(out_dir / "phase1_3_exact_failure_patterns.csv", exact_rows)

    relaxation_specs = {
        "all_gates": GATES,
        "drop_geometry": [gate for gate in GATES if gate[0] != "geometry"],
        "drop_clean_correct": [gate for gate in GATES if gate[0] != "clean_correct"],
        "drop_clean_stable": [gate for gate in GATES if gate[0] != "clean_stable"],
        "drop_vpa": [gate for gate in GATES if gate[0] != "vpa"],
        "clean_vpa_only": [gate for gate in GATES if gate[0] in {"clean_correct", "clean_stable", "vpa"}],
        "geometry_vpa_only": [gate for gate in GATES if gate[0] in {"geometry", "vpa"}],
        "geometry_clean_correct_vpa": [
            gate for gate in GATES if gate[0] in {"geometry", "clean_correct", "vpa"}
        ],
        "geometry_clean_stable_vpa": [
            gate for gate in GATES if gate[0] in {"geometry", "clean_stable", "vpa"}
        ],
        "vpa_only": [gate for gate in GATES if gate[0] == "vpa"],
        "geometry_only": [gate for gate in GATES if gate[0] == "geometry"],
    }
    relaxation_rows = []
    for name, gates in relaxation_specs.items():
        subset = [row for row in rows if pass_all(row, gates)]
        item = summarize_subset(name, subset, args.positive_threshold)
        item["required_gates"] = "+".join(gate_name for gate_name, _ in gates)
        relaxation_rows.append(item)
    write_csv(out_dir / "phase1_3_gate_relaxation_scenarios.csv", relaxation_rows)

    scene_rows = []
    for scenario in ["all_gates", "drop_geometry", "drop_clean_correct", "drop_clean_stable"]:
        gates = relaxation_specs[scenario]
        subset = [row for row in rows if pass_all(row, gates)]
        by_scene = defaultdict(list)
        for row in subset:
            by_scene[row["scene_name"]].append(row)
        for scene, scene_rows_for_subset in sorted(by_scene.items()):
            item = summarize_subset(f"{scenario}:{scene}", scene_rows_for_subset, args.positive_threshold)
            item["scenario"] = scenario
            item["scene_name"] = scene
            scene_rows.append(item)
    write_csv(out_dir / "phase1_3_gate_relaxation_by_scene.csv", scene_rows)

    report = {
        "gate_table": args.gate_table,
        "out_dir": str(out_dir),
        "n_frames": len(rows),
        "n_scenes": len({row["scene_name"] for row in rows}),
        "single_gate_dropoff_csv": str(out_dir / "phase1_3_single_gate_dropoff.csv"),
        "cumulative_dropoff_csv": str(out_dir / "phase1_3_cumulative_gate_dropoff.csv"),
        "failure_patterns_csv": str(out_dir / "phase1_3_exact_failure_patterns.csv"),
        "relaxation_scenarios_csv": str(out_dir / "phase1_3_gate_relaxation_scenarios.csv"),
        "relaxation_by_scene_csv": str(out_dir / "phase1_3_gate_relaxation_by_scene.csv"),
    }
    (out_dir / "phase1_3_gate_dropoff_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
