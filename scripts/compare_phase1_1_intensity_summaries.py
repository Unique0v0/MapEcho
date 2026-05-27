#!/usr/bin/env python3
import argparse
import csv
import json
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


def read_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def summarize(rows, label, tokens):
    token_set = set(tokens)
    out = []
    for condition in ["attack_keep", "attack_reset_all", "attack_reset_query", "attack_reset_bev"]:
        for offset in [1, 2]:
            subset = [
                row
                for row in rows
                if row.get("target_token") in token_set
                and row.get("attack_condition") == condition
                and int(row.get("frame_offset", -999)) == offset
            ]
            vals = [float(row["delta_cd_to_diverge_m"]) for row in subset]
            out.append(
                {
                    "label": label,
                    "condition": condition,
                    "frame_offset": offset,
                    "n": len(vals),
                    "median_delta_cd_diverge_m": median(vals),
                    "positive_rate_gt_0p01": sum(value > 0.01 for value in vals) / len(vals) if vals else None,
                    "positive_count_gt_0p01": sum(value > 0.01 for value in vals),
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare Phase 1.1 intensity sensitivity summaries.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-label", default="power_3000")
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--intensity-root", required=True)
    parser.add_argument("--powers", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    tokens = read_tokens(args.tokens_file)
    rows = summarize(read_csv(args.baseline_summary), args.baseline_label, tokens)
    for power in args.powers.split():
        safe_power = power.replace(".", "p")
        path = Path(args.intensity_root) / f"power_{safe_power}" / "summary" / "phase1_1_map_matched_deltas_enriched.csv"
        if not path.exists():
            continue
        rows.extend(summarize(read_csv(path), f"power_{power}", tokens))

    out_dir = Path(args.out_dir)
    out_csv = out_dir / "phase1_1_intensity_map_comparison.csv"
    write_csv(out_csv, rows, list(rows[0].keys()) if rows else [])
    report = {
        "tokens_file": args.tokens_file,
        "n_tokens": len(tokens),
        "baseline_summary": args.baseline_summary,
        "intensity_root": args.intensity_root,
        "powers": args.powers.split(),
        "comparison_csv": str(out_csv),
    }
    (out_dir / "phase1_1_intensity_comparison_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
