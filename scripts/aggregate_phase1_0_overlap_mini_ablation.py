#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def summarize_numeric(rows, group_fields, value_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(group)
        for field in value_fields:
            values = [to_float(row.get(field)) for row in group]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_median"] = median(values)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase1.0 overlap mini-ablation outputs.")
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_dir = Path(args.out_dir)
    tokens = read_tokens(args.tokens_file)

    internal_rows = []
    internal_reductions = []
    map_delta_rows = []
    available_tokens = []
    missing = []
    for token in tokens:
        root = out_root / token
        internal_path = root / "phase1_0_attack_reset_ablation" / "phase1_0_single_sequence_reset_ablation_matched_baseline.csv"
        reduction_path = root / "phase1_0_attack_reset_ablation" / "phase1_0_single_sequence_reset_ablation_matched_reductions.csv"
        map_path = root / "phase1_0_map_level" / "phase1_0_single_sequence_map_matched_deltas.csv"
        if not (internal_path.exists() and reduction_path.exists() and map_path.exists()):
            missing.append(
                {
                    "target_token": token,
                    "internal_exists": internal_path.exists(),
                    "reduction_exists": reduction_path.exists(),
                    "map_exists": map_path.exists(),
                }
            )
            continue
        available_tokens.append(token)
        for row in read_csv(internal_path):
            row["target_token"] = token
            internal_rows.append(row)
        for row in read_csv(reduction_path):
            row["target_token"] = token
            internal_reductions.append(row)
        for row in read_csv(map_path):
            row["target_token"] = token
            map_delta_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    if internal_rows:
        write_csv(out_dir / "overlap_internal_matched_baseline_all.csv", internal_rows, list(internal_rows[0].keys()))
        internal_summary = summarize_numeric(
            internal_rows,
            ["condition", "frame_offset"],
            [
                "query_score_mean_abs",
                "pred_vector_mean_abs",
                "topk_embedding_mean_abs",
                "current_bev_norm_delta",
                "fused_bev_norm_delta",
            ],
        )
        write_csv(out_dir / "overlap_internal_matched_baseline_summary.csv", internal_summary, list(internal_summary[0].keys()))
    else:
        internal_summary = []

    if internal_reductions:
        write_csv(out_dir / "overlap_internal_matched_reductions_all.csv", internal_reductions, list(internal_reductions[0].keys()))
        reduction_summary = summarize_numeric(
            internal_reductions,
            ["condition", "frame_offset"],
            [
                "query_score_mean_abs_reduction",
                "pred_vector_mean_abs_reduction",
                "topk_embedding_mean_abs_reduction",
                "current_bev_norm_delta_reduction",
                "fused_bev_norm_delta_reduction",
            ],
        )
        write_csv(out_dir / "overlap_internal_matched_reductions_summary.csv", reduction_summary, list(reduction_summary[0].keys()))
    else:
        reduction_summary = []

    if map_delta_rows:
        write_csv(out_dir / "overlap_map_matched_deltas_all.csv", map_delta_rows, list(map_delta_rows[0].keys()))
        map_summary = summarize_numeric(
            map_delta_rows,
            ["attack_condition", "frame_offset"],
            [
                "delta_cd_to_diverge_m",
                "delta_cd_to_reference_m",
                "delta_wrong_reference_preference_m",
            ],
        )
        write_csv(out_dir / "overlap_map_matched_deltas_summary.csv", map_summary, list(map_summary[0].keys()))
    else:
        map_summary = []

    summary = {
        "tokens_file": str(Path(args.tokens_file)),
        "out_root": str(out_root),
        "requested_tokens": len(tokens),
        "available_tokens": len(available_tokens),
        "available_token_list": available_tokens,
        "missing": missing,
        "internal_summary_csv": str(out_dir / "overlap_internal_matched_baseline_summary.csv"),
        "internal_reductions_summary_csv": str(out_dir / "overlap_internal_matched_reductions_summary.csv"),
        "map_summary_csv": str(out_dir / "overlap_map_matched_deltas_summary.csv"),
    }
    (out_dir / "overlap_mini_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
