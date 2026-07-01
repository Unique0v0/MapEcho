#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


CONDITIONS = [
    "attack_keep",
    "attack_reset_all",
    "attack_reset_bev",
    "attack_reset_query",
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


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def positive_count(values, threshold):
    values = [v for v in values if v is not None]
    return sum(v > threshold for v in values)


def load_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Package Phase 1.14 patch_rsa pilot summaries.")
    parser.add_argument("--tokens-file", default="/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_tokens.txt")
    parser.add_argument("--optimizer-root", default="/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_optimizer")
    parser.add_argument("--replay-root", default="/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_replay")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_summary")
    parser.add_argument("--positive-thr", type=float, default=0.01)
    args = parser.parse_args()

    tokens = load_tokens(args.tokens_file)
    out_dir = Path(args.out_dir)
    rows = []
    missing = []
    for token in tokens:
        opt_summary_path = Path(args.optimizer_root) / token / "patch_scoring_summary.json"
        deltas_path = Path(args.replay_root) / token / "phase1_0_map_level" / "phase1_0_single_sequence_map_matched_deltas.csv"
        if not opt_summary_path.exists() or not deltas_path.exists():
            missing.append(token)
            continue
        opt_summary = json.loads(opt_summary_path.read_text())
        deltas = read_csv(deltas_path)
        by_key = {
            (row["attack_condition"], int(row["frame_offset"])): row
            for row in deltas
        }
        row = {
            "sample_token": token,
            "sample_idx_hint": "",
            "optimizer_clean_loss": opt_summary.get("clean_loss"),
            "optimizer_best_loss": opt_summary.get("best_loss"),
            "optimizer_best_rank": opt_summary.get("best_rank"),
            "optimizer_best_visible_cameras": " ".join(opt_summary.get("best_visible_cameras", [])),
            "num_locations_optimized": opt_summary.get("num_locations_optimized"),
            "patch_steps": opt_summary.get("patch_steps"),
        }
        for condition in CONDITIONS:
            for offset in [0, 1, 2]:
                delta_row = by_key.get((condition, offset), {})
                if delta_row and not row["sample_idx_hint"]:
                    row["sample_idx_hint"] = delta_row.get("sample_idx", "")
                row[f"{condition}_t{offset}_delta_cd"] = delta_row.get("delta_cd_to_diverge_m", "")
                row[f"{condition}_t{offset}_delta_wrong_ref_pref"] = delta_row.get("delta_wrong_reference_preference_m", "")
        rows.append(row)

    write_csv(out_dir / "patch_rsa_pilot_token_summary.csv", rows)

    condition_summary = []
    for condition in CONDITIONS:
        item = {"condition": condition, "num_tokens": len(rows)}
        for offset in [0, 1, 2]:
            values = [to_float(row.get(f"{condition}_t{offset}_delta_cd")) for row in rows]
            item[f"t{offset}_median_delta_cd"] = median(values)
            item[f"t{offset}_positive_count_gt_{args.positive_thr}"] = positive_count(values, args.positive_thr)
        condition_summary.append(item)
    write_csv(out_dir / "patch_rsa_pilot_condition_summary.csv", condition_summary)

    summary = {
        "tokens_file": args.tokens_file,
        "optimizer_root": args.optimizer_root,
        "replay_root": args.replay_root,
        "out_dir": str(out_dir),
        "num_requested_tokens": len(tokens),
        "num_completed_tokens": len(rows),
        "missing_tokens": missing,
        "positive_threshold": args.positive_thr,
        "token_summary_csv": str(out_dir / "patch_rsa_pilot_token_summary.csv"),
        "condition_summary_csv": str(out_dir / "patch_rsa_pilot_condition_summary.csv"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patch_rsa_pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
