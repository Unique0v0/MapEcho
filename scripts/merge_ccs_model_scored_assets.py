#!/usr/bin/env python3
import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser(description="Merge per-token CCS model-scored best-location assets.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-tokens", required=True)
    parser.add_argument("--asset-filename", default="ccs_model_scored_best_location_asset.csv")
    args = parser.parse_args()

    tokens = [line.strip() for line in Path(args.tokens_file).read_text().splitlines() if line.strip()]
    rows = []
    missing = []
    for token in tokens:
        path = Path(args.out_root) / token / args.asset_filename
        if not path.exists():
            missing.append(token)
            continue
        item_rows = read_csv(path)
        if not item_rows:
            missing.append(token)
            continue
        rows.append(item_rows[0])

    rows.sort(key=lambda row: (row.get("scene_name", ""), row.get("scene_pos", ""), row.get("sample_token", "")))
    write_csv(Path(args.out_csv), rows)
    Path(args.out_tokens).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tokens).write_text("\n".join(row["sample_token"] for row in rows) + ("\n" if rows else ""))
    report = {
        "requested_tokens": len(tokens),
        "merged_assets": len(rows),
        "missing": missing,
        "out_csv": args.out_csv,
        "out_tokens": args.out_tokens,
        "asset_filename": args.asset_filename,
    }
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
