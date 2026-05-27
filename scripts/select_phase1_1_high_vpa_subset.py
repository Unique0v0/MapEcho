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


def main():
    parser = argparse.ArgumentParser(description="Select a high-VPA Phase 1.1 diagnostic subset.")
    parser.add_argument("--probe-assets", default="/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_assets.csv")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset")
    parser.add_argument("--min-vpa", type=float, default=0.265)
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    rows = read_csv(args.probe_assets)
    rows = [row for row in rows if float(row["diverge_vpa_coverage"]) >= args.min_vpa]
    rows.sort(
        key=lambda row: (
            -float(row["diverge_vpa_coverage"]),
            -float(row["selection_score"]),
            row["scene_name"],
            int(row["scene_pos"]),
        )
    )
    selected = rows[: args.max_frames]

    out_dir = Path(args.out_dir)
    out_csv = out_dir / "phase1_1_high_vpa_assets.csv"
    out_tokens = out_dir / "phase1_1_high_vpa_tokens.txt"
    write_csv(out_csv, selected, list(selected[0].keys()) if selected else list(read_csv(args.probe_assets)[0].keys()))
    out_tokens.write_text("\n".join(row["sample_token"] for row in selected) + ("\n" if selected else ""))
    summary = {
        "source": args.probe_assets,
        "min_vpa": args.min_vpa,
        "max_frames": args.max_frames,
        "selected_frames": len(selected),
        "selected_scenes": len({row["scene_name"] for row in selected}),
        "scene_counts": {
            scene: sum(row["scene_name"] == scene for row in selected)
            for scene in sorted({row["scene_name"] for row in selected})
        },
        "median_vpa": sorted(float(row["diverge_vpa_coverage"]) for row in selected)[len(selected) // 2] if selected else None,
        "assets_csv": str(out_csv),
        "tokens_txt": str(out_tokens),
    }
    (out_dir / "phase1_1_high_vpa_selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
