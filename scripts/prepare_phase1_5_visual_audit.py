#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def collect_paths(run_root, token):
    base = Path(run_root) / token
    fig_dir = base / "phase1_0_map_level" / "figures"
    attack_dir = base / "attack_assets" / "images" / token
    attack_overlays = sorted(attack_dir.glob("*_overlay.png"))
    return {
        "attack_overlay": str(attack_overlays[0]) if attack_overlays else "",
        "map_overlay_t0": str(fig_dir / "offset_+0_boundary_overlay.png"),
        "map_overlay_t1": str(fig_dir / "offset_+1_boundary_overlay.png"),
        "map_overlay_t2": str(fig_dir / "offset_+2_boundary_overlay.png"),
    }


def make_thumb(path, size):
    if not path:
        return Image.new("RGB", size, (245, 245, 245))
    p = Path(path)
    if not p.exists():
        return Image.new("RGB", size, (245, 245, 245))
    img = Image.open(p).convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_label(draw, xy, text):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.multiline_text(xy, text, fill=(10, 10, 10), font=font, spacing=3)


def contact_sheet(rows, out_path, title, thumb_size=(300, 220)):
    cols = ["attack_overlay", "map_overlay_t0", "map_overlay_t1", "map_overlay_t2"]
    header_h = 70
    label_w = 290
    row_h = thumb_size[1] + 26
    width = label_w + len(cols) * thumb_size[0]
    height = header_h + max(1, len(rows)) * row_h
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw_label(draw, (12, 10), title)
    for i, col in enumerate(cols):
        draw_label(draw, (label_w + i * thumb_size[0] + 8, 48), col)
    for r, row in enumerate(rows):
        y = header_h + r * row_h
        label = (
            f"{row['audit_group']} #{row['audit_rank']}\n"
            f"{row['sample_token'][:8]}  {row['scene_name']}\n"
            f"t0={row['delta_cd_diverge_t']}  "
            f"t1={row['delta_cd_diverge_t1_attack_keep']}  "
            f"t2={row['delta_cd_diverge_t2_attack_keep']}"
        )
        draw_label(draw, (10, y + 8), label)
        for c, col in enumerate(cols):
            thumb = make_thumb(row[col], thumb_size)
            sheet.paste(thumb, (label_w + c * thumb_size[0], y))
        draw.line((0, y + row_h - 1, width, y + row_h - 1), fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def add_group(out, source_rows, group_name, limit=None):
    rows = source_rows[:limit] if limit is not None else source_rows
    for idx, row in enumerate(rows, 1):
        item = dict(row)
        item["audit_group"] = group_name
        item["audit_rank"] = idx
        out.append(item)


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase 1.5 lightweight visual audit and frozen inputs.")
    parser.add_argument(
        "--geometry-v2-table",
        default="/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/geometry_quality_v2_table.csv",
    )
    parser.add_argument(
        "--top-bottom-csv",
        default="/data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics/phase1_2_top_bottom_failure_cases.csv",
    )
    parser.add_argument(
        "--relaxed-tokens",
        default="/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/high_quality_relaxed_v2_tokens.txt",
    )
    parser.add_argument(
        "--relaxed-assets",
        default="/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/high_quality_relaxed_v2_assets.csv",
    )
    parser.add_argument(
        "--run-root",
        default="/data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000",
    )
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_5_visual_audit")
    parser.add_argument("--freeze-dir", default="/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    freeze_dir = Path(args.freeze_dir)
    geometry_rows = read_csv(args.geometry_v2_table)
    top_bottom_rows = read_csv(args.top_bottom_csv)

    audit_rows = []
    false_neg = [row for row in geometry_rows if yes(row.get("geometry_false_negative_case"))]
    false_pos = [row for row in geometry_rows if yes(row.get("geometry_false_positive_case"))]
    top = [row for row in top_bottom_rows if row.get("rank_group") == "top_t1_residue"]
    bottom = [row for row in top_bottom_rows if row.get("rank_group") == "bottom_t1_failure"]

    add_group(audit_rows, false_neg, "old_geometry_false_negative")
    add_group(audit_rows, false_pos, "old_geometry_false_positive")
    add_group(audit_rows, top, "top_residue_t1", limit=5)
    add_group(audit_rows, bottom, "bottom_failure_t1", limit=5)

    normalized = []
    seen_group_token = set()
    for row in audit_rows:
        token = row.get("sample_token") or row.get("target_token")
        key = (row["audit_group"], token)
        if key in seen_group_token:
            continue
        seen_group_token.add(key)
        paths = collect_paths(args.run_root, token)
        item = {
            "audit_group": row["audit_group"],
            "audit_rank": row["audit_rank"],
            "sample_token": token,
            "scene_name": row.get("scene_name", ""),
            "visible_cam": row.get("visible_cam", ""),
            "tag_confidence": row.get("tag_confidence", ""),
            "asymmetry_score": row.get("asymmetry_score", ""),
            "centrality_score": row.get("centrality_score", ""),
            "vpa_coverage": row.get("vpa_coverage", row.get("diverge_vpa_coverage", "")),
            "delta_cd_diverge_t": row.get("delta_cd_diverge_t", row.get("delta_cd_t0", "")),
            "delta_cd_diverge_t1_attack_keep": row.get(
                "delta_cd_diverge_t1_attack_keep", row.get("delta_cd_t1", "")
            ),
            "delta_cd_diverge_t2_attack_keep": row.get(
                "delta_cd_diverge_t2_attack_keep", row.get("delta_cd_t2", "")
            ),
            "geometry_v2_relaxed_pass": row.get("geometry_v2_relaxed_pass", ""),
            "geometry_v2_strict_pass": row.get("geometry_v2_strict_pass", ""),
            "scene_json": row.get("scene_json", ""),
            **paths,
        }
        normalized.append(item)

    manifest = out_dir / "visual_audit_manifest.csv"
    write_csv(manifest, normalized)

    by_group = {}
    for row in normalized:
        by_group.setdefault(row["audit_group"], []).append(row)
    contact_paths = {}
    for group, rows in by_group.items():
        out_path = out_dir / f"{group}_contact_sheet.png"
        contact_sheet(rows, out_path, group)
        contact_paths[group] = str(out_path)

    freeze_dir.mkdir(parents=True, exist_ok=True)
    frozen_tokens = freeze_dir / "phase1_5_high_quality_relaxed_v2_tokens.txt"
    frozen_assets = freeze_dir / "phase1_5_high_quality_relaxed_v2_assets.csv"
    shutil.copyfile(args.relaxed_tokens, frozen_tokens)
    shutil.copyfile(args.relaxed_assets, frozen_assets)

    summary = {
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "contact_sheets": contact_paths,
        "counts": {group: len(rows) for group, rows in by_group.items()},
        "total_manifest_rows": len(normalized),
        "freeze_dir": str(freeze_dir),
        "frozen_tokens": str(frozen_tokens),
        "frozen_assets": str(frozen_assets),
    }
    (out_dir / "phase1_5_visual_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
