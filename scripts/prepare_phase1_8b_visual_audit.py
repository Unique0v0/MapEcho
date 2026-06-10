#!/usr/bin/env python3
import argparse
import csv
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


def as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def image_source(rule_dir, token):
    for subdir in ("scenes_asymmetric_dist", "scenes_candidate"):
        path = Path(rule_dir) / subdir / f"{token}.png"
        if path.exists():
            return path
    return None


def choose_audit_rows(rows, primary_per_scene, low_conf_n, high_conf_n):
    selected = {}
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["scene_name"], []).append(row)

    for scene_name, scene_rows in sorted(by_scene.items()):
        scene_rows.sort(
            key=lambda row: (
                -as_float(row.get("mapecho_tag_confidence")),
                abs(as_int(row.get("scene_pos")) - (as_int(row.get("scene_len")) - 1) / 2.0),
                row["sample_token"],
            )
        )
        for row in scene_rows[:primary_per_scene]:
            selected[row["sample_token"]] = ("scene_primary", row)

    sorted_low = sorted(rows, key=lambda row: (as_float(row.get("mapecho_tag_confidence")), row["sample_token"]))
    for row in sorted_low[:low_conf_n]:
        selected.setdefault(row["sample_token"], ("low_confidence", row))

    sorted_high = sorted(rows, key=lambda row: (-as_float(row.get("mapecho_tag_confidence")), row["sample_token"]))
    for row in sorted_high[:high_conf_n]:
        selected.setdefault(row["sample_token"], ("high_confidence", row))

    out = []
    for reason, row in selected.values():
        copied = dict(row)
        copied["audit_reason"] = reason
        out.append(copied)
    out.sort(key=lambda row: (row["audit_reason"], row["scene_name"], as_int(row.get("scene_pos")), row["sample_token"]))
    return out


def annotate_image(src, dst, row):
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    label = (
        f"{row['audit_id']} | {row['scene_name']} pos {row.get('scene_pos','')}/{row.get('scene_len','')} | "
        f"{row['sample_token'][:8]}"
    )
    label2 = (
        f"reason={row['audit_reason']} tag_conf={as_float(row.get('mapecho_tag_confidence')):.3f} "
        f"ref_dist={as_float(row.get('mapecho_distance_to_reference_boundary_m')):.2f}m"
    )
    pad = 8
    box_h = 58
    draw.rectangle([0, 0, img.width, box_h], fill=(255, 255, 255))
    draw.text((pad, 6), label, fill=(0, 0, 0), font=font)
    draw.text((pad, 32), label2, fill=(0, 0, 0), font=small)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def make_contact_sheet(image_paths, dst, cols=4, thumb_w=420):
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / img.width
        thumb_h = int(img.height * ratio)
        thumbs.append(img.resize((thumb_w, thumb_h)))
    rows = (len(thumbs) + cols - 1) // cols
    thumb_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (240, 240, 240))
    for i, img in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h
        sheet.paste(img, (x, y))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare visual audit package for Phase 1.8-B selected data.")
    parser.add_argument("--selected-assets", default="/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_assets.csv")
    parser.add_argument("--boundary-tags", default="/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_boundary_tags.csv")
    parser.add_argument("--rule-dir", default="/data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_8b_visual_audit")
    parser.add_argument("--primary-per-scene", type=int, default=1)
    parser.add_argument("--low-conf-n", type=int, default=16)
    parser.add_argument("--high-conf-n", type=int, default=12)
    args = parser.parse_args()

    assets = read_csv(args.selected_assets)
    tags = {row["sample_token"]: row for row in read_csv(args.boundary_tags)}
    merged = []
    for row in assets:
        item = dict(row)
        tag = tags.get(row["sample_token"], {})
        for key in ("diverge_boundary_id", "reference_boundary_id", "diverge_side", "tag_confidence", "asymmetry_score"):
            item[key] = tag.get(key, item.get(key, ""))
        merged.append(item)

    audit_rows = choose_audit_rows(merged, args.primary_per_scene, args.low_conf_n, args.high_conf_n)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    raw_dir = out_dir / "raw_png"
    image_paths = []
    review_rows = []

    for i, row in enumerate(audit_rows, 1):
        audit_id = f"A{i:03d}"
        row["audit_id"] = audit_id
        token = row["sample_token"]
        src = image_source(args.rule_dir, token)
        row["audit_png"] = ""
        row["raw_png"] = ""
        if src:
            raw_dst = raw_dir / f"{audit_id}_{row['scene_name']}_{token[:8]}.png"
            ann_dst = image_dir / f"{audit_id}_{row['scene_name']}_{token[:8]}.png"
            raw_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, raw_dst)
            annotate_image(src, ann_dst, row)
            row["audit_png"] = str(ann_dst)
            row["raw_png"] = str(raw_dst)
            image_paths.append(ann_dst)
        review_rows.append(
            {
                "audit_id": audit_id,
                "sample_token": token,
                "scene_name": row["scene_name"],
                "scene_pos": row.get("scene_pos", ""),
                "scene_len": row.get("scene_len", ""),
                "audit_reason": row["audit_reason"],
                "tag_confidence": row.get("mapecho_tag_confidence", ""),
                "distance_to_reference_boundary_m": row.get("mapecho_distance_to_reference_boundary_m", ""),
                "diverge_boundary_id": row.get("diverge_boundary_id", ""),
                "reference_boundary_id": row.get("reference_boundary_id", ""),
                "audit_png": row["audit_png"],
                "manual_label": "",
                "manual_notes": "",
            }
        )

    write_csv(out_dir / "phase1_8b_visual_audit_cases.csv", audit_rows)
    write_csv(out_dir / "phase1_8b_visual_audit_review_template.csv", review_rows)
    make_contact_sheet(image_paths, out_dir / "phase1_8b_visual_audit_contact_sheet.jpg")
    (out_dir / "phase1_8b_visual_audit_tokens.txt").write_text(
        "\n".join(row["sample_token"] for row in audit_rows) + ("\n" if audit_rows else "")
    )
    summary = {
        "selected_assets": args.selected_assets,
        "num_audit_cases": len(audit_rows),
        "num_scenes": len({row["scene_name"] for row in audit_rows}),
        "out_dir": str(out_dir),
        "cases_csv": str(out_dir / "phase1_8b_visual_audit_cases.csv"),
        "review_template_csv": str(out_dir / "phase1_8b_visual_audit_review_template.csv"),
        "contact_sheet": str(out_dir / "phase1_8b_visual_audit_contact_sheet.jpg"),
    }
    (out_dir / "phase1_8b_visual_audit_summary.json").write_text(__import__("json").dumps(summary, indent=2))
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
