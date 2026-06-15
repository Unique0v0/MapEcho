#!/usr/bin/env python3
"""Assemble Phase 1.10 qualitative figures from existing selected114 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_9_paper_evidence/cases/qualitative_case_selection.csv"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "top400_selected114_controlled_check"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/data/dj/MapEcho/artifacts/phase1_8b_downstream/"
            "phase1_10_qualitative_figures"
        ),
    )
    parser.add_argument("--max-cases", type=int, default=15)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int = 14):
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_thumb(path: Path | None, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (248, 248, 248))
    if path is None or not path.exists():
        return canvas
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(20, 20, 20)) -> None:
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=4)


def case_slug(row: dict[str, str]) -> str:
    return (
        f"{row.get('case_group', 'case')}_"
        f"{int(float(row.get('case_rank') or 0)):02d}_"
        f"{row.get('target_token', row.get('sample_token', 'unknown'))[:8]}"
    )


def resolve_paths(run_root: Path, token: str) -> dict[str, Path | None]:
    base = run_root / token
    attack_dir = base / "attack_assets" / "images" / token
    map_dir = base / "phase1_0_map_level" / "figures"
    paths: dict[str, Path | None] = {
        "six_camera_contact_sheet": attack_dir / "six_camera_attacked_contact_sheet.png",
        "map_overlay_t0": map_dir / "offset_+0_boundary_overlay.png",
        "map_overlay_t1": map_dir / "offset_+1_boundary_overlay.png",
        "map_overlay_t2": map_dir / "offset_+2_boundary_overlay.png",
    }
    for cam in CAMERA_ORDER:
        paths[f"{cam}_overlay"] = attack_dir / f"{cam}_overlay.png"
        paths[f"{cam}_attacked"] = attack_dir / f"{cam}_attacked.png"
    return {key: path if path and path.exists() else None for key, path in paths.items()}


def copy_assets(row: dict[str, str], paths: dict[str, Path | None], out_case_dir: Path) -> dict[str, str]:
    out_case_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, src in paths.items():
        if src is None:
            copied[key] = ""
            continue
        dst = out_case_dir / src.name
        shutil.copy2(src, dst)
        copied[key] = str(dst)
    return copied


def make_case_figure(row: dict[str, str], paths: dict[str, Path | None], out_path: Path) -> None:
    title_font = load_font(18)
    font = load_font(14)
    small = load_font(12)
    width = 1600
    pad = 24
    title_h = 112
    camera_h = 300
    map_h = 320
    gap = 18
    height = title_h + camera_h + gap + 3 * map_h + 3 * gap + 24

    fig = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(fig)

    token = row.get("target_token", row.get("sample_token", ""))
    title = (
        f"{row.get('case_group')} #{row.get('case_rank')} | "
        f"{token[:8]} | {row.get('scene_name', '')}"
    )
    metrics = (
        f"keep t+1 Delta CD: {float(row.get('delta_cd_to_diverge_m') or 0):+.4f} m   "
        f"reset-BEV t+1: {float(row.get('reset_bev_t1_delta_cd') or 0):+.4f} m   "
        f"rank: {row.get('ccs_dense_rank', '')}"
    )
    draw_text(draw, (pad, 18), title, title_font)
    draw_text(draw, (pad, 52), metrics, font)
    draw_text(draw, (pad, 78), row.get("selection_reason", ""), small)

    y = title_h
    draw_text(draw, (pad, y), "Six-camera target-frame input", font)
    six = make_thumb(paths.get("six_camera_contact_sheet"), (width - 2 * pad, camera_h - 28))
    fig.paste(six, (pad, y + 24))

    for label, key in [
        ("Map overlay at t", "map_overlay_t0"),
        ("Map overlay at t+1", "map_overlay_t1"),
        ("Map overlay at t+2", "map_overlay_t2"),
    ]:
        y += camera_h + gap if key == "map_overlay_t0" else map_h + gap
        draw_text(draw, (pad, y), label, font)
        thumb = make_thumb(paths.get(key), (width - 2 * pad, map_h - 28))
        fig.paste(thumb, (pad, y + 24))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out_path)


def make_contact_sheet(rows: list[dict[str, str]], out_path: Path) -> None:
    font = load_font(13)
    title_font = load_font(18)
    thumb_size = (380, 160)
    label_w = 310
    header_h = 72
    row_h = thumb_size[1] + 36
    cols = [
        ("six-camera", "six_camera_contact_sheet"),
        ("map t+1", "map_overlay_t1"),
        ("map t+2", "map_overlay_t2"),
    ]
    width = label_w + len(cols) * thumb_size[0]
    height = header_h + max(1, len(rows)) * row_h
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw_text(draw, (14, 14), "Phase 1.10 qualitative case contact sheet", title_font)
    for idx, (label, _) in enumerate(cols):
        draw_text(draw, (label_w + idx * thumb_size[0] + 8, 48), label, font)
    for r, row in enumerate(rows):
        y = header_h + r * row_h
        token = row.get("target_token", row.get("sample_token", ""))
        label = (
            f"{row.get('case_group')} #{row.get('case_rank')}\n"
            f"{token[:8]} {row.get('scene_name', '')}\n"
            f"keep t+1={float(row.get('delta_cd_to_diverge_m') or 0):+.4f}m "
            f"resetBEV={float(row.get('reset_bev_t1_delta_cd') or 0):+.4f}m"
        )
        draw_text(draw, (12, y + 10), label, font)
        paths = row["_resolved_paths"]
        for c, (_, key) in enumerate(cols):
            thumb = make_thumb(paths.get(key), thumb_size)
            sheet.paste(thumb, (label_w + c * thumb_size[0], y))
        draw.line((0, y + row_h - 1, width, y + row_h - 1), fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_markdown_index(rows: list[dict[str, str]], out_path: Path) -> None:
    lines = [
        "# Phase 1.10 Qualitative Figure Index",
        "",
        "| Group | Rank | Token | Scene | keep t+1 Delta CD | reset-BEV t+1 | Figure |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        token = row.get("target_token", row.get("sample_token", ""))
        figure = Path(row["case_figure"]).name
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("case_group", ""),
                    row.get("case_rank", ""),
                    token,
                    row.get("scene_name", ""),
                    f"{float(row.get('delta_cd_to_diverge_m') or 0):+.4f} m",
                    f"{float(row.get('reset_bev_t1_delta_cd') or 0):+.4f} m",
                    f"`figures/{figure}`",
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    figures_dir = args.out_dir / "figures"
    case_assets_dir = args.out_dir / "case_assets"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    case_assets_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.case_csv)[: args.max_cases]
    manifest_rows: list[dict[str, str]] = []
    sheet_rows: list[dict[str, str]] = []

    for row in rows:
        token = row.get("target_token", row.get("sample_token", ""))
        slug = case_slug(row)
        paths = resolve_paths(args.run_root, token)
        copied = copy_assets(row, paths, case_assets_dir / slug)
        figure_path = figures_dir / f"{slug}_qualitative_panel.png"
        make_case_figure(row, paths, figure_path)

        out_row = dict(row)
        out_row.update(copied)
        out_row["case_slug"] = slug
        out_row["case_figure"] = str(figure_path)
        out_row["has_six_camera_contact_sheet"] = str(paths["six_camera_contact_sheet"] is not None)
        out_row["has_map_t0"] = str(paths["map_overlay_t0"] is not None)
        out_row["has_map_t1"] = str(paths["map_overlay_t1"] is not None)
        out_row["has_map_t2"] = str(paths["map_overlay_t2"] is not None)
        manifest_rows.append(out_row)

        sheet_row = dict(row)
        sheet_row["_resolved_paths"] = paths
        sheet_rows.append(sheet_row)

    write_csv(args.out_dir / "phase1_10_qualitative_figure_manifest.csv", manifest_rows)
    write_markdown_index(manifest_rows, args.out_dir / "phase1_10_qualitative_figure_index.md")
    make_contact_sheet(sheet_rows, args.out_dir / "phase1_10_qualitative_contact_sheet.png")

    summary = {
        "case_csv": str(args.case_csv),
        "run_root": str(args.run_root),
        "out_dir": str(args.out_dir),
        "num_cases": len(manifest_rows),
        "num_with_six_camera": sum(row["has_six_camera_contact_sheet"] == "True" for row in manifest_rows),
        "num_with_map_t0": sum(row["has_map_t0"] == "True" for row in manifest_rows),
        "num_with_map_t1": sum(row["has_map_t1"] == "True" for row in manifest_rows),
        "num_with_map_t2": sum(row["has_map_t2"] == "True" for row in manifest_rows),
        "contact_sheet": str(args.out_dir / "phase1_10_qualitative_contact_sheet.png"),
        "index_md": str(args.out_dir / "phase1_10_qualitative_figure_index.md"),
        "manifest_csv": str(args.out_dir / "phase1_10_qualitative_figure_manifest.csv"),
    }
    (args.out_dir / "phase1_10_qualitative_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[MapEcho] Phase 1.10 qualitative figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
