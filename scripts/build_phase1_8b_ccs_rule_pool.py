#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

from PIL import Image


ORIG_REPO = "/home/dj/physical-online-map-attack"
if ORIG_REPO not in sys.path:
    sys.path.insert(0, ORIG_REPO)

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

from attack_toolkit.src.utils.utils_prompt import VectorizedLocalMap
import dataset_processing.config as ccs_config
import dataset_processing.rule_based_classifier as rule_module
from dataset_processing.rule_based_classifier import RuleBasedClassifier


MAPS = [
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


def normalize_infos(raw_infos, limit=0, max_per_scene=0):
    if isinstance(raw_infos, dict):
        infos = raw_infos["infos"]
    else:
        infos = raw_infos
    out = []
    scene_counts = {}
    for info in infos:
        scene = info.get("scene_name", "")
        if max_per_scene and max_per_scene > 0:
            count = scene_counts.get(scene, 0)
            if count >= max_per_scene:
                continue
            scene_counts[scene] = count + 1
        item = dict(info)
        if "ego2global_rotation" not in item and "e2g_rotation" in item:
            item["ego2global_rotation"] = item["e2g_rotation"]
        if "ego2global_translation" not in item and "e2g_translation" in item:
            item["ego2global_translation"] = item["e2g_translation"]
        if "map_location" not in item and "location" in item:
            item["map_location"] = item["location"]
        out.append(item)
        if limit and limit > 0 and len(out) >= limit:
            break
    return {"infos": out, "metadata": {"source": "streammapnet_newsplit_val"}}


def patch_output_dirs(out_dir):
    mapping = {
        "SCENES_CANDIDATE_DIR": out_dir / "scenes_candidate",
        "SCENES_ASYMMETRIC_DIST_DIR": out_dir / "scenes_asymmetric_dist",
        "SCENES_SYMMETRIC_DIST_DIR": out_dir / "scenes_symmetric_dist",
        "SCENES_ASYMMETRIC_CURVATURE_DIR": out_dir / "scenes_asymmetric_curvature",
        "SCENES_ASYMMETRIC_CURVATURE_INVALID_DIR": out_dir / "scenes_asymmetric_curvature_invalid",
        "SCENES_ASYMMETRIC_CURVATURE_SELECTED_DIR": out_dir / "scenes_asymmetric_curvature_selected",
        "SCENES_SYMMETRIC_CURVATURE_DIR": out_dir / "scenes_symmetric_curvature",
        "SCENES_SYMMETRIC_CURVATURE_SELECTED_DIR": out_dir / "scenes_symmetric_curvature_selected",
    }
    for name, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        setattr(ccs_config, name, str(path))
        setattr(rule_module, name, str(path))


def make_classifier(data_root):
    nusc = NuScenes(version="v1.0-trainval", dataroot=data_root, verbose=False)
    nusc_maps = {name: NuScenesMap(dataroot=data_root, map_name=name) for name in MAPS}
    map_explorer = {name: NuScenesMapExplorer(nusc_maps[name]) for name in MAPS}
    vector_map = VectorizedLocalMap(
        data_root,
        patch_size=ccs_config.PATCH_SIZE,
        map_classes=ccs_config.MAP_CLASSES,
        fixed_ptsnum_per_line=ccs_config.FIXED_PTSNUM_PER_LINE,
        padding_value=ccs_config.PADDING_VALUE,
    )
    car_img = Image.new("RGBA", (24, 30), (255, 0, 0, 128))
    return RuleBasedClassifier(nusc, nusc_maps, map_explorer, vector_map, car_img), nusc


def scene_name(nusc, token):
    sample = nusc.get("sample", token)
    scene = nusc.get("scene", sample["scene_token"])
    return scene["name"]


def summarize(tokens, nusc):
    return {
        "frames": len(tokens),
        "scenes": len({scene_name(nusc, token) for token in tokens}),
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild newsplit-val candidates with the original CCS rule-based pipeline.")
    parser.add_argument("--data-root", default="/data/yuy/dataset/nuScenes/full")
    parser.add_argument("--newsplit-val-ann", default="/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl")
    parser.add_argument("--out-dir", default="/data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit over newsplit-val infos; 0 means full val.")
    parser.add_argument("--max-per-scene-input", type=int, default=0, help="Debug cap before preprocessing; 0 means all frames.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-lane-width", action="store_true")
    parser.add_argument("--skip-curvature", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    patch_output_dirs(out_dir)
    classifier, nusc = make_classifier(args.data_root)

    raw_infos = load_pickle(args.newsplit_val_ann)
    infos = normalize_infos(raw_infos, args.limit, args.max_per_scene_input)

    candidate_tokens_path = out_dir / "sample_token_candidates.txt"
    if args.skip_preprocess:
        candidate_tokens = [line.strip() for line in candidate_tokens_path.read_text().splitlines() if line.strip()]
    else:
        candidate_tokens = classifier.preprocess_scenes(infos)
        write_tokens(candidate_tokens_path, candidate_tokens)

    if args.skip_lane_width:
        asym_dist_path = out_dir / "sample_tokens_asymmetric_dist.txt"
        sym_dist_path = out_dir / "sample_tokens_symmetric_dist.txt"
        asym_dist_tokens = [line.strip() for line in asym_dist_path.read_text().splitlines() if line.strip()]
        sym_dist_tokens = [line.strip() for line in sym_dist_path.read_text().splitlines() if line.strip()]
    else:
        asym_dist_tokens, sym_dist_tokens = classifier.classify_by_lane_width(candidate_tokens)
        write_tokens(out_dir / "sample_tokens_asymmetric_dist.txt", asym_dist_tokens)
        write_tokens(out_dir / "sample_tokens_symmetric_dist.txt", sym_dist_tokens)

    if args.skip_curvature:
        asym_curv_tokens = []
        sym_curv_tokens = []
        invalid_tokens = []
        diverge_points = {}
    else:
        asym_curv_tokens, sym_curv_tokens, invalid_tokens, diverge_points = classifier.classify_by_curvature(asym_dist_tokens)
        write_tokens(out_dir / "sample_tokens_asymmetric_curvature.txt", asym_curv_tokens)
        write_tokens(out_dir / "sample_tokens_symmetric_curvature.txt", sym_curv_tokens)
        write_tokens(out_dir / "sample_tokens_invalid_curvature.txt", invalid_tokens)
        (out_dir / "diverge_points.json").write_text(json.dumps(diverge_points, indent=2))

    rows = []
    for set_name, tokens in [
        ("candidates", candidate_tokens),
        ("asymmetric_dist", asym_dist_tokens),
        ("symmetric_dist", sym_dist_tokens),
        ("asymmetric_curvature", asym_curv_tokens),
        ("symmetric_curvature", sym_curv_tokens),
        ("invalid_curvature", invalid_tokens),
    ]:
        stats = summarize(tokens, nusc)
        rows.append({"set": set_name, **stats})
    write_csv(out_dir / "phase1_8b_ccs_rule_summary.csv", rows)

    by_scene = {}
    for token in asym_curv_tokens:
        name = scene_name(nusc, token)
        by_scene.setdefault(name, 0)
        by_scene[name] += 1
    write_csv(
        out_dir / "phase1_8b_asymmetric_curvature_by_scene.csv",
        [{"scene_name": k, "frames": v} for k, v in sorted(by_scene.items())],
    )

    report = {
        "data_root": args.data_root,
        "newsplit_val_ann": args.newsplit_val_ann,
        "limit": args.limit,
        "max_per_scene_input": args.max_per_scene_input,
        "out_dir": str(out_dir),
        "sets": rows,
        "outputs": {
            "summary_csv": str(out_dir / "phase1_8b_ccs_rule_summary.csv"),
            "asymmetric_curvature_tokens": str(out_dir / "sample_tokens_asymmetric_curvature.txt"),
            "asymmetric_curvature_by_scene": str(out_dir / "phase1_8b_asymmetric_curvature_by_scene.csv"),
        },
    }
    (out_dir / "phase1_8b_ccs_rule_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
