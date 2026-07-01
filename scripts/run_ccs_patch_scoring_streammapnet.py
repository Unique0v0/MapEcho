#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

ROOT = Path("/home/dj/MapEcho")
STREAMMAPNET_ROOT = ROOT / "src" / "StreamMapNet"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(STREAMMAPNET_ROOT))

from build_ccs_dense_location_candidates import load_boundaries, write_csv
from ccs_patch_utils import (
    CAMERA_ORDER,
    apply_patch,
    create_pseudo_area,
    denormalize_img,
    denormalize_streammapnet_lines,
    find_best_matching_boundary_torch,
    get_patch_heading_facing_ego,
    get_phy_patch_mask,
    get_proj_scale,
    get_target_boundary_pts,
    init_patch_mask,
    restore_stream_memory,
    sample_to_global2img,
    sample_to_lidar2global,
    snapshot_stream_memory,
    torch_chamfer_distance,
)
from run_ccs_location_scoring_fast import build_model_once, make_data_loader


IMG_NORM_CFG = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def unwrap_batch(data, device):
    img = data["img"].data[0].to(device)
    img_metas = data["img_metas"].data[0]
    return img, img_metas


def forward_preds(model, img, img_metas):
    feats = model.backbone(img, img_metas, points=None)
    if model.streaming_bev:
        model.bev_memory.eval()
        feats = model.update_bev_feature(feats, img_metas)
    bev_feats = model.neck(feats)
    preds_list = model.head(bev_feats, img_metas=img_metas, return_loss=False)
    return preds_list[-1]


def post_process_for_debug(model, preds_dict, img_metas):
    tokens = [meta["token"] for meta in img_metas]
    return model.head.post_process(preds_dict, tokens)


def patch_rsa_loss(preds_dict, diverge_boundary, target_boundary, roi_size, score_thr):
    lines = preds_dict["lines"][0].view(-1, model_num_points(preds_dict), 2)
    lines = denormalize_streammapnet_lines(lines, roi_size=roi_size)
    scores = preds_dict["scores"][0]
    raw_scores, labels = scores.max(dim=-1)
    score_values = raw_scores.sigmoid()
    keep = (score_values > score_thr) & (labels == 2)
    candidates = lines[keep]
    pred_boundary = find_best_matching_boundary_torch(candidates, diverge_boundary)
    if pred_boundary is None:
        return torch.tensor(20.0, dtype=lines.dtype, device=lines.device), {
            "num_boundary_candidates": int(keep.sum().detach().cpu()),
            "status": "no_boundary",
        }
    if pred_boundary[:, 1].max().detach().item() > 0:
        pred_boundary = pred_boundary[pred_boundary[:, 1] > 0]
    if pred_boundary.shape[0] < 2:
        return torch.tensor(20.0, dtype=lines.dtype, device=lines.device), {
            "num_boundary_candidates": int(keep.sum().detach().cpu()),
            "status": "short_boundary",
        }
    loss = torch_chamfer_distance(pred_boundary, target_boundary)
    return loss, {
        "num_boundary_candidates": int(keep.sum().detach().cpu()),
        "status": "ok",
    }


def model_num_points(preds_dict):
    num_points2 = preds_dict["lines"][0].shape[1]
    return num_points2 // 2


def run_clean_warmup_to_target(cfg, model, clean_ann, target_token, device):
    dataset, loader = make_data_loader(cfg, clean_ann)
    module = model.module
    if hasattr(module, "reset_temporal_state"):
        module.reset_temporal_state("all")

    target_data = None
    with torch.no_grad():
        for data in loader:
            img, img_metas = unwrap_batch(data, device)
            token = img_metas[0]["token"]
            if token == target_token:
                target_data = copy.deepcopy(data)
                break
            forward_preds(module, img, img_metas)

    if target_data is None:
        raise KeyError(f"target token {target_token} not found in {clean_ann}")

    memory_snapshot = snapshot_stream_memory(module)
    return target_data, memory_snapshot, len(dataset)


def build_patch_cfg(row, patch_width, patch_height):
    heading = row.get("patch_heading", "")
    heading = float(heading) if heading != "" else get_patch_heading_facing_ego([float(row["x"]), float(row["y"])])
    return {
        "type": row.get("patch_type", "vertical"),
        "lat": float(row["x"]),
        "long": float(row["y"]),
        "width": patch_width,
        "height": patch_height,
        "heading": heading,
        "lidar2vehfront": 0.94,
        "lidar2ground": 1.84,
    }


def apply_single_patch_to_img(img, sample, patch, mask, patch_cfg, pseudo_area, img_norm_cfg):
    lidar2global = sample_to_lidar2global(sample)
    global2img = sample_to_global2img(sample)
    raw_img = cv2.imread(sample["cams"][CAMERA_ORDER[0]]["img_fpath"], cv2.IMREAD_COLOR)
    if raw_img is None:
        raise FileNotFoundError(sample["cams"][CAMERA_ORDER[0]]["img_fpath"])
    raw_h, raw_w = raw_img.shape[:2]
    model_h, model_w = int(img.shape[-2]), int(img.shape[-1])
    scale_x = model_w / float(raw_w)
    scale_y = model_h / float(raw_h)
    global2img = np.asarray(global2img, dtype=np.float64).copy()
    global2img[:, 0, :] *= scale_x
    global2img[:, 1, :] *= scale_y
    patch_trans_list, mask_trans_list, visible_cam_indices = get_phy_patch_mask(
        patch,
        mask,
        patch_cfg,
        pseudo_area,
        lidar2global,
        global2img,
        (model_h, model_w),
    )
    if visible_cam_indices is None:
        return img, []
    return apply_patch(img, patch_trans_list, mask_trans_list, img_norm_cfg, visible_cam_indices), visible_cam_indices


def save_patched_images_and_ann(clean_ann, target_token, best, out_dir):
    samples = [copy.deepcopy(sample) for sample in load_pickle(clean_ann)]
    target = next(sample for sample in samples if sample["token"] == target_token)
    images_dir = out_dir / "best_patch_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    raw = denormalize_img(best["patched_img"].detach().cpu(), IMG_NORM_CFG)
    raw = raw.clamp(0, 255)[0]
    for cam_idx, cam_name in enumerate(CAMERA_ORDER):
        img = raw[cam_idx].permute(1, 2, 0).numpy().astype(np.uint8)
        path = images_dir / f"{target_token}_{cam_name}.png"
        cv2.imwrite(str(path), img)
        target["cams"][cam_name]["img_fpath"] = str(path)

    target["mapecho_patch_objective"] = best["objective"]
    target["mapecho_patch_loss"] = float(best["loss"])
    target["mapecho_patch_rank"] = int(best["candidate"]["rank"])
    target["mapecho_patch_visible_cameras"] = [CAMERA_ORDER[i] for i in best["visible_cam_indices"]]

    out_ann = out_dir / "anns" / f"patch_{best['objective']}_sequence_ann.pkl"
    dump_pickle(samples, out_ann)
    return out_ann


def main():
    parser = argparse.ArgumentParser(
        description="Optimize CCS physical-patch candidates on StreamMapNet for one target token."
    )
    parser.add_argument("--target-token", required=True)
    parser.add_argument("--clean-ann", required=True)
    parser.add_argument("--patch-candidates-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--objective", choices=["rsa", "eta"], default="rsa")
    parser.add_argument("--config", default="/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py")
    parser.add_argument("--checkpoint", default="/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth")
    parser.add_argument("--max-locations", type=int, default=2)
    parser.add_argument("--patch-steps", type=int, default=2)
    parser.add_argument("--patch-width", type=float, default=3.0)
    parser.add_argument("--patch-height", type=float, default=2.0)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--roi-size", default="60,30")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.objective == "eta":
        raise RuntimeError(
            "Strict patch_eta requires CCS diverging-route centerline JSON for newsplit tokens. "
            "Run patch_rsa first, or port centerline generation before ETA."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(STREAMMAPNET_ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(STREAMMAPNET_ROOT))

    out_root = Path(args.out_root) / args.target_token
    out_root.mkdir(parents=True, exist_ok=True)
    roi_size = tuple(float(item.strip()) for item in args.roi_size.split(","))

    clean_samples = load_pickle(args.clean_ann)
    target_sample = next(sample for sample in clean_samples if sample["token"] == args.target_token)
    diverge_tag, diverge_np, reference_np = load_boundaries(
        next(row["scene_json"] for row in read_csv(args.patch_candidates_csv) if row["sample_token"] == args.target_token)
    )
    target_np = get_target_boundary_pts(diverge_np, reference_np, diverge_tag, "asymmetric", step=5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_args = argparse.Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        save_debug=False,
    )
    cfg, model = build_model_once(model_args)
    module = model.module
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)

    target_data, memory_snapshot, num_frames = run_clean_warmup_to_target(
        cfg, model, args.clean_ann, args.target_token, device
    )
    target_img, target_img_metas = unwrap_batch(target_data, device)

    diverge_boundary = torch.tensor(diverge_np, dtype=target_img.dtype, device=device)
    target_boundary = torch.tensor(target_np, dtype=target_img.dtype, device=device)

    restore_stream_memory(module, memory_snapshot)
    with torch.no_grad():
        clean_preds = forward_preds(module, target_img, target_img_metas)
        clean_loss, clean_info = patch_rsa_loss(
            clean_preds, diverge_boundary, target_boundary, roi_size, args.score_thr
        )

    candidates = [
        row for row in read_csv(args.patch_candidates_csv)
        if row["sample_token"] == args.target_token
    ]
    candidates.sort(key=lambda row: int(row["rank"]))
    candidates = candidates[: args.max_locations]
    if not candidates:
        raise ValueError(f"no patch candidates for {args.target_token}")

    _, _, _, model_h, model_w = target_img.shape
    ori_img_shape = (3, model_h, model_w)

    best = None
    score_rows = []
    for row in candidates:
        patch_cfg = build_patch_cfg(row, args.patch_width, args.patch_height)
        proj_scale = get_proj_scale(patch_cfg["lat"], patch_cfg["long"], model_w)
        pseudo_area = create_pseudo_area(patch_cfg, ori_img_shape, proj_scale)
        if pseudo_area is None:
            continue
        patch, mask = init_patch_mask(ori_img_shape, device, pseudo_area, mode="random")
        optimizer = optim.Adam([patch], lr=255.0 / args.patch_steps, betas=(0.5, 0.9))
        scheduler = StepLR(optimizer, 10, gamma=0.9)

        loc_best = None
        for step in range(args.patch_steps):
            optimizer.zero_grad()
            patch.data.clamp_(0, 255)
            imgs_adv = target_img.clone().detach()
            imgs_adv, visible = apply_single_patch_to_img(
                imgs_adv,
                target_sample,
                patch,
                mask,
                patch_cfg,
                pseudo_area,
                IMG_NORM_CFG,
            )
            if not visible:
                continue
            restore_stream_memory(module, memory_snapshot)
            preds = forward_preds(module, imgs_adv, target_img_metas)
            loss, info = patch_rsa_loss(preds, diverge_boundary, target_boundary, roi_size, args.score_thr)
            if loc_best is None or loss.detach().item() < loc_best["loss"]:
                loc_best = {
                    "loss": float(loss.detach().item()),
                    "step": step + 1,
                    "patch": patch.detach().clone().cpu(),
                    "mask": mask.detach().clone().cpu(),
                    "patched_img": imgs_adv.detach().clone().cpu(),
                    "visible_cam_indices": list(visible),
                    "info": dict(info),
                }
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                scheduler.step()

        if loc_best is None:
            continue
        candidate_summary = {
            "sample_token": args.target_token,
            "rank": row["rank"],
            "loss": loc_best["loss"],
            "clean_loss": float(clean_loss.detach().item()),
            "loss_delta_vs_clean": loc_best["loss"] - float(clean_loss.detach().item()),
            "best_step": loc_best["step"],
            "visible_cameras": " ".join(CAMERA_ORDER[i] for i in loc_best["visible_cam_indices"]),
            "status": loc_best["info"].get("status", ""),
            "num_boundary_candidates": loc_best["info"].get("num_boundary_candidates", ""),
        }
        score_rows.append(candidate_summary)
        write_csv(out_root / "patch_candidate_scores.csv", score_rows)

        if best is None or loc_best["loss"] < best["loss"]:
            best = {
                "objective": args.objective,
                "loss": loc_best["loss"],
                "clean_loss": float(clean_loss.detach().item()),
                "candidate": dict(row),
                "patch_cfg": patch_cfg,
                "pseudo_area": pseudo_area,
                "patch": loc_best["patch"],
                "mask": loc_best["mask"],
                "patched_img": loc_best["patched_img"],
                "visible_cam_indices": loc_best["visible_cam_indices"],
                "info": loc_best["info"],
            }

    if best is None:
        raise RuntimeError(f"no visible/valid patch candidate for {args.target_token}")

    best_pkl = out_root / f"best_patch_{args.objective}.pkl"
    dump_pickle(
        {
            "target_token": args.target_token,
            "objective": args.objective,
            "best_loss": best["loss"],
            "clean_loss": best["clean_loss"],
            "candidate": best["candidate"],
            "patch_cfg": best["patch_cfg"],
            "pseudo_area": best["pseudo_area"],
            "patch": best["patch"],
            "mask": best["mask"],
            "visible_cameras": [CAMERA_ORDER[i] for i in best["visible_cam_indices"]],
        },
        best_pkl,
    )
    out_ann = save_patched_images_and_ann(args.clean_ann, args.target_token, best, out_root)

    best_asset = {
        "sample_token": args.target_token,
        "scene_name": best["candidate"].get("scene_name", ""),
        "scene_pos": best["candidate"].get("scene_pos", ""),
        "has_patch_rsa_asset": True,
        "patch_rsa_best_pkl": str(best_pkl),
        "patch_rsa_sequence_ann": str(out_ann),
        "patch_rsa_loss": best["loss"],
        "patch_rsa_clean_loss": best["clean_loss"],
        "patch_rsa_rank": best["candidate"]["rank"],
        "patch_rsa_x": best["candidate"]["x"],
        "patch_rsa_y": best["candidate"]["y"],
        "patch_rsa_z": best["candidate"]["z"],
        "patch_rsa_heading": best["patch_cfg"]["heading"],
        "patch_rsa_visible_cameras": " ".join(CAMERA_ORDER[i] for i in best["visible_cam_indices"]),
        "mapecho_patch_method": "ccs_patch_streammapnet_optimizer",
        "num_frames_in_clean_sequence": num_frames,
        "num_locations_optimized": len(candidates),
        "patch_steps": args.patch_steps,
    }
    write_csv(out_root / "ccs_patch_best_asset_rsa.csv", [best_asset])

    summary = {
        "target_token": args.target_token,
        "objective": args.objective,
        "out_root": str(out_root),
        "clean_ann": args.clean_ann,
        "patch_candidates_csv": args.patch_candidates_csv,
        "num_locations_optimized": len(candidates),
        "patch_steps": args.patch_steps,
        "clean_loss": float(clean_loss.detach().item()),
        "clean_loss_status": clean_info.get("status"),
        "best_loss": best["loss"],
        "best_rank": best["candidate"]["rank"],
        "best_visible_cameras": [CAMERA_ORDER[i] for i in best["visible_cam_indices"]],
        "best_patch_pkl": str(best_pkl),
        "patch_sequence_ann": str(out_ann),
        "scores_csv": str(out_root / "patch_candidate_scores.csv"),
        "best_asset_csv": str(out_root / "ccs_patch_best_asset_rsa.csv"),
    }
    (out_root / "patch_scoring_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
