#!/usr/bin/env python3
import argparse
import copy
import json
import pickle
from pathlib import Path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Build a small StreamMapNet annotation pkl for target-centered temporal sequences."
    )
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=19)
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument("--target-token", default="")
    args = parser.parse_args()

    samples = load_pickle(args.stream_ann)
    if args.target_token:
        target_tokens = [args.target_token]
    else:
        target_tokens = load_tokens(args.tokens)[: args.max_targets]

    by_token = {sample["token"]: sample for sample in samples}
    by_scene = {}
    for sample in samples:
        by_scene.setdefault(sample["scene_name"], []).append(sample)
    for scene_samples in by_scene.values():
        scene_samples.sort(key=lambda sample: sample["sample_idx"])

    subset = []
    sequence_rows = []
    for seq_id, token in enumerate(target_tokens):
        if token not in by_token:
            raise KeyError(f"target token not found in stream ann: {token}")
        target = by_token[token]
        scene_samples = by_scene[target["scene_name"]]
        target_pos = next(i for i, sample in enumerate(scene_samples) if sample["token"] == token)
        start = target_pos - args.warmup
        end = target_pos + args.recovery + 1
        if start < 0 or end > len(scene_samples):
            raise ValueError(
                f"target {token} is not W={args.warmup}/L={args.recovery} eligible"
            )
        seq_samples = [copy.deepcopy(sample) for sample in scene_samples[start:end]]
        for offset, sample in enumerate(seq_samples):
            frame_offset = offset - args.warmup
            sample["mapecho_sequence_id"] = seq_id
            sample["mapecho_target_token"] = token
            sample["mapecho_frame_offset"] = frame_offset
            sample["prev"] = -1 if offset == 0 else seq_samples[offset - 1]["token"]
            sample["next"] = -1 if offset == len(seq_samples) - 1 else seq_samples[offset + 1]["token"]
            subset.append(sample)
            sequence_rows.append(
                {
                    "sequence_id": seq_id,
                    "sample_token": sample["token"],
                    "target_token": token,
                    "scene_name": sample["scene_name"],
                    "sample_idx": sample["sample_idx"],
                    "frame_offset": frame_offset,
                    "is_target": frame_offset == 0,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(subset, f)

    summary = {
        "source_ann": str(Path(args.stream_ann).resolve()),
        "out": str(out_path),
        "warmup": args.warmup,
        "recovery": args.recovery,
        "target_tokens": target_tokens,
        "num_targets": len(target_tokens),
        "num_frames": len(subset),
        "sequences": sequence_rows,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "sequences"}, indent=2))


if __name__ == "__main__":
    main()
