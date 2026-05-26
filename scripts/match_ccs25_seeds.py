#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokens(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Match CCS'25 asymmetric seed tokens to StreamMapNet oldsplit annotations."
    )
    parser.add_argument("--stream-ann", required=True)
    parser.add_argument("--ccs25-info", required=True)
    parser.add_argument("--ccs25-tokens", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=19)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_samples = load_pickle(args.stream_ann)
    ccs25_obj = load_pickle(args.ccs25_info)
    ccs25_infos = ccs25_obj["infos"] if isinstance(ccs25_obj, dict) else ccs25_obj
    ccs25_tokens = load_tokens(args.ccs25_tokens)

    stream_by_token = {sample["token"]: sample for sample in stream_samples}
    stream_order_by_token = {sample["token"]: i for i, sample in enumerate(stream_samples)}

    scene_samples = defaultdict(list)
    for i, sample in enumerate(stream_samples):
        scene_samples[sample["scene_name"]].append((i, sample))
    scene_pos_by_token = {}
    scene_len_by_name = {}
    for scene_name, samples in scene_samples.items():
        samples.sort(key=lambda item: item[1]["sample_idx"])
        scene_len_by_name[scene_name] = len(samples)
        for scene_pos, (_, sample) in enumerate(samples):
            scene_pos_by_token[sample["token"]] = scene_pos

    ccs25_info_tokens = [info["token"] for info in ccs25_infos]
    token_set_from_info = set(ccs25_info_tokens)
    token_set_from_txt = set(ccs25_tokens)

    rows = []
    for rank, token in enumerate(ccs25_tokens):
        sample = stream_by_token.get(token)
        matched = sample is not None
        scene_name = sample.get("scene_name") if matched else ""
        scene_pos = scene_pos_by_token[token] if matched else ""
        scene_len = scene_len_by_name[scene_name] if matched else ""
        global_idx = stream_order_by_token[token] if matched else ""
        sample_idx = sample.get("sample_idx") if matched else ""
        prev_token = sample.get("prev") if matched else ""
        next_token = sample.get("next") if matched else ""
        temporal_ok = False
        if matched:
            temporal_ok = scene_pos >= args.warmup and scene_pos + args.recovery < scene_len
        rows.append(
            {
                "seed_rank": rank,
                "token": token,
                "matched_streammapnet": matched,
                "present_in_ccs25_info": token in token_set_from_info,
                "stream_global_idx": global_idx,
                "scene_name": scene_name,
                "sample_idx": sample_idx,
                "scene_pos": scene_pos,
                "scene_len": scene_len,
                "prev": prev_token,
                "next": next_token,
                "temporal_eligible_W10_L19": temporal_ok,
            }
        )

    matched_rows = [row for row in rows if row["matched_streammapnet"]]
    temporal_rows = [row for row in matched_rows if row["temporal_eligible_W10_L19"]]

    summary = {
        "stream_ann": str(Path(args.stream_ann).resolve()),
        "ccs25_info": str(Path(args.ccs25_info).resolve()),
        "ccs25_tokens": str(Path(args.ccs25_tokens).resolve()),
        "warmup": args.warmup,
        "recovery": args.recovery,
        "stream_samples": len(stream_samples),
        "stream_scenes": len(scene_samples),
        "ccs25_tokens_txt": len(ccs25_tokens),
        "ccs25_infos": len(ccs25_infos),
        "ccs25_token_txt_minus_info": sorted(token_set_from_txt - token_set_from_info),
        "ccs25_token_info_minus_txt": sorted(token_set_from_info - token_set_from_txt),
        "matched": len(matched_rows),
        "missing": len(rows) - len(matched_rows),
        "temporal_eligible": len(temporal_rows),
        "unique_matched_scenes": len({row["scene_name"] for row in matched_rows}),
        "unique_temporal_eligible_scenes": len({row["scene_name"] for row in temporal_rows}),
    }

    csv_path = out_dir / "ccs25_seed_streammapnet_match.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "ccs25_seed_streammapnet_match_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (out_dir / "temporal_eligible_tokens_W10_L19.txt").write_text(
        "\n".join(row["token"] for row in temporal_rows) + ("\n" if temporal_rows else "")
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
