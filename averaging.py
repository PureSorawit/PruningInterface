import warnings
warnings.filterwarnings("ignore")

import json
import argparse
import os
import glob


INPUT_DIR = "input"

OUT_SCORES = "output/combined_scores.json"
OUT_MASKS  = "output/combined_masks.json"

COMPONENTS = ["ffn"]
# COMPONENTS = ["ffn", "heads", "qkv_dim"]


def load_scores(path):
    with open(path, "r") as f:
        return json.load(f)


def minmax_norm(d):
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) if hi > lo else 1.0
    return {k: (v - lo) / rng for k, v in d.items()}


def normalize_all(sc):
    return {c: minmax_norm(sc.get(c, {})) for c in COMPONENTS}


def union_keys(dicts):
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    return sorted(
        keys,
        key=lambda x: (
            int(x.split(":")[0]),
            int(x.split(":")[1])
        ) if ":" in x else (9999, 9999)
    )


def average_many(dict_list):
    if not dict_list:
        return {}

    keys = union_keys(dict_list)
    m = len(dict_list)

    out = {}
    for k in keys:
        out[k] = sum(d.get(k, 0.0) for d in dict_list) / m
    return out


def build_mask(scores, frac):
    n = len(scores)
    if n == 0 or frac <= 0:
        return {k: 1 for k in scores}

    prune_n = int(round(n * frac))
    ordered = sorted(scores.items(), key=lambda kv: kv[1])  # lowest first
    prune_keys = {k for k, _ in ordered[:prune_n]}

    return {k: (0 if k in prune_keys else 1) for k in scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", type=float, required=True,
                        help="fraction to prune (e.g., 0.30 for 30%)")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not paths:
        raise RuntimeError(f"No JSON files found in folder: {INPUT_DIR}")

    print(f"Found {len(paths)} score files:")
    for p in paths:
        print("  -", p)

    normalized_list = [normalize_all(load_scores(p)) for p in paths]

    combined_scores = {
        c: average_many([d[c] for d in normalized_list])
        for c in COMPONENTS
    }

    with open(OUT_SCORES, "w") as f:
        json.dump(combined_scores, f, indent=2)

    combined_masks = {
        c: build_mask(combined_scores[c], args.prune)
        for c in COMPONENTS
    }

    with open(OUT_MASKS, "w") as f:
        json.dump(combined_masks, f, indent=2)

    print("\n[OK] Wrote output:")
    print(" →", OUT_SCORES)
    print(" →", OUT_MASKS)


if __name__ == "__main__":
    main()
