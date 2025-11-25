import json
import argparse
import glob
import os

# -------------------------------------------------------
# USER CONFIG
# -------------------------------------------------------
INPUT_DIR = "input"   # folder containing score JSONs

# Output inside "output" folder
OUT_MASK = "output/final_intersect_mask.json"

INIT = 0.10
STEP = 0.05
ERROR_MARGIN = 0.01

COMPONENTS = ["ffn"]
# COMPONENTS = ["ffn", "heads", "qkv_dim"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def flatten(d):
    return list(d.items())


def mask_from_scores(score_dict, prune_frac):
    items = flatten(score_dict)
    n = len(items)
    k = int(round(prune_frac * n))

    items_sorted = sorted(items, key=lambda x: x[1])
    pruned_keys = {k for k, _ in items_sorted[:k]}

    return {k: (0 if k in pruned_keys else 1) for k, _ in items_sorted}


def intersect_many(mask_list):
    out = {c: {} for c in COMPONENTS}

    for comp in COMPONENTS:
        keys = set()
        for m in mask_list:
            keys.update(m[comp].keys())

        for k in keys:
            keep = 1
            for m in mask_list:
                keep = keep and m[comp].get(k, 1)
            out[comp][k] = 1 if keep else 0

    return out


def prune_ratio(mask):
    total = sum(len(mask[c]) for c in COMPONENTS)
    pruned = sum(
        sum(1 for v in mask[c].values() if v == 0)
        for c in COMPONENTS
    )
    return pruned / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True,
                        help="target prune ratio (0.0–1.0)")
    args = parser.parse_args()

    # -----------------------------------------------
    # Ensure output folder exists
    # -----------------------------------------------
    os.makedirs("output", exist_ok=True)

    # -----------------------------------------------
    # Load score files
    # -----------------------------------------------
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not paths:
        raise RuntimeError(f"No score JSON files found in folder: {INPUT_DIR}")

    print("Found score files:")
    for p in paths:
        print("  -", p)

    scores_list = [load_json(p) for p in paths]

    prune_frac = INIT
    final_mask = None

    # -----------------------------------------------
    # Iterative intersect search
    # -----------------------------------------------
    while prune_frac <= 1.0:
        mask_list = []
        for sc in scores_list:
            m = {comp: mask_from_scores(sc[comp], prune_frac)
                 for comp in COMPONENTS}
            mask_list.append(m)

        inter = intersect_many(mask_list)
        r = prune_ratio(inter)

        print(f"[Iter] prune_frac={prune_frac:.3f}  intersect_ratio={r:.3f}")

        if r >= args.target - ERROR_MARGIN:
            final_mask = inter
            break

        prune_frac += STEP

    if final_mask is None:
        final_mask = inter

    # -----------------------------------------------
    # Save to output folder
    # -----------------------------------------------
    with open(OUT_MASK, "w") as f:
        json.dump(final_mask, f, indent=2)

    print("\n[OK] Saved:", OUT_MASK)
    print("Final intersection prune ratio:",
          round(prune_ratio(final_mask) * 100, 2), "%")


if __name__ == "__main__":
    main()
