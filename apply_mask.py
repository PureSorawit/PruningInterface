import json
import argparse
import torch

from process_models import load_model_timm

COMPONENTS = ["ffn", "heads", "qkv_dim"]

MODEL_TYPE = "B/16"
DATASET_NAME = "cifar100"
TOPK_CHECKPOINT = 8
VERBOSE = True      # for logging


def is_packed_component(comp_dict):
    if not comp_dict:
        return True
    sample_val = next(iter(comp_dict.values()))
    return isinstance(sample_val, list)


def pack_component(flat_comp):
    per_layer = {}
    for key, v in flat_comp.items():
        if ":" not in key:
            continue
        l_str, i_str = key.split(":")
        l = int(l_str); i = int(i_str)
        per_layer.setdefault(l, {})[i] = int(v)

    packed = {}
    for l, idx_map in per_layer.items():
        max_idx = max(idx_map.keys())
        arr = [1] * (max_idx + 1)
        for i, v in idx_map.items():
            arr[i] = v
        packed[str(l)] = arr
    return packed


def normalize_masks(masks):
    norm = {}
    for comp in COMPONENTS:
        raw = masks.get(comp, {})
        if is_packed_component(raw):
            norm[comp] = {str(k): v for k, v in raw.items()}
        else:
            norm[comp] = pack_component(raw)
    return norm


# ---------------------------------------------------------
# compute pruning ratios
# ---------------------------------------------------------
def compute_pruned_stats(masks):
    def comp_stats(name):
        total = 0; pruned = 0
        for arr in masks[name].values():
            total += len(arr)
            pruned += sum(1 for x in arr if x == 0)
        return total, pruned

    ffn_t, ffn_p = comp_stats("ffn")
    hd_t, hd_p = comp_stats("heads")
    qd_t, qd_p = comp_stats("qkv_dim")

    all_t = ffn_t + hd_t + qd_t
    all_p = ffn_p + hd_p + qd_p

    pct = lambda p, t: 100 * p / t if t else 0.0

    print("[Stats] Mask-based pruning ratios:")
    print(f"  FFN units : {ffn_p}/{ffn_t}  ({pct(ffn_p, ffn_t):.2f}%)")
    print(f"  Heads     : {hd_p}/{hd_t}  ({pct(hd_p, hd_t):.2f}%)")
    print(f"  QKV dims  : {qd_p}/{qd_t}  ({pct(qd_p, qd_t):.2f}%)")
    print(f"  Overall   : {all_p}/{all_t}  ({pct(all_p, all_t):.2f}%)")


# ---------------------------------------------------------
# Apply the masks (soft pruning)
# ---------------------------------------------------------
@torch.no_grad()
def apply_masks(model, masks):
    for l, blk in enumerate(model.blocks):
        # FFN
        if str(l) in masks["ffn"]:
            arr = masks["ffn"][str(l)]
            dead = torch.tensor(arr, device=blk.mlp.fc1.weight.device) == 0
            dead_idx = dead.nonzero(as_tuple=True)[0]

            fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
            if len(dead_idx) > 0:
                fc1.weight[dead_idx, :] = 0
                if fc1.bias is not None:
                    fc1.bias[dead_idx] = 0
                fc2.weight[:, dead_idx] = 0

        # Heads
        if str(l) in masks["heads"]:
            arr = masks["heads"][str(l)]
            attn = blk.attn
            hm = torch.tensor(arr, device=attn.qkv.weight.device)

            nh = attn.num_heads
            D = attn.qkv.weight.shape[1]
            hd = D // nh

            qkv = attn.qkv
            proj = attn.proj

            for h in range(min(nh, len(hm))):
                if hm[h] == 0:
                    cols = slice(h * hd, (h + 1) * hd)
                    qkv.weight[cols, :] = 0
                    qkv.weight[D + cols.start:D + cols.stop, :] = 0
                    qkv.weight[2*D + cols.start:2*D + cols.stop, :] = 0
                    proj.weight[:, cols] = 0

        # QKV per-dim
        if str(l) in masks["qkv_dim"]:
            arr = masks["qkv_dim"][str(l)]
            attn = blk.attn
            dm = torch.tensor(arr, device=attn.qkv.weight.device)

            nh = attn.num_heads
            D = attn.qkv.weight.shape[1]
            hd = D // nh

            qkv = attn.qkv
            proj = attn.proj

            for d in range(min(hd, len(dm))):
                if dm[d] == 0:
                    for h in range(nh):
                        c = h * hd + d
                        qkv.weight[c, :] = 0
                        qkv.weight[D + c, :] = 0
                        qkv.weight[2*D + c, :] = 0
                        proj.weight[:, c] = 0

    print("[apply_masks] Finished zeroing masked weights.")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Apply pruning masks to a ViT model.")
    parser.add_argument("--mask", required=True, help="Mask JSON (flat or packed)")
    parser.add_argument("--save", required=True, help="Output .pth file")
    args = parser.parse_args()

    # 1) Load model
    print(f"[Load] model_type={MODEL_TYPE}, dataset={DATASET_NAME}, topk={TOPK_CHECKPOINT}")
    model = load_model_timm(
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        top10_idx=TOPK_CHECKPOINT,
        verbose=VERBOSE,
    )

    # 2) Load mask
    print(f"[Load] mask: {args.mask}")
    with open(args.mask, "r") as f:
        raw = json.load(f)
    masks = normalize_masks(raw)

    # 3) pruning ratios
    compute_pruned_stats(masks)

    # 4) Apply masks
    print("[Mask] Applying…")
    apply_masks(model, masks)

    # 5) Save pruned checkpoint
    torch.save(model.state_dict(), args.save)
    print(f"[Save] Done: {args.save}")


if __name__ == "__main__":
    main()
