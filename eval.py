import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from process_datasets import load_dataset, DSSplit, DSName
from process_models import load_model_timm, device as pm_device


MODEL_TYPE = "B/16"
DATASET_NAME = "cifar100"
TOP10_IDX = 8

BATCH_SIZE = 128
SUBSET_SIZE = 1.0
RESOLUTION = 224
USE_AMP_IF_CUDA = True


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)

    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


def load_model(ckpt_override: str | None = None) -> nn.Module:
    print(f"[Model] load_model_timm(model_type={MODEL_TYPE}, dataset={DATASET_NAME}, top10_idx={TOP10_IDX})")
    model = load_model_timm(
        MODEL_TYPE,
        DATASET_NAME,
        top10_idx=TOP10_IDX,
        verbose=True,
    )

    if ckpt_override is not None:
        print(f"[Model] overriding weights from ckpt={ckpt_override}")
        sd = torch.load(ckpt_override, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  override: missing={len(missing)}, unexpected={len(unexpected)}")

    return model


def build_val_loader():
    if DATASET_NAME == "cifar100":
        ds_enum = DSName.CIFAR100
    else:
        ds_enum = DSName.IIIT_PET

    train_loader, val_loader = load_dataset(
        dataset=ds_enum,
        batch_size=BATCH_SIZE,
        subset_size=SUBSET_SIZE,
        res=RESOLUTION,
        split=DSSplit.TrainVal,
        download_dataset=True,
    )
    return val_loader



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ViT accuracy"
    )
    parser.add_argument(
        "--ckpt",
        default=None
    )
    args = parser.parse_args()

    device = pm_device
    print(f"[Device] {device}")
    use_amp = (device.type == "cuda") and USE_AMP_IF_CUDA
    print(f"[AMP] enabled={use_amp}")

    # Model
    model = load_model(ckpt_override=args.ckpt)
    model.to(device)

    print("\n[Eval] CIFAR-100 validation")
    val_loader = build_val_loader()

    t0 = time.time()
    loss, acc = evaluate_accuracy(model, val_loader, device, use_amp=use_amp)
    dt = time.time() - t0

    print(f"  Val Loss : {loss:.4f}")
    print(f"  Val Acc  : {acc*100:.2f}%")
    print(f"  Time     : {dt:.1f} s")


if __name__ == "__main__":
    main()
