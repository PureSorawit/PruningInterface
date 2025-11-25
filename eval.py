import time
import argparse

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from process_models import load_model_timm


MODEL_TYPE      = "B/16"
DATASET_NAME    = "cifar100"
TOPK_CHECKPOINT = 8


CKPT_OVERRIDE   = None  # for pruned model checkpoint evaluation

IMG_SIZE     = 224
BATCH_SIZE   = 128
SPEED_ITERS  = 100
SPEED_WARMUP = 10


def load_model(ckpt_override=None):
    print(f"[Load] model_type={MODEL_TYPE}, dataset={DATASET_NAME}, topk={TOPK_CHECKPOINT}")
    model = load_model_timm(
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        top10_idx=TOPK_CHECKPOINT,
        verbose=True,
    )

    if ckpt_override is not None:
        print(f"[Load] overriding weights from ckpt={ckpt_override}")
        sd = torch.load(ckpt_override, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  override: missing={len(missing)}, unexpected={len(unexpected)}")

    return model


def build_cifar100_testloader():
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def benchmark_speed(model, device):
    model.eval()
    x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)

    # Warmup
    for _ in range(SPEED_WARMUP):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(SPEED_ITERS):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.time()

    avg = (end - start) / SPEED_ITERS
    ms_per_batch = avg * 1000.0
    imgs_per_sec = BATCH_SIZE / avg
    return ms_per_batch, imgs_per_sec


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ViT model accuracy and speed on CIFAR-100."
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional .pth checkpoint to override base weights "
             "(e.g. pruned_model.pth). If omitted, uses the same "
             "checkpoint as HAS/load_model_timm."
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt if args.ckpt is not None else CKPT_OVERRIDE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = load_model(ckpt_override=ckpt_path)
    model.to(device)

    # Accuracy on CIFAR-100
    print("\n[Eval] CIFAR-100 test set")
    test_loader = build_cifar100_testloader()
    t0 = time.time()
    loss, acc = evaluate_accuracy(model, test_loader, device)
    dt = time.time() - t0
    print(f"  Test Loss : {loss:.4f}")
    print(f"  Test Acc  : {acc*100:.2f}%")
    print(f"  Time      : {dt:.1f} s")

    # Speed test on random data
    # print("\n[Speed] Random input benchmark")
    # ms, ips = benchmark_speed(model, device)
    # print(f"  {ms:.3f} ms / batch (batch_size={BATCH_SIZE})")
    # print(f"  {ips:.1f} images / second")


if __name__ == "__main__":
    main()
