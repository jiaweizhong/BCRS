"""Fine-tunes a COCO-pretrained torchvision Faster R-CNN / RetinaNet on a
YOLO-format dataset already prepared for this project's other arms.

Resume-compatible with scripts/esod_baseline/run_torchvision_baseline.sh's
run_arm()-style loop: writes one line to results.txt per completed epoch
(the same awk 'NF{count++}' resume-detection convention as
run_uavdt.sh/run_seaperson.sh), and `--resume last.pt` alone (no other
flags) fully restores the original run config from the checkpoint, matching
exactly how those scripts invoke `python train.py --resume "$last_ckpt"`.

Usage (fresh run):
  python train.py --model fasterrcnn \
    --train-images-dir .../images/train --train-labels-dir .../labels/train \
    --val-images-dir .../images/test --val-labels-dir .../labels/test \
    --classes car,truck,bus --epochs 50 --batch-size 8 --img-size 1280 \
    --device 0 --project /root/esod_baseline_runs/train --name uavdt_fasterrcnn --exist-ok

Usage (resume):
  python train.py --resume /root/esod_baseline_runs/train/uavdt_fasterrcnn/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import YoloDetectionDataset, collate_fn  # noqa: E402
from models import MODEL_NAMES, build_model  # noqa: E402


def _device(spec: str) -> torch.device:
    if spec.lower() == "cpu":
        return torch.device("cpu")
    if spec.isdigit():
        return torch.device(f"cuda:{spec}")
    return torch.device(spec)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--train-images-dir", default=None)
    parser.add_argument("--train-labels-dir", default=None)
    parser.add_argument("--val-images-dir", default=None)
    parser.add_argument("--val-labels-dir", default=None)
    parser.add_argument(
        "--classes", default=None, help="comma-separated class names in id order"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--img-size", type=int, default=1280, help="long-side target; see models.py"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help=(
            "0.005 (the old default) was confirmed too high for fine-tuning "
            "from a COCO-pretrained checkpoint at small batch sizes: "
            "seaperson_fasterrcnn's first real run (2026-08-23, batch=2) "
            "diverged from epoch 0 onward (val_loss 0.696 -> 1.08 over 50 "
            "epochs, best.pt stuck at epoch 0), consistent with the LR being "
            "too aggressive for fine-tuning rather than from-scratch training. "
            "0.0005 is an order of magnitude lower, a standard fine-tuning "
            "magnitude at this batch size -- not independently verified "
            "optimal, just the first thing worth trying."
        ),
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="last.pt checkpoint; if given, all other flags are ignored",
    )
    return parser


def _validate_fresh_run(opt: argparse.Namespace) -> None:
    required = [
        "model",
        "train_images_dir",
        "train_labels_dir",
        "val_images_dir",
        "val_labels_dir",
        "classes",
        "project",
        "name",
    ]
    missing = [name for name in required if getattr(opt, name) is None]
    if missing:
        raise SystemExit(
            f"Missing required flag(s) for a fresh run: {', '.join('--' + m.replace('_', '-') for m in missing)}"
        )


def completed_epochs(train_dir: Path) -> int:
    results_file = train_dir / "results.txt"
    if not results_file.is_file():
        return 0
    return sum(
        1
        for line in results_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@torch.no_grad()
def _eval_loss(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    # torchvision detection models only return a loss dict in train() mode;
    # wrapping the forward pass in no_grad() + train() gives a val-loss proxy
    # without updating weights or requiring a second (eval-mode) code path.
    model.train()
    total, count = 0.0, 0
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
            for t in targets
        ]
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()))
        count += 1
    return total / count if count else float("inf")


def train(opt: argparse.Namespace) -> None:
    device = _device(opt.device)
    class_names = tuple(name.strip() for name in opt.classes.split(",") if name.strip())

    train_ds = YoloDetectionDataset(
        opt.train_images_dir, opt.train_labels_dir, class_names
    )
    val_ds = YoloDetectionDataset(opt.val_images_dir, opt.val_labels_dir, class_names)
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=collate_fn,
    )

    model = build_model(
        opt.model, len(class_names), min_size=opt.img_size, max_size=opt.img_size
    )
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    train_dir = Path(opt.project) / opt.name
    weights_dir = train_dir / "weights"
    if train_dir.exists() and not opt.exist_ok and not opt.resume:
        raise SystemExit(f"{train_dir} already exists; pass --exist-ok to reuse it")
    weights_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from {opt.resume} at epoch {start_epoch}/{opt.epochs}")

    results_path = train_dir / "results.txt"
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        running_loss, batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{opt.epochs - 1}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
                for t in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{running_loss / batches:.4f}")
        scheduler.step()
        train_loss = running_loss / batches if batches else float("nan")
        val_loss = _eval_loss(model, val_loader, device)
        print(
            f"epoch {epoch}/{opt.epochs - 1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch} {train_loss:.4f} {val_loss:.4f}\n")

        best_val_loss = min(best_val_loss, val_loss)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "opt": vars(opt),
            "config": {
                "model_name": opt.model,
                "class_names": class_names,
                "min_size": opt.img_size,
                "max_size": opt.img_size,
            },
        }
        torch.save(checkpoint, weights_dir / "last.pt")
        if val_loss <= best_val_loss:
            torch.save(checkpoint, weights_dir / "best.pt")

    print("Training complete.")


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        opt = argparse.Namespace(**ckpt["opt"])
        opt.resume = args.resume
    else:
        _validate_fresh_run(args)
        opt = args

    train(opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
