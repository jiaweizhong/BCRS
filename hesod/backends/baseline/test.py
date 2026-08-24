"""Evaluates a trained torchvision Faster R-CNN / RetinaNet checkpoint.

`--task test` writes `{name}_predictions.json` in the exact schema
scripts/esod_baseline/audit_buckets.py and vt_diagnose.py already consume
(list of {"image_id": <stem>, "category_id": 0-indexed int,
"bbox": [x,y,w,h] abs px, "score": float}) -- both audit tools then run
completely unmodified, giving true per-bucket recall comparability against
every other arm. It also runs pycocotools COCOeval (via coco_utils.py)
against a cached YOLO-derived COCO ground-truth json for standard
AP/AP50/AP75 -- note this is a DIFFERENT code path than the custom
ap_per_class() powering every other arm's mAP column (see
HESOD-Experiment-Plan.md SS9.3), not claimed to be numerically identical.

`--task measure` forces batch=1 and reports GFLOPs (via
fvcore.nn.FlopCountAnalysis, matching hesod/backends/hesod/test.py's own
measure task; thop is tried second but is confirmed broken on Python 3.12
-- see the comment in run_measure() -- falling back to param-count-only if
both fail) and wall-clock FPS.

Usage:
  python test.py --weights .../weights/best.pt \
    --images-dir .../images/test --labels-dir .../labels/test \
    --task test --batch-size 8 --device 0 --save-json \
    --project /root/esod_baseline_runs/test --name uavdt_fasterrcnn --exist-ok
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_utils import build_coco_gt, evaluate_coco  # noqa: E402
from datasets import YoloDetectionDataset, collate_fn  # noqa: E402
from models import build_model  # noqa: E402


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
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument(
        "--model", default=None, help="overrides the checkpoint's stored model name"
    )
    parser.add_argument(
        "--classes", default=None, help="overrides the checkpoint's stored class list"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="overrides the checkpoint's stored img-size",
    )
    parser.add_argument("--task", choices=("test", "measure"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def _load_model(weights_path: str, opt: argparse.Namespace, device: torch.device):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_name = opt.model or config["model_name"]
    class_names = (
        tuple(opt.classes.split(",")) if opt.classes else tuple(config["class_names"])
    )
    img_size = opt.img_size or config["min_size"]

    model = build_model(
        model_name, len(class_names), min_size=img_size, max_size=img_size
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, class_names, img_size


def prediction_dict(
    image_id: str, box_xyxy: tuple[float, float, float, float], label: int, score: float
) -> dict:
    """Builds one row of the ESOD-schema predictions.json this project's
    audit_buckets.py/vt_diagnose.py already consume unmodified:
    {"image_id","category_id" (0-indexed),"bbox":[x,y,w,h],"score"}.
    `label` is torchvision's 1-indexed class id (0 reserved for background).
    """
    x1, y1, x2, y2 = box_xyxy
    return {
        "image_id": image_id,
        "category_id": int(label) - 1,
        "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
        "score": round(float(score), 5),
    }


def _write_predictions(
    model,
    dataset: YoloDetectionDataset,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
) -> None:
    predictions: list[dict] = []
    seen = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="predict"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = dataset.image_id(int(target["image_id"]))
                boxes = output["boxes"].cpu().tolist()
                labels = output["labels"].cpu().tolist()
                scores = output["scores"].cpu().tolist()
                for box, label, score in zip(boxes, labels, scores):
                    predictions.append(
                        prediction_dict(image_id, tuple(box), label, score)
                    )
            seen += len(images)
    out_path.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"Saved {len(predictions)} predictions for {seen} images to {out_path}")


def run_test(opt: argparse.Namespace) -> None:
    device = _device(opt.device)
    model, class_names, img_size = _load_model(opt.weights, opt, device)

    dataset = YoloDetectionDataset(opt.images_dir, opt.labels_dir, class_names)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=collate_fn,
    )

    save_dir = Path(opt.project) / opt.name
    if save_dir.exists() and not opt.exist_ok:
        raise SystemExit(f"{save_dir} already exists; pass --exist-ok to reuse it")
    save_dir.mkdir(parents=True, exist_ok=True)

    weights_stem = Path(opt.weights).stem
    pred_path = save_dir / f"{weights_stem}_predictions.json"
    if not opt.save_json:
        raise SystemExit(
            "--task test requires --save-json (matches every other arm's convention)"
        )
    _write_predictions(model, dataset, loader, device, pred_path)

    cache_path = save_dir / "coco_gt.json"
    gt_path = build_coco_gt(dataset, cache_path)
    try:
        stats = evaluate_coco(gt_path, pred_path)
        print("COCOeval stats:", json.dumps(stats, indent=2))
        (save_dir / f"{weights_stem}_coco_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
    except (
        Exception
    ) as exc:  # pragma: no cover - diagnostic path, matches test.py's own try/except discipline
        print(f"pycocotools unable to run: \n{exc}")


def run_measure(opt: argparse.Namespace) -> None:
    device = _device(opt.device)
    opt.batch_size = 1
    model, class_names, img_size = _load_model(opt.weights, opt, device)

    dataset = YoloDetectionDataset(opt.images_dir, opt.labels_dir, class_names)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=collate_fn,
    )

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_images, _ = next(iter(loader))
    sample = [sample_images[0].to(device)]

    # fvcore first, matching hesod/backends/hesod/test.py's own measure-task
    # approach (already proven working in this project's environment).
    # `thop` (this repo's other tracked FLOP-counting dependency) is NOT a
    # safe fallback here: thop==0.1.1 imports the stdlib `distutils` module,
    # removed in Python 3.12 -- confirmed broken (ImportError) on this exact
    # Python version. Kept as a second attempt in case an older Python or a
    # patched thop is in use; final fallback is param-count-only.
    gflops = None
    try:
        from fvcore.nn import FlopCountAnalysis

        fca = FlopCountAnalysis(model, inputs=(sample,))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        gflops = fca.total() / 1e9 * 2
    except Exception as exc:
        print(f"fvcore profiling failed ({exc}); trying thop")
        try:
            from thop import profile

            flops, _params = profile(model, inputs=(sample,), verbose=False)
            gflops = flops / 1e9
        except Exception as exc2:
            print(f"thop profiling also failed, reporting param count only: {exc2}")

    params = sum(p.numel() for p in model.parameters())

    # Warm-up, then time inference-only wall-clock at batch=1.
    n = min(50, len(dataset))
    with torch.no_grad():
        for _ in range(3):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        count = 0
        for images, _targets in loader:
            images = [img.to(device) for img in images]
            model(images)
            count += 1
            if count >= n:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    fps = count / elapsed if elapsed > 0 else float("nan")
    result = {
        "gflops": gflops,
        "params": params,
        "fps": fps,
        "ms_per_image": 1000.0 / fps if fps else None,
    }
    print(f"GFLOPs: {gflops}. Params: {params}. FPS: {fps:.2f}")
    (save_dir / "measure.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    opt = parser.parse_args(argv)
    if opt.task == "test":
        run_test(opt)
    else:
        run_measure(opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
