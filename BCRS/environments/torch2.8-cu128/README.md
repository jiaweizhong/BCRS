# Modern Environment Specification (PyTorch 2.8+ / CUDA 12.8 / RTX 5090)

This directory defines the modern runtime environment for BCRS, configured to support modern GPU architectures (such as NVIDIA RTX 5090) on Python 3.12 with PyTorch 2.8+ and NumPy 2.x.

## Tested Stack Specifications

- **Python**: `3.12.x`
- **CUDA**: `12.8`
- **PyTorch**: `2.8.0+cu128`
- **Torchvision**: `0.23.0+cu128`
- **NumPy**: `2.2.3`
- **GPU Architecture**: NVIDIA Blackwell / Ada Lovelace (e.g. RTX 5090, 32GB VRAM)

## Source Code Modifications for PyTorch 2.8+ & NumPy 2.x

To enable seamless execution without deprecation crashes or pickle load failures:

1. **`torch.load` Unpickle Compatibility**:
   PyTorch 2.6+ changed `weights_only` default from `False` to `True`. `vendor/esod/utils/datasets.py` is monkeypatched to ensure `.cache` files load safely with `weights_only=False`.

2. **NumPy 2.0+ `np.trapz` Removal**:
   NumPy 2.0 removed `np.trapz`. `vendor/esod/utils/metrics.py` was updated to dynamically use `getattr(np, 'trapezoid', getattr(np, 'trapz', None))`.

3. **Label Clipping and Deduplication**:
   Raw VisDrone bounding boxes with boundary overshoots (`x + w / W > 1.0`) are clipped to `[0.0, 1.0]` and deduplicated in `src/bcrs/datasets/visdrone.py` and `vendor/esod/utils/datasets.py`.

## Installation

```bash
pip install -r environments/torch2.8-cu128/requirements.txt
```
