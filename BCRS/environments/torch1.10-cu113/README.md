# Unified reference environment

This is the least-change compatibility target for all three copied backends:

- Linux x86-64 and Python 3.9;
- PyTorch 1.10.1 / torchvision 0.11.2;
- CUDA 11.3 with a CUDA development toolkit (`nvcc` is required);
- MMDetection 2.24.1 / MMCV 1.5.1;
- Detectron2 0.6 and spconv 2.1.25.

Create and populate the environment from the BCRS project root:

```bash
conda create -n bcrs python=3.9 pip ninja -y
conda activate bcrs
python -m pip install -r environments/torch1.10-cu113/requirements.txt
python -m pip install -e '.[test]'
python -m pip install -v ./vendor/ceasc/Sparse_conv
```

The last command compiles CEASC's original CUDA extension against the selected
PyTorch/CUDA ABI. It is intentionally not hidden in the BCRS package install.
Set `BCRS_PYTHON` only when the CLI itself is running from another environment.

This target favors source fidelity. Moving to a newer PyTorch/CUDA release is a
separate compatibility project because CEASC includes private ATen/THC headers.
