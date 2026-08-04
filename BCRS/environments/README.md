# BCRS Managed Environment Specifications

BCRS maintains two environment specifications to balance modern hardware support with historical baseline fidelity:

| Environment | Purpose | Target Hardware & OS | Key Stack |
| :--- | :--- | :--- | :--- |
| **`torch2.8-cu128`** *(Recommended)* | Modern Primary Environment | NVIDIA RTX 5090 / Blackwell GPUs (CUDA 12.8) | Python 3.12, PyTorch 2.8+, NumPy 2.2+ |
| **`torch1.10-cu113`** *(Legacy)* | Historical Baseline Reference | NVIDIA V100 / Ampere GPUs (CUDA 11.3) | Python 3.9, PyTorch 1.10.1, NumPy 1.23 |

## Recommended Setup (RTX 5090 / Modern Hardware)

```bash
pip install -r environments/torch2.8-cu128/requirements.txt
pip install -e .
```

## Legacy Reference Setup (CUDA 11.3 / V100)

```bash
pip install -r environments/torch1.10-cu113/requirements.txt
pip install -e '.[test]'
```
