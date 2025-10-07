# Sem7 AI/ML Honours Lab Codes

This repository contains practical experiments and code for the Semester 7 AI/ML Honours lab. It includes PyTorch and NumPy implementations for common experiments used in coursework and viva preparation.

Contents
- `expt1/` — CIFAR-10 PyTorch training experiment, supporting files, checkpoints and plots.
- `expt2/` — Alternate implementations (Keras, NumPy, PyTorch) and related notes.
- `NNDL_EXPT1_Pytorch.pdf`, `NNDL_EXPT2_LG_ANN.pdf` — Lab reports.

Quick start
1. Create and activate a Python 3.8+ virtual environment.
2. Install dependencies for experiment 1:

   pip install -r expt1/requirements.txt

3. To reproduce training (example):

   python expt1/train_cifar10.py

Notes
- The `expt1/` folder contains a `checkpoints/` directory with pre-saved models (`best.pth`, `checkpoint.pth`) and `history.npz` with training curves.
- Data: `expt1/data/cifar-10-python.tar.gz` contains the CIFAR-10 dataset; the code expects the `cifar-10-batches-py/` folder under `expt1/data/`.

Recommended experiments
- Train from scratch: run `train_cifar10.py` and monitor `checkpoints/` and `training_curves.png`.
- Quick evaluation: load `checkpoints/best.pth` in `train_cifar10.py` or a small inference script.

Project structure (high level)

```
expt1/
  ├─ train_cifar10.py         # main training script (PyTorch)
  ├─ requirements.txt         # Python dependencies for expt1
  ├─ checkpoints/             # model checkpoints and training history
  └─ data/                    # compressed dataset and extracted batches
expt2/
  ├─ expt2_torch.py
  ├─ expt2_keras.py
  └─ expt2_numpy_only.py
```

Coding conventions
- Code uses standard Python packaging and style. Please run linters (optional) and formatters (black) before submitting changes.

Where to look first
- If you're inspecting results, open `expt1/checkpoints/training_curves.png` and `expt1/output.txt`.

Contact / author
- Maintainer: Joseph Jonathan Fernandes

License and contributing
- See `LICENSE` and `CONTRIBUTING.md` for license and contribution guidelines.
