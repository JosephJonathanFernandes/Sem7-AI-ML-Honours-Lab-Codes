Experiment 1: Case Study on PyTorch - CIFAR-10 CNN

Files
- train_cifar10.py: Training script for CIFAR-10 using a simple CNN.
- requirements.txt: Python dependencies.

Quick start (PowerShell)

# Create a virtual environment (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run training (uses GPU if available; will run on CPU if no GPU)
# If you do NOT have a GPU, add the `--no-cuda` flag and reduce batch size (e.g., 32 or 64)
python train_cifar10.py --epochs 20 --batch-size 128

# CPU-only example (recommended batch size: 32 or 64)
python train_cifar10.py --epochs 20 --batch-size 64 --no-cuda

Dataset
- CIFAR-10 is automatically downloaded by the script using torchvision when you run it the first time. The torchvision dataset will be stored under the `data/` folder by default.
- If you prefer to download manually or inspect the dataset, the official CIFAR-10 page is: https://www.cs.toronto.edu/~kriz/cifar.html

Running without GPU
- If you don't have a GPU, the script will run on CPU. Training will be significantly slower. To reduce memory and CPU usage, use a smaller `--batch-size` (32 or 64) and set `--num_workers` to 0 in the DataLoader (you can edit `train_cifar10.py` to change num_workers in get_data_loaders).

Outputs
- Checkpoints and training_curves.png will be saved in the ./checkpoints directory by default.
