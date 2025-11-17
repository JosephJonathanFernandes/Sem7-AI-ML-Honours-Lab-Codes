# Experiment Execution Guide
## Quick Start Instructions for NNDL Experiment 1

### Pre-Experiment Checklist
- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space (2GB minimum, 10GB recommended)
- [ ] GPU drivers installed (if using CUDA)

### Step-by-Step Execution

#### 1. Environment Setup (5 minutes)
```powershell
# Navigate to experiment directory
cd "c:\Users\Joseph\Desktop\NNDL\expt1"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

#### 2. Dataset Preparation (Automatic)
- CIFAR-10 will be automatically downloaded on first run
- Total download: ~163 MB
- Extraction location: `./data/cifar-10-batches-py/`

#### 3. Training Execution

##### Option A: GPU Training (Recommended if available)
```powershell
python train_cifar10.py --epochs 20 --batch-size 128
```
*Expected time: 8-15 minutes*

##### Option B: CPU Training
```powershell
python train_cifar10.py --epochs 20 --batch-size 64 --no-cuda
```
*Expected time: 45-60 minutes*

#### 4. Results Collection
After training completion, check:
- [ ] `./checkpoints/best.pth` - Best model weights
- [ ] `./checkpoints/training_curves.png` - Performance visualization
- [ ] `./checkpoints/history.npz` - Training history data
- [ ] Terminal output - Final accuracy and loss values

### Expected Outputs

#### Console Output Example:
```
Epoch 1/20 - train_loss: 1.8234, train_acc: 32.15%, val_loss: 1.5432, val_acc: 43.21%
Epoch 2/20 - train_loss: 1.4567, train_acc: 45.78%, val_loss: 1.3456, val_acc: 52.34%
...
Epoch 20/20 - train_loss: 0.4123, train_acc: 85.67%, val_loss: 0.7234, val_acc: 78.45%
Training completed in 12.34 minutes. Best val acc: 79.12%
```

#### Performance Targets:
- **Training Accuracy**: 80-90%
- **Validation Accuracy**: 75-85%
- **Training Time**: 
  - GPU: 8-15 minutes
  - CPU: 45-60 minutes

### Troubleshooting Common Issues

#### CUDA Out of Memory
```powershell
# Reduce batch size
python train_cifar10.py --epochs 20 --batch-size 32
```

#### Slow CPU Training
```powershell
# Reduce workers and batch size
python train_cifar10.py --epochs 10 --batch-size 32 --no-cuda
```

#### Package Installation Issues
```powershell
# Update pip and reinstall
python -m pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

### Post-Experiment Analysis

#### 1. Review Training Curves
Open `./checkpoints/training_curves.png` to analyze:
- Loss convergence patterns
- Overfitting indicators
- Learning rate schedule effects

#### 2. Update Experiment Report
Fill in the actual results in `Experiment_1_Report.md`:
- Section 9.1: Training Performance
- Section 9.2: Learning Curves Analysis
- Add screenshots of training curves

#### 3. Performance Documentation
Record in your report:
- Hardware specifications used
- Actual training time
- Final accuracy numbers
- Any issues encountered

### Report Submission Checklist
- [ ] Complete experiment execution
- [ ] Update results section with actual numbers
- [ ] Include training curves screenshot
- [ ] Document hardware configuration
- [ ] Analyze strengths/weaknesses observed
- [ ] Complete future work section based on experience
- [ ] Proofread for technical accuracy

### Extended Experiments (Optional)
If time permits, try these variations:
```powershell
# Different learning rates
python train_cifar10.py --lr 5e-4 --epochs 20

# Longer training
python train_cifar10.py --epochs 50 --batch-size 128

# Different random seed
python train_cifar10.py --seed 123 --epochs 20
```

### File Organization
```
expt1/
├── Experiment_1_Report.md         # Main experiment report
├── Experiment_Execution_Guide.md  # This file
├── train_cifar10.py               # Training script
├── requirements.txt               # Dependencies
├── NNDL_expt1.ipynb              # Jupyter notebook version
├── checkpoints/                   # Generated results
│   ├── best.pth
│   ├── training_curves.png
│   └── history.npz
└── data/                         # CIFAR-10 dataset
    └── cifar-10-batches-py/
```

Remember to document any deviations from expected results and analyze why they might have occurred!