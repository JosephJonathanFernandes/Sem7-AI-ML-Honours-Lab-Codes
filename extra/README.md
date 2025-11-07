# Experiment 11–12: Advanced Sequence & Structure Models (PyTorch)

This folder contains concise, runnable scripts demonstrating major RNN, Recursive NN, and GNN variants:

- LSTM (lstm_pytorch.py)
- GRU (gru_pytorch.py)
- BiLSTM (bilstm_pytorch.py)
- Seq2Seq: Encoder–Decoder (seq2seq_pytorch.py)
- Tree-RNN (tree_rnn_numpy.py)
- RNTN (tree_rntn_numpy.py)
- Tree-LSTM (tree_lstm_pytorch.py)
- Graph Neural Network (gnn_pytorch.py)

All scripts are CPU-friendly and use synthetic data.

## Setup (Windows PowerShell)

```powershell
# CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# NumPy for the tree demos
pip install numpy
```

## How to run

```powershell
python .\expt11_12\lstm_pytorch.py
python .\expt11_12\gru_pytorch.py
python .\expt11_12\bilstm_pytorch.py
python .\expt11_12\seq2seq_pytorch.py
python .\expt11_12\tree_rnn_numpy.py
python .\expt11_12\tree_rntn_numpy.py
python .\expt11_12\tree_lstm_pytorch.py
python .\expt11_12\gnn_pytorch.py
```

Each script prints a brief training/progress message or a small output to verify correctness.