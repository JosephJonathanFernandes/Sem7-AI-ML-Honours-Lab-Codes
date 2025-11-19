# PyTorch RNN Sequence Modeling - Comprehensive Implementation

This project implements and compares three types of Recurrent Neural Networks using **PyTorch**:
1. **Basic RNN (Vanilla RNN)**
2. **LSTM (Long Short-Term Memory)**
3. **GRU (Gated Recurrent Unit)**

## Overview

### Architecture Comparison

#### 1. Basic RNN (Vanilla RNN)
- **Architecture**: Simple recurrent connections
- **Pros**: Lightweight, fast training, simple implementation
- **Cons**: Vanishing gradient problem, poor long-term dependencies
- **Best for**: Short sequences, simple pattern recognition
- **PyTorch Layer**: `nn.RNN`

#### 2. LSTM (Long Short-Term Memory)
- **Architecture**: Cell state + hidden state with three gates (forget, input, output)
- **Pros**: Excellent long-term memory, solves vanishing gradients
- **Cons**: More complex, slower training, more parameters
- **Best for**: Long sequences, complex temporal dependencies
- **PyTorch Layer**: `nn.LSTM`

#### 3. GRU (Gated Recurrent Unit)
- **Architecture**: Hidden state with two gates (update, reset)
- **Pros**: Simpler than LSTM, good performance, fewer parameters
- **Cons**: Less expressive than LSTM for very complex tasks
- **Best for**: Good balance of performance and efficiency
- **PyTorch Layer**: `nn.GRU`

## Experiments Implemented

### Task 1: Text Classification
- **Dataset**: Synthetic IMDB-like movie reviews (5,000 samples)
- **Objective**: Binary sentiment classification (positive/negative)
- **Architecture**: 
  - Embedding layer → RNN/LSTM/GRU → Dense → Sigmoid
  - Vocabulary size: 5,000 words
  - Sequence length: 200 tokens
- **Metrics**: Accuracy, Binary Cross-Entropy Loss

### Task 2: Next-Word Prediction
- **Dataset**: Custom text corpus with vocabulary
- **Objective**: Predict next word in sequence
- **Architecture**: 
  - Embedding layer → RNN/LSTM/GRU → Dense → Softmax
  - Sequence length: 10 words
- **Metrics**: Categorical Accuracy, Cross-Entropy Loss

### Task 3: Time Series Forecasting
- **Dataset**: Synthetic time series (sine waves + noise)
- **Objective**: Predict next value in sequence
- **Architecture**: 
  - RNN/LSTM/GRU (2 layers) → Dense layers
  - Sequence length: 50 time steps
- **Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE)

## PyTorch Implementation Features

### Model Architectures
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return self.sigmoid(output)
```

### Training Features
- **Custom Dataset Classes**: Efficient data loading with `torch.utils.data.Dataset`
- **DataLoader**: Batched training with automatic shuffling
- **GPU Support**: Automatic CUDA detection and model deployment
- **Training Loop**: Comprehensive training with validation
- **Early Stopping**: Prevention of overfitting
- **Loss Functions**: Appropriate for each task type
- **Optimizers**: Adam optimizer with learning rate scheduling

### Visualization and Analysis
- Training history plots (loss and accuracy curves)
- Time series prediction comparisons
- Performance summary tables
- Parameter count comparisons
- Model architecture diagrams

## Expected Performance Patterns

### Text Classification
- **LSTM**: Highest accuracy (~75-85%) due to better sequence understanding
- **GRU**: Good balance of performance and speed (~70-80%)
- **RNN**: Lower accuracy (~60-70%) due to vanishing gradients

### Next-Word Prediction
- **LSTM**: Best at capturing word dependencies (~40-60% accuracy)
- **GRU**: Competitive performance with faster training
- **RNN**: Struggles with longer context dependencies

### Time Series Forecasting
- **LSTM**: Superior pattern recognition in temporal data
- **GRU**: Excellent performance with faster convergence
- **RNN**: May struggle with long-term patterns

## Key PyTorch Advantages

1. **Dynamic Computation Graph**: Flexible model debugging
2. **Pythonic**: Natural Python programming style
3. **Research-Friendly**: Easy experimentation and modification
4. **GPU Acceleration**: Seamless CUDA integration
5. **Custom Datasets**: Flexible data pipeline creation
6. **Model Serialization**: Easy saving and loading of trained models

## Implementation Highlights

### Memory Management
- Efficient tensor operations
- Gradient accumulation control
- Automatic memory cleanup

### Training Optimization
- Batch processing for efficient GPU utilization
- Validation monitoring during training
- Loss tracking and visualization

### Model Comparison
- Fair comparison with identical architectures
- Parameter counting and analysis
- Performance benchmarking across tasks

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements_pytorch.txt
```

### Running the Experiments
```bash
python pytorch_rnn_models.py
```

### Expected Output
1. **Training Progress**: Real-time loss and accuracy updates
2. **Performance Summary**: Comparison table of all models
3. **Visualizations**: Training curves and prediction plots
4. **Model Statistics**: Parameter counts and architecture details

## File Structure
```
expt9/
├── pytorch_rnn_models.py          # Main implementation
├── requirements_pytorch.txt       # Dependencies
├── README_pytorch.md             # This documentation
└── Generated outputs:
    ├── pytorch_rnn_classifier_training_history.png
    ├── pytorch_rnn_predictor_training_history.png
    ├── pytorch_rnn_forecaster_training_history.png
    └── pytorch_rnn_time_series_predictions.png
```

## Technical Details

### Gradient Handling
- **RNN**: Suffers from vanishing/exploding gradients
- **LSTM**: Gates control gradient flow effectively
- **GRU**: Simplified gating mechanism

### Parameter Efficiency
- **RNN**: ~15K parameters (fastest)
- **GRU**: ~45K parameters (balanced)
- **LSTM**: ~60K parameters (most expressive)

### Training Characteristics
- **Convergence**: LSTM > GRU > RNN
- **Training Speed**: RNN > GRU > LSTM
- **Memory Usage**: RNN < GRU < LSTM

## Conclusion

This implementation demonstrates the evolution of RNN architectures and their trade-offs:

1. **Basic RNN**: Simple but limited by vanishing gradients
2. **LSTM**: Complex but powerful for long sequences
3. **GRU**: Optimal balance for most applications

The PyTorch implementation provides a clean, efficient, and research-ready codebase for understanding and experimenting with recurrent neural networks across different sequence modeling tasks.