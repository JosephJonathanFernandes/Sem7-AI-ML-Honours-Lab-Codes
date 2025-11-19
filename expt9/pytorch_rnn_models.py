"""
Recurrent Neural Network (RNN) Implementation for Sequence Modeling using PyTorch
Author: AI Assistant
Date: November 19, 2025

This script implements three types of RNN models using PyTorch:
1. Basic RNN (Vanilla RNN)
2. LSTM (Long Short-Term Memory)
3. GRU (Gated Recurrent Unit)

For sequence modeling tasks including:
- Text Classification (IMDB sentiment analysis)
- Next-word prediction
- Time-series forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TextDataset(Dataset):
    """Custom dataset for text classification"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class SequenceDataset(Dataset):
    """Custom dataset for sequence prediction"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series forecasting"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

class RNNClassifier(nn.Module):
    """Basic RNN for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        
        # Take the last output
        final_output = rnn_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        return self.sigmoid(output)

class LSTMClassifier(nn.Module):
    """LSTM for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Take the last output
        final_output = lstm_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        return self.sigmoid(output)

class GRUClassifier(nn.Module):
    """GRU for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        # Take the last output
        final_output = gru_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        return self.sigmoid(output)

class RNNPredictor(nn.Module):
    """Basic RNN for sequence prediction"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.2):
        super(RNNPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        final_output = rnn_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        return output

class LSTMPredictor(nn.Module):
    """LSTM for sequence prediction"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_output = lstm_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        return output

class GRUPredictor(nn.Module):
    """GRU for sequence prediction"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        final_output = gru_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        return output

class RNNForecaster(nn.Module):
    """Basic RNN for time series forecasting"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super(RNNForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.fc2 = nn.Linear(25, output_dim)
        
    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        final_output = rnn_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = F.relu(self.fc1(final_output))
        output = self.fc2(output)
        return output

class LSTMForecaster(nn.Module):
    """LSTM for time series forecasting"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.fc2 = nn.Linear(25, output_dim)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = F.relu(self.fc1(final_output))
        output = self.fc2(output)
        return output

class GRUForecaster(nn.Module):
    """GRU for time series forecasting"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.fc2 = nn.Linear(25, output_dim)
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        final_output = gru_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = F.relu(self.fc1(final_output))
        output = self.fc2(output)
        return output

class RNNSequenceModels:
    """Class to implement and compare different RNN architectures using PyTorch"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.results = {}
        self.device = device
        
    def prepare_imdb_data_pytorch(self, max_words=5000, max_len=200):
        """Prepare IMDB-like dataset for text classification"""
        print("Creating synthetic IMDB-like dataset...")
        
        # Generate synthetic movie reviews with sentiment labels
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 
                         'outstanding', 'perfect', 'beautiful', 'incredible', 'awesome', 'superb']
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'worst', 'disappointing', 
                         'boring', 'waste', 'useless', 'poor', 'disgusting', 'pathetic']
        neutral_words = ['movie', 'film', 'story', 'character', 'plot', 'scene', 'actor', 
                        'director', 'script', 'cinema', 'watch', 'time', 'people', 'make', 'see']
        
        def generate_review(sentiment, length=50):
            words = []
            if sentiment == 1:  # Positive
                for _ in range(length):
                    if np.random.random() < 0.3:
                        words.append(np.random.choice(positive_words))
                    elif np.random.random() < 0.1:
                        words.append(np.random.choice(negative_words))
                    else:
                        words.append(np.random.choice(neutral_words))
            else:  # Negative
                for _ in range(length):
                    if np.random.random() < 0.3:
                        words.append(np.random.choice(negative_words))
                    elif np.random.random() < 0.1:
                        words.append(np.random.choice(positive_words))
                    else:
                        words.append(np.random.choice(neutral_words))
            return ' '.join(words)
        
        # Generate dataset
        reviews = []
        labels = []
        
        for i in range(5000):  # 5000 samples
            sentiment = i % 2  # Alternate between positive and negative
            review = generate_review(sentiment, np.random.randint(30, 80))
            reviews.append(review)
            labels.append(sentiment)
        
        # Create vocabulary
        all_words = []
        for review in reviews:
            words = review.lower().split()
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(max_words-2)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        # Convert reviews to sequences
        sequences = []
        for review in reviews:
            words = review.lower().split()
            seq = [word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
            sequences.append(seq[:max_len])  # Truncate if too long
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))  # 0 is <PAD>
            padded_sequences.append(seq)
        
        # Split data
        split_idx = int(0.8 * len(padded_sequences))
        x_train = np.array(padded_sequences[:split_idx])
        y_train = np.array(labels[:split_idx])
        x_test = np.array(padded_sequences[split_idx:])
        y_test = np.array(labels[split_idx:])
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Vocabulary size: {len(vocab)}")
        
        return (x_train, y_train), (x_test, y_test), len(vocab)
    
    def generate_text_sequence_data(self, seq_length=10):
        """Generate sequences for next-word prediction"""
        # Create a larger sample text
        sample_text = """
        The quick brown fox jumps over the lazy dog. Machine learning is a powerful tool 
        for solving complex problems. Neural networks can learn patterns from data. 
        Recurrent neural networks are particularly useful for sequence modeling tasks.
        Deep learning has revolutionized artificial intelligence. Natural language processing
        enables computers to understand human language. Time series forecasting predicts
        future values based on historical data. Computer vision allows machines to interpret
        visual information. Reinforcement learning teaches agents to make decisions.
        Data science combines statistics programming and domain expertise.
        """ * 20  # Repeat to have more data
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', sample_text.lower())
        words = text.split()
        
        # Create vocabulary
        unique_words = list(set(words))
        word_to_idx = {word: i for i, word in enumerate(unique_words)}
        idx_to_word = {i: word for i, word in enumerate(unique_words)}
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(words) - seq_length):
            seq = [word_to_idx[words[j]] for j in range(i, i + seq_length)]
            target = word_to_idx[words[i + seq_length]]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets), word_to_idx, idx_to_word
    
    def generate_time_series_data(self, n_samples=1000, seq_length=50):
        """Generate synthetic time series data"""
        # Create a combination of sine waves with noise
        t = np.linspace(0, 100, n_samples + seq_length)
        series = (np.sin(0.1 * t) + 0.5 * np.sin(0.2 * t) + 
                 0.3 * np.sin(0.15 * t) + 0.1 * np.random.randn(len(t)))
        
        # Create sequences
        X, y = [], []
        for i in range(n_samples):
            X.append(series[i:i + seq_length])
            y.append(series[i + seq_length])
        
        # Normalize data
        scaler = MinMaxScaler()
        X = np.array(X)
        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y = np.array(y).reshape(-1, 1)
        y_scaled = scaler.transform(y)
        
        return X_scaled, y_scaled.flatten(), scaler
    
    def train_model(self, model, train_loader, val_loader, epochs, criterion, optimizer, task_type):
        """Generic training function"""
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        model.to(self.device)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                
                if task_type == 'classification':
                    loss = criterion(output.squeeze(), target)
                    pred = (output.squeeze() > 0.5).float()
                    correct += (pred == target).sum().item()
                elif task_type == 'sequence_prediction':
                    loss = criterion(output, target)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                else:  # regression
                    loss = criterion(output.squeeze(), target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                total += target.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    if task_type == 'classification':
                        loss = criterion(output.squeeze(), target)
                        pred = (output.squeeze() > 0.5).float()
                        val_correct += (pred == target).sum().item()
                    elif task_type == 'sequence_prediction':
                        loss = criterion(output, target)
                        pred = output.argmax(dim=1)
                        val_correct += (pred == target).sum().item()
                    else:  # regression
                        loss = criterion(output.squeeze(), target)
                    
                    val_loss += loss.item()
                    val_total += target.size(0)
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if task_type in ['classification', 'sequence_prediction']:
                train_acc = 100. * correct / total
                val_acc = 100. * val_correct / val_total
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs}: '
                          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs}: '
                          f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs if task_type in ['classification', 'sequence_prediction'] else None,
            'val_acc': val_accs if task_type in ['classification', 'sequence_prediction'] else None
        }
        
        return history
    
    def train_text_classification(self):
        """Train all three RNN types for text classification"""
        print("\n" + "="*60)
        print("TASK 1: TEXT CLASSIFICATION (Synthetic IMDB-like Dataset)")
        print("="*60)
        
        # Prepare data
        (x_train, y_train), (x_test, y_test), vocab_size = self.prepare_imdb_data_pytorch()
        
        # Split training data for validation
        split_idx = int(0.8 * len(x_train))
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # Create datasets and dataloaders
        train_dataset = TextDataset(x_train, y_train)
        val_dataset = TextDataset(x_val, y_val)
        test_dataset = TextDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Model configurations
        models_config = {
            'RNN': RNNClassifier,
            'LSTM': LSTMClassifier,
            'GRU': GRUClassifier
        }
        
        for model_name, model_class in models_config.items():
            print(f"\nTraining {model_name} for text classification...")
            
            # Initialize model
            model = model_class(vocab_size=vocab_size, embed_dim=64, hidden_dim=64, output_dim=1)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            history = self.train_model(model, train_loader, val_loader, epochs=20, 
                                     criterion=criterion, optimizer=optimizer, 
                                     task_type='classification')
            
            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    loss = criterion(output.squeeze(), target)
                    pred = (output.squeeze() > 0.5).float()
                    
                    test_loss += loss.item()
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    
                    predictions.extend(pred.cpu().numpy())
                    true_labels.extend(target.cpu().numpy())
            
            test_acc = 100. * correct / total
            avg_test_loss = test_loss / len(test_loader)
            
            # Store results
            self.models[f'{model_name}_classifier'] = model
            self.histories[f'{model_name}_classifier'] = history
            self.results[f'{model_name}_classifier'] = {
                'test_accuracy': test_acc,
                'test_loss': avg_test_loss,
                'predictions': predictions,
                'true_labels': true_labels
            }
            
            print(f"{model_name} Test Accuracy: {test_acc:.2f}%")
            print(f"{model_name} Test Loss: {avg_test_loss:.4f}")
    
    def train_sequence_prediction(self):
        """Train all three RNN types for next-word prediction"""
        print("\n" + "="*60)
        print("TASK 2: NEXT-WORD PREDICTION")
        print("="*60)
        
        # Prepare data
        X, y, word_to_idx, idx_to_word = self.generate_text_sequence_data()
        vocab_size = len(word_to_idx)
        
        # Split data
        split_idx = int(0.8 * len(X))
        x_train, x_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training for validation
        val_split = int(0.8 * len(x_train))
        x_val = x_train[val_split:]
        y_val = y_train[val_split:]
        x_train = x_train[:val_split]
        y_train = y_train[:val_split]
        
        # Create datasets and dataloaders
        train_dataset = SequenceDataset(x_train, y_train)
        val_dataset = SequenceDataset(x_val, y_val)
        test_dataset = SequenceDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Model configurations
        models_config = {
            'RNN': RNNPredictor,
            'LSTM': LSTMPredictor,
            'GRU': GRUPredictor
        }
        
        for model_name, model_class in models_config.items():
            print(f"\nTraining {model_name} for sequence prediction...")
            
            # Initialize model
            model = model_class(vocab_size=vocab_size, embed_dim=50, hidden_dim=100)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            history = self.train_model(model, train_loader, val_loader, epochs=30, 
                                     criterion=criterion, optimizer=optimizer, 
                                     task_type='sequence_prediction')
            
            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    loss = criterion(output, target)
                    pred = output.argmax(dim=1)
                    
                    test_loss += loss.item()
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            test_acc = 100. * correct / total
            avg_test_loss = test_loss / len(test_loader)
            
            # Store results
            self.models[f'{model_name}_predictor'] = model
            self.histories[f'{model_name}_predictor'] = history
            self.results[f'{model_name}_predictor'] = {
                'test_accuracy': test_acc,
                'test_loss': avg_test_loss,
                'vocab_size': vocab_size,
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word
            }
            
            print(f"{model_name} Test Accuracy: {test_acc:.2f}%")
            print(f"{model_name} Test Loss: {avg_test_loss:.4f}")
    
    def train_time_series_forecasting(self):
        """Train all three RNN types for time series forecasting"""
        print("\n" + "="*60)
        print("TASK 3: TIME SERIES FORECASTING")
        print("="*60)
        
        # Generate data
        X, y, scaler = self.generate_time_series_data()
        
        # Split data
        split_idx = int(0.8 * len(X))
        x_train, x_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training for validation
        val_split = int(0.8 * len(x_train))
        x_val = x_train[val_split:]
        y_val = y_train[val_split:]
        x_train = x_train[:val_split]
        y_train = y_train[:val_split]
        
        # Reshape for RNN input (batch_size, seq_len, input_size)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        # Create datasets and dataloaders
        train_dataset = TimeSeriesDataset(x_train, y_train)
        val_dataset = TimeSeriesDataset(x_val, y_val)
        test_dataset = TimeSeriesDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Model configurations
        models_config = {
            'RNN': RNNForecaster,
            'LSTM': LSTMForecaster,
            'GRU': GRUForecaster
        }
        
        for model_name, model_class in models_config.items():
            print(f"\nTraining {model_name} for time series forecasting...")
            
            # Initialize model
            model = model_class(input_dim=1, hidden_dim=50, output_dim=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            history = self.train_model(model, train_loader, val_loader, epochs=40, 
                                     criterion=criterion, optimizer=optimizer, 
                                     task_type='regression')
            
            # Evaluate on test set
            model.eval()
            predictions = []
            true_values = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    predictions.extend(output.squeeze().cpu().numpy())
                    true_values.extend(target.cpu().numpy())
            
            # Calculate metrics
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            
            # Store results
            self.models[f'{model_name}_forecaster'] = model
            self.histories[f'{model_name}_forecaster'] = history
            self.results[f'{model_name}_forecaster'] = {
                'mse': mse,
                'mae': mae,
                'predictions': predictions,
                'true_values': true_values,
                'scaler': scaler
            }
            
            print(f"{model_name} MSE: {mse:.4f}")
            print(f"{model_name} MAE: {mae:.4f}")
    
    def plot_training_history(self, task_type='classifier'):
        """Plot training history for comparison"""
        model_names = ['RNN', 'LSTM', 'GRU']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, model_name in enumerate(model_names):
            model_key = f'{model_name}_{task_type}'
            
            if model_key in self.histories:
                history = self.histories[model_key]
                
                # Plot loss
                axes[0, i].plot(history['train_loss'], label='Training Loss', color='blue')
                axes[0, i].plot(history['val_loss'], label='Validation Loss', color='red')
                axes[0, i].set_title(f'{model_name} - Loss')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('Loss')
                axes[0, i].legend()
                axes[0, i].grid(True)
                
                # Plot accuracy or second metric
                if task_type != 'forecaster' and history['train_acc'] is not None:
                    axes[1, i].plot(history['train_acc'], label='Training Accuracy', color='blue')
                    axes[1, i].plot(history['val_acc'], label='Validation Accuracy', color='red')
                    axes[1, i].set_title(f'{model_name} - Accuracy')
                    axes[1, i].set_xlabel('Epoch')
                    axes[1, i].set_ylabel('Accuracy (%)')
                    axes[1, i].legend()
                    axes[1, i].grid(True)
                else:
                    # For forecaster, just plot loss again or hide
                    axes[1, i].plot(history['train_loss'], label='Training Loss', color='blue')
                    axes[1, i].plot(history['val_loss'], label='Validation Loss', color='red')
                    axes[1, i].set_title(f'{model_name} - Loss (Detailed)')
                    axes[1, i].set_xlabel('Epoch')
                    axes[1, i].set_ylabel('Loss')
                    axes[1, i].legend()
                    axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'pytorch_rnn_{task_type}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series_predictions(self):
        """Plot time series predictions"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_names = ['RNN', 'LSTM', 'GRU']
        
        for i, model_name in enumerate(model_names):
            model_key = f'{model_name}_forecaster'
            
            if model_key in self.results:
                results = self.results[model_key]
                true_values = results['true_values']
                predictions = results['predictions']
                
                # Plot first 100 points for clarity
                n_points = min(100, len(true_values))
                axes[i].plot(true_values[:n_points], label='True Values', alpha=0.7, color='blue')
                axes[i].plot(predictions[:n_points], label='Predictions', alpha=0.7, color='red')
                axes[i].set_title(f'{model_name} Predictions\nMSE: {results["mse"]:.4f}')
                axes[i].set_xlabel('Time Steps')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig('pytorch_rnn_time_series_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        print("\n" + "="*80)
        print("PYTORCH RNN PERFORMANCE SUMMARY")
        print("="*80)
        
        # Text Classification Results
        print("\n1. TEXT CLASSIFICATION (Synthetic IMDB-like Dataset)")
        print("-" * 50)
        for model_name in ['RNN', 'LSTM', 'GRU']:
            key = f'{model_name}_classifier'
            if key in self.results:
                acc = self.results[key]['test_accuracy']
                loss = self.results[key]['test_loss']
                print(f"{model_name:>8}: Accuracy = {acc:.2f}%, Loss = {loss:.4f}")
        
        # Sequence Prediction Results
        print("\n2. NEXT-WORD PREDICTION")
        print("-" * 50)
        for model_name in ['RNN', 'LSTM', 'GRU']:
            key = f'{model_name}_predictor'
            if key in self.results:
                acc = self.results[key]['test_accuracy']
                loss = self.results[key]['test_loss']
                print(f"{model_name:>8}: Accuracy = {acc:.2f}%, Loss = {loss:.4f}")
        
        # Time Series Forecasting Results
        print("\n3. TIME SERIES FORECASTING")
        print("-" * 50)
        for model_name in ['RNN', 'LSTM', 'GRU']:
            key = f'{model_name}_forecaster'
            if key in self.results:
                mse = self.results[key]['mse']
                mae = self.results[key]['mae']
                print(f"{model_name:>8}: MSE = {mse:.4f}, MAE = {mae:.4f}")
        
        print("\n" + "="*80)
        
        # Model parameter comparison
        print("\nMODEL PARAMETER COMPARISON")
        print("-" * 30)
        for model_name in ['RNN', 'LSTM', 'GRU']:
            classifier_key = f'{model_name}_classifier'
            if classifier_key in self.models:
                model = self.models[classifier_key]
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"{model_name:>8}: Total params = {total_params:,}, Trainable = {trainable_params:,}")
    
    def run_all_experiments(self):
        """Run all RNN experiments"""
        print("Starting PyTorch RNN Sequence Modeling Experiments...")
        print("This will train and evaluate RNN, LSTM, and GRU models on three tasks:")
        print("1. Text Classification (Synthetic IMDB-like)")
        print("2. Next-word Prediction")  
        print("3. Time Series Forecasting")
        
        # Run all experiments
        self.train_text_classification()
        self.train_sequence_prediction() 
        self.train_time_series_forecasting()
        
        # Generate plots
        print("\nGenerating visualization plots...")
        self.plot_training_history('classifier')
        self.plot_training_history('predictor')
        self.plot_training_history('forecaster')
        self.plot_time_series_predictions()
        
        # Print summary
        self.print_performance_summary()
        
        return self.models, self.results


def main():
    """Main function to run all experiments"""
    # Initialize the RNN models class
    rnn_models = RNNSequenceModels()
    
    # Run all experiments
    models, results = rnn_models.run_all_experiments()
    
    print("\nAll PyTorch experiments completed successfully!")
    print("Models trained: RNN, LSTM, GRU")
    print("Tasks completed: Text Classification, Sequence Prediction, Time Series Forecasting")
    print("Plots saved as PNG files in the current directory.")
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()