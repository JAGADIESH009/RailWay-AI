"""
Train Model Pipeline for RailWay AI
Trains a PyTorch Bidirectional LSTM text classifier from scratch with randomly initialized weights.
Saves model weights, vocabulary, label mapping, configurations, and evaluation plots.
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set deterministic random seeds for full reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Simple regex-based tokenizer
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.strip().split()
    return tokens

# Build vocabulary strictly from training tokens
def build_vocab(tokenized_texts, min_freq=1):
    freq = {}
    for tokens in tokenized_texts:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab

# Text to index sequence
def text_to_sequence(tokens, vocab, max_len=35):
    unk_idx = vocab.get("<UNK>", 1)
    pad_idx = vocab.get("<PAD>", 0)
    seq = [vocab.get(t, unk_idx) for t in tokens]
    if len(seq) < max_len:
        seq = seq + [pad_idx] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# PyTorch Dataset
class RailwayDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Model Architecture: Bidirectional LSTM with Embedding from scratch
class RailwayLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_classes=6, 
                 num_layers=2, bidirectional=True, dropout=0.3, pad_idx=0):
        super(RailwayLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        # Randomly initialize embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0)
            
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        # Both mean-pool and max-pool concatenated gives 2 * lstm_out_dim features
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(lstm_out_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        lstm_out, (hn, cn) = self.lstm(embedded)
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Dual Pooling: Mean-pooling and Max-pooling across sequence dimension
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, hidden_dim * 4]
        
        x = self.dropout1(pooled)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

def train():
    os.makedirs("model_artifacts", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Load dataset
    csv_path = os.path.join("data", "railway_questions_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_dataset.py first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} samples across {df['category'].nunique()} classes.")
    
    # Label encoding
    classes = sorted(df["category"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for i, c in enumerate(classes)}
    
    # 2. Stratified Split: 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["category"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["category"]
    )
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # 3. Tokenization & Vocab built STRICTLY from train set (Zero Leakage)
    train_tokens = [tokenize(q) for q in train_df["question"]]
    val_tokens = [tokenize(q) for q in val_df["question"]]
    test_tokens = [tokenize(q) for q in test_df["question"]]
    
    vocab = build_vocab(train_tokens, min_freq=1)
    vocab_size = len(vocab)
    max_len = 35
    print(f"Vocabulary size (train only): {vocab_size} tokens")
    
    # Convert to sequences
    X_train = [text_to_sequence(t, vocab, max_len) for t in train_tokens]
    y_train = [label2idx[c] for c in train_df["category"]]
    
    X_val = [text_to_sequence(t, vocab, max_len) for t in val_tokens]
    y_val = [label2idx[c] for c in val_df["category"]]
    
    X_test = [text_to_sequence(t, vocab, max_len) for t in test_tokens]
    y_test = [label2idx[c] for c in test_df["category"]]
    
    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(RailwayDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RailwayDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(RailwayDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    model = RailwayLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        num_classes=len(classes),
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
        pad_idx=0
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 5. Training Loop
    epochs = 28
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    best_val_loss = float("inf")
    best_model_state = None
    
    print("\nStarting Training Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)
            
        train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation pass
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                total_val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)
                
        val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val
        scheduler.step(val_loss)
        
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.1f}%")
            
    # Load best weights
    model.load_state_dict(best_model_state)
    
    # 6. Test Set Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    test_acc = accuracy_score(all_targets, all_preds)
    print(f"\n==========================================")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"==========================================")
    
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=classes)
    print("\nClassification Report on Unseen Test Set:")
    print(report_text)
    
    cm = confusion_matrix(all_targets, all_preds)
    
    # 7. Save Plots
    # A) Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(f"RailWay AI LSTM - Test Confusion Matrix (Acc: {test_acc*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("True Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join("plots", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # B) Training Curves
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, epochs + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2, linestyle="--")
    plt.title("Cross-Entropy Loss vs. Epochs", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#2ca02c", lw=2)
    plt.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    plt.title("Accuracy (%) vs. Epochs", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    curves_path = os.path.join("plots", "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {curves_path}")
    
    # 8. Save Artifacts
    # Model Weights
    pth_path = os.path.join("model_artifacts", "railway_lstm.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Saved model weights to {pth_path}")
    
    # Vocab
    vocab_path = os.path.join("model_artifacts", "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
        
    # Label mapping
    label_path = os.path.join("model_artifacts", "label_mapping.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "label2idx": label2idx,
            "idx2label": idx2label
        }, f, indent=2)
        
    # Model Configuration
    config = {
        "model_architecture": "Bidirectional LSTM with Randomly Initialized Embeddings",
        "vocab_size": vocab_size,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "num_classes": len(classes),
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.3,
        "max_seq_len": max_len,
        "pad_idx": 0,
        "unk_idx": 1,
        "classes": classes,
        "framework": f"PyTorch {torch.__version__}",
        "pretraining": "None (Trained strictly from scratch)",
        "external_api_dependencies": "None"
    }
    config_path = os.path.join("model_artifacts", "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    # Evaluation Metrics JSON
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_sample_count": len(test_df),
        "training_history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    metrics_path = os.path.join("model_artifacts", "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved configuration and metrics to model_artifacts/")
    print("Training pipeline finished successfully.")
    return test_acc, history

if __name__ == "__main__":
    train()
