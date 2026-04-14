"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 4b: LSTM Deep Learning Model Training
========================================================================
This script implements a Long Short-Term Memory (LSTM) network to
predict heatwave risk based on sequential weather and AQI data.

Key features:
  - Sequence-to-One architecture (uses last W days to predict today)
  - Multi-task learning (Classification + Regression)
  - PyTorch implementation
========================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
import os, json, pickle

# ── CONFIG ─────────────────────────────────────────────────────────────
PROC_DIR  = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE = 14  # Use last 14 days
BATCH_SIZE  = 32
EPOCHS      = 50
LEARNING_RATE = 0.001
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi",
    "heat_index", "humidex", "temp_range", "temp_mean", "feels_like_excess",
    "wind_heat_ratio", "temp_max_roll7", "humidity_roll7", "aqi_roll7",
    "rainfall_roll7", "consec_hot_days", "is_summer"
]
# We'll use a subset of features for the LSTM input sequences to avoid redundancy
# while keeping the most critical temporal signals.

# ── DATASET CLASS ──────────────────────────────────────────────────────
class HeatwaveDataset(Dataset):
    def __init__(self, sequences, y_cls, y_reg):
        self.sequences = torch.FloatTensor(sequences)
        self.y_cls = torch.LongTensor(y_cls)
        self.y_reg = torch.FloatTensor(y_reg)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.y_cls[idx], self.y_reg[idx]

# ── LSTM MODEL ─────────────────────────────────────────────────────────
class HeatwaveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4):
        super(HeatwaveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        
        # Shared representations
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Heads
        self.classifier = nn.Linear(32, num_classes)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        shared = self.fc_shared(last_out)
        
        logits = self.classifier(shared)
        score = self.regressor(shared).squeeze(-1)
        
        return logits, score

# ── LOADING & PREVIEW ──────────────────────────────────────────────────
def prepare_sequences(df, window_size=WINDOW_SIZE):
    """Creates sequences for each city separately."""
    all_seqs = []
    all_y_cls = []
    all_y_reg = []
    
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    
    for city in df["city"].unique():
        city_df = df[df["city"] == city].sort_values("date")
        data = city_df[FEATURE_COLS].values
        labels_cls = city_df["risk_level"].values
        labels_reg = city_df["composite_score"].values
        
        for i in range(len(data) - window_size):
            all_seqs.append(data[i : i + window_size])
            all_y_cls.append(labels_cls[i + window_size])
            all_y_reg.append(labels_reg[i + window_size])
            
    return np.array(all_seqs), np.array(all_y_cls), np.array(all_y_reg), scaler

# ── TRAINING LOOP ──────────────────────────────────────────────────────
def train_lstm():
    print("\n" + "="*60)
    print("  LSTM TRAINING: Heatwave Risk Prediction")
    print("="*60)
    
    # Load data
    path = os.path.join(PROC_DIR, "labelled_all.csv")
    if not os.path.exists(path):
        print("[!] Labelled data not found. Run previous steps first.")
        return
    
    df = pd.read_csv(path, parse_dates=["date"])
    
    # Drop NaNs in critical columns
    df.dropna(subset=FEATURE_COLS + ["risk_level", "composite_score"], inplace=True)
    
    X_seq, y_cls, y_reg, scaler = prepare_sequences(df)
    
    print(f"  [✓] Sequences prepared: {X_seq.shape}")
    
    # Split
    n = len(X_seq)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    tr_idx, te_idx = indices[:split], indices[split:]
    
    train_ds = HeatwaveDataset(X_seq[tr_idx], y_cls[tr_idx], y_reg[tr_idx])
    test_ds  = HeatwaveDataset(X_seq[te_idx], y_cls[te_idx], y_reg[te_idx])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = HeatwaveLSTM(input_size=len(FEATURE_COLS)).to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print(f"  [>] Training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seqs, cls_labels, reg_labels in train_loader:
            seqs, cls_labels, reg_labels = seqs.to(DEVICE), cls_labels.to(DEVICE), reg_labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits, score = model(seqs)
            
            loss_cls = criterion_cls(logits, cls_labels)
            loss_reg = criterion_reg(score, reg_labels)
            
            loss = loss_cls + 0.1 * loss_reg # Weighted loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:2d}/{EPOCHS}: Loss = {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds_cls = []
    all_true_cls = []
    all_preds_reg = []
    all_true_reg = []
    
    with torch.no_grad():
        for seqs, cls_labels, reg_labels in test_loader:
            seqs, cls_labels, reg_labels = seqs.to(DEVICE), cls_labels.to(DEVICE), reg_labels.to(DEVICE)
            logits, score = model(seqs)
            
            all_preds_cls.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_true_cls.extend(cls_labels.cpu().numpy())
            all_preds_reg.extend(score.cpu().numpy())
            all_true_reg.extend(reg_labels.cpu().numpy())
            
    f1 = f1_score(all_true_cls, all_preds_cls, average="macro")
    mae = mean_absolute_error(all_true_reg, all_preds_reg)
    
    print("\n  [✓] Evaluation Results:")
    print(f"    Macro F1 (Classification): {f1:.3f}")
    print(f"    MAE (Regression)        : {mae:.2f}")
    
    # Save
    model_path = os.path.join(MODEL_DIR, "lstm_heatwave.pth")
    torch.save(model.state_dict(), model_path)
    
    meta = {
        "window_size": WINDOW_SIZE,
        "feature_cols": FEATURE_COLS,
        "hidden_size": 64,
        "num_layers": 2
    }
    with open(os.path.join(MODEL_DIR, "lstm_meta.json"), "w") as f:
        json.dump(meta, f)
        
    with open(os.path.join(MODEL_DIR, "lstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"\n  [✓] Model saved to {model_path}")
    print("  [✓] Metadata and scaler saved.")

if __name__ == "__main__":
    train_lstm()
