"""
Improved LSTM Ensemble Training for Predictive Maintenance.

Key improvements over original:
  1. Bootstrap sampling: Each model trains on a different 80% random subset,
     creating genuine ensemble diversity (critical for meaningful uncertainty).
  2. Learning rate scheduler: ReduceLROnPlateau halves LR when validation loss plateaus.
  3. Early stopping: Stops training when validation loss hasn't improved for PATIENCE epochs.
  4. Gradient clipping: Prevents exploding gradients on noisy C-MAPSS data.
  5. Explicit random seeds: Ensures reproducible but diverse initialisations.

Usage:
  python main_train_ensemble.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import time

from src.config import *
from src.preprocessing import load_combined_data, calculate_rul, process_data, create_sequences, CMAPSSDataset
from src.lstm_model import RUL_LSTM


def bootstrap_split(dataset, ratio, seed):
    """
    Create a bootstrap sample: randomly select `ratio` fraction of the dataset.
    Each ensemble member gets a DIFFERENT subset, driving model disagreement
    on ambiguous inputs (which is exactly what we want for uncertainty).
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)
    n_sample = int(n * ratio)
    indices = rng.choice(n, size=n_sample, replace=False)
    
    # Use remaining indices for validation
    all_indices = set(range(n))
    train_indices = set(indices.tolist())
    val_indices = list(all_indices - train_indices)
    
    return Subset(dataset, indices.tolist()), Subset(dataset, val_indices)


def train_one_model(model_idx, full_dataset, save_dir='models'):
    """Train a single ensemble member with bootstrap sampling and early stopping."""
    
    seed = 42 + model_idx * 17  # Deterministic but different seed per model
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"  Training Model {model_idx + 1}/{ENSEMBLE_SIZE}  (seed={seed})")
    print(f"{'='*60}")
    
    # --- Bootstrap Split ---
    train_subset, val_subset = bootstrap_split(full_dataset, BOOTSTRAP_RATIO, seed)
    print(f"  Bootstrap: {len(train_subset)} train / {len(val_subset)} val samples")
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model ---
    model = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(save_dir, f'ensemble_model_{model_idx}.pth')
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # --- LR Scheduler ---
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- Logging ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"Train={avg_train_loss:.6f}  Val={avg_val_loss:.6f}  "
                  f"LR={current_lr:.6f}")
        
        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break
    
    print(f"  Model {model_idx} saved: best val loss = {best_val_loss:.6f}")
    
    # Convert to approximate RMSE in cycles for interpretability
    # MSE is on normalised targets (0-1), so RMSE_cycles = sqrt(MSE) * 125
    approx_rmse = np.sqrt(best_val_loss) * 125.0
    print(f"  Approx validation RMSE: {approx_rmse:.2f} cycles")
    
    return best_val_loss


if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Prepare Data
    print("=" * 60)
    print("  LSTM ENSEMBLE TRAINING (Improved)")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Config: {ENSEMBLE_SIZE} models, {EPOCHS} max epochs, "
          f"LR={LEARNING_RATE}, patience={PATIENCE}")
    print(f"Bootstrap ratio: {BOOTSTRAP_RATIO} (each model sees {int(BOOTSTRAP_RATIO*100)}% of data)\n")
    
    print("--- Loading Combined Data (FD001 + FD002) ---")
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, scaler = process_data(df, DROP_SENSORS, DROP_SETTINGS)
    
    # Generate all sequences
    features = [c for c in df_clean.columns if c not in ['unit', 'time', 'RUL']]
    print(f"Features: {len(features)}")
    X, y = create_sequences(df_clean, WINDOW_SIZE, features)
    print(f"Total sequences: {len(X)}")
    
    # Create full dataset (bootstrap splitting happens per model)
    full_dataset = CMAPSSDataset(X, y)
    
    # 2. Train Ensemble
    if not os.path.exists('models'):
        os.makedirs('models')
    
    all_losses = []
    for i in range(ENSEMBLE_SIZE):
        loss = train_one_model(i, full_dataset)
        all_losses.append(loss)
    
    # 3. Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Models trained: {ENSEMBLE_SIZE}")
    print(f"  Average best val loss: {np.mean(all_losses):.6f}")
    print(f"  Approx avg RMSE: {np.sqrt(np.mean(all_losses)) * 125:.2f} cycles")
    print(f"{'='*60}")
    
    # Save the scaler for evaluation
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    print("\nNext step: python main_evaluate_ensemble.py")