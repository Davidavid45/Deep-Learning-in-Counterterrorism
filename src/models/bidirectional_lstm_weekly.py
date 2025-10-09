"""
Bidirectional LSTM for Weekly Terrorism Attack Forecasting

This model improves upon the standard LSTM by processing sequences in both
forward and backward directions simultaneously. This bidirectional approach
captures temporal dependencies that work in both directions:
- Forward: past → present (attack build-up, escalation patterns)
- Backward: future → present (aftermath patterns, retaliation cycles)

Architecture:
    Input (sequence_length, features)
    → Bidirectional LSTM Layer 1 (32 units × 2 directions = 64 total)
    → Dropout (0.2)
    → Bidirectional LSTM Layer 2 (32 units × 2 directions = 64 total)
    → Dropout (0.2)
    → Dense (1 unit, linear activation)

Key Advantages:
    - Captures attack preparation phases (forward) and aftermath/retaliation (backward)
    - Learns symmetric seasonal patterns (ramp-up ↔ decline)
    - Better at detecting attack clusters and regional spillover effects


Usage:
    python src/models/bidirectional_lstm_weekly.py
"""

import os
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_config, save_model, save_figure, save_table, ensure_directories


class BidirectionalLSTMModel:
    """
    Bidirectional LSTM model for terrorism forecasting
    
    Processes sequences in both forward and backward directions to capture
    temporal dependencies that work bidirectionally in terrorism patterns.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.seed = config.get('seed', 42)
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def build_model(self, input_shape):
        """
        Build Bidirectional LSTM model
        
        Args:
            input_shape: Tuple (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # First Bidirectional LSTM layer
        # Forward: processes weeks 1→30
        # Backward: processes weeks 30→1
        # Output: concatenation of both directions (64 units total)
        x = Bidirectional(
            LSTM(32, activation='relu', return_sequences=True),
            name='bidirectional_lstm_1'
        )(inputs)
        x = Dropout(0.2, name='dropout_1')(x)
        
        # Second Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(32, activation='relu', return_sequences=False),
            name='bidirectional_lstm_2'
        )(x)
        x = Dropout(0.2, name='dropout_2')(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='Bidirectional_LSTM')
        
        # Compile
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_sequences(self, data, target, sequence_length=30, batch_size=5000):
        """
        Create sequences for LSTM training with memory-efficient batching
        
        Args:
            data: Feature array (n_samples, n_features)
            target: Target array (n_samples,)
            sequence_length: Number of time steps to look back
            batch_size: Number of sequences to create at once (memory management)
            
        Returns:
            X, y: Sequence arrays with explicit float32 dtype
        """
        n_sequences = len(data) - sequence_length
        n_features = data.shape[1]
        
        # Pre-allocate arrays (more memory efficient than appending)
        X = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)
        y = np.empty(n_sequences, dtype=np.float32)
        
        # Create sequences in batches to avoid memory spikes
        for start_idx in range(0, n_sequences, batch_size):
            end_idx = min(start_idx + batch_size, n_sequences)
            
            for i in range(start_idx, end_idx):
                X[i] = data[i:(i + sequence_length)].astype(np.float32)
                y[i] = target[i + sequence_length].astype(np.float32)
        
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with early stopping and learning rate reduction
        """
        print("\n" + "="*60)
        print("TRAINING BIDIRECTIONAL LSTM")
        print("="*60)
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Input shape: {X_train.shape}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='src/models/checkpoints/bidirectional_lstm_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Ensure checkpoint directory exists
        ensure_directories(['src/models/checkpoints'])
        
        # Train
        print("\nTraining...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        print(f"Final train loss: {self.history.history['loss'][-1]:.6f}")
        print(f"Final val loss: {self.history.history['val_loss'][-1]:.6f}")
        print(f"Epochs trained: {len(self.history.history['loss'])}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Predictions
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        rmse = sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  R²:   {r2:.4f}")
        
        return metrics, predictions
    
    def plot_training_history(self, save_path='bidirectional_lstm_training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Bidirectional LSTM: Training History', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, save_path)
        print(f"✅ Training history saved to: reports/figures/{save_path}")
    
    def plot_predictions(self, y_true, y_pred, save_path='bidirectional_lstm_predictions.png'):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        axes[0].plot(y_true, label='Actual', linewidth=2, alpha=0.7)
        axes[0].plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('Attack Count', fontsize=12)
        axes[0].set_title('Bidirectional LSTM: Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=30)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Attack Count', fontsize=12)
        axes[1].set_ylabel('Predicted Attack Count', fontsize=12)
        axes[1].set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, save_path)
        print(f"✅ Predictions plot saved to: reports/figures/{save_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("BIDIRECTIONAL LSTM - WEEKLY TERRORISM FORECASTING")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Ensure directories
    ensure_directories(['reports/figures', 'reports/tables', 'src/models/checkpoints'])
    
    # Load data
    data_dir = config['data']['data_dir']
    weekly_file = os.path.join(data_dir, '03_weekly_aggregated.csv')
    
    if not os.path.exists(weekly_file):
        print(f"\n❌ Error: Weekly aggregated data not found at {weekly_file}")
        print("Please run preprocessing first: python src/preprocess/run_preprocess.py")
        return
    
    print(f"\nLoading data from: {weekly_file}")
    data = pd.read_csv(weekly_file)
    
    # Determine target column
    target_col = 'attack_count' if 'attack_count' in data.columns else 'Weekly_Attacks'
    
    if target_col not in data.columns:
        print(f"\n❌ Error: Target column not found in data")
        print(f"Available columns: {data.columns.tolist()}")
        return
    
    print(f"Target column: {target_col}")
    print(f"Dataset shape: {data.shape}")
    
    # Prepare data - only exclude date/text columns
    exclude_cols = ['week_start', 'first_attack_date', 'region_txt', 'country_txt', target_col]
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"Features selected ({len(feature_cols)}): {feature_cols}")
    
    data_model = data[feature_cols + [target_col]].copy()
    
    # Handle NaN values (from lag features at start of series)
    print(f"\nChecking for NaN values...")
    nan_count = data_model.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️  Found {nan_count} NaN values (from lag features), filling with 0")
        data_model = data_model.fillna(0)
    else:
        print(f"✅ No NaN values found")
    
    # Scale features to 0-1 range (critical for LSTM performance)
    print(f"\n🔧 Scaling features with MinMaxScaler (0-1 range)...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = data_model[feature_cols].values
    target_data = data_model[target_col].values
    
    feature_data_scaled = scaler.fit_transform(feature_data)
    
    # Save scaler for future use
    scaler_path = 'src/models/feature_scaler_blstm.pkl'
    os.makedirs('src/models', exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to {scaler_path}")
    
    # Reconstruct dataframe with scaled features
    data_scaled = pd.DataFrame(feature_data_scaled, columns=feature_cols, index=data_model.index)
    data_scaled[target_col] = target_data
    
    print(f"✅ Features scaled to [0, 1] range")
    print(f"   Target kept in original scale (attack_count)")
    data_model = data_scaled
    
    # Create model
    model = BidirectionalLSTMModel(config)
    
    # Create sequences
    sequence_length = config['model'].get('sequence_length', 30)
    print(f"\nCreating sequences (length={sequence_length})...")
    
    # Extract features and target as numpy arrays
    features_for_sequences = data_model[feature_cols].values
    target_for_sequences = data_model[target_col].values
    
    X, y = model.create_sequences(features_for_sequences, target_for_sequences, sequence_length)
    
    print(f"Total sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    
    # Free memory after sequence creation
    del data_model, features_for_sequences, target_for_sequences
    gc.collect()
    
    # Train/val/test split (chronological)
    train_ratio = config['split'].get('train_ratio', 0.70)
    val_ratio = config['split'].get('val_ratio', 0.15)
    
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(X_val)} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(X_test)} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Train model
    epochs = config['model'].get('epochs', 50)
    batch_size = config['model'].get('batch_size', 32)
    
    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    metrics, predictions = model.evaluate(X_test, y_test)
    
    # Plot results
    model.plot_training_history()
    model.plot_predictions(y_test, predictions)
    
    # Save model
    save_model(model.model, 'bidirectional_lstm_weekly')
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'Model': 'Bidirectional_LSTM_Weekly',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MSE': metrics['MSE'],
        'R2': metrics['R2'],
        'Epochs': len(history.history['loss']),
        'Parameters': model.model.count_params()
    }])
    
    save_table(metrics_df, 'bidirectional_lstm_metrics', format='both')
    
    # Compare with previous models
    print("\n" + "="*60)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("="*60)
    
    # Load previous results
    try:
        lstm_attention_metrics = pd.read_csv('reports/tables/lstm_attention_metrics.csv')
        lstm_attention_rmse = lstm_attention_metrics['RMSE'].values[0]
        print(f"LSTM Attention RMSE: {lstm_attention_rmse:.4f}")
    except:
        lstm_attention_rmse = 9.19
        print(f"LSTM Attention RMSE: {lstm_attention_rmse:.4f} (from previous run)")
    
    print(f"Bidirectional LSTM RMSE: {metrics['RMSE']:.4f}")
    
    improvement_vs_attention = (lstm_attention_rmse - metrics['RMSE']) / lstm_attention_rmse * 100
    improvement_vs_baseline = (9.8858 - metrics['RMSE']) / 9.8858 * 100
    
    print(f"\nImprovement vs LSTM Attention: {improvement_vs_attention:+.2f}%")
    print(f"Improvement vs Baseline (Linear Reg): {improvement_vs_baseline:+.2f}%")
    
    if metrics['RMSE'] < lstm_attention_rmse:
        print("\n🎉 Bidirectional LSTM is the NEW BEST MODEL!")
    else:
        print("\n⚠️  LSTM Attention still performs better")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  • Model: src/models/bidirectional_lstm_weekly/")
    print("  • Figures: reports/figures/bidirectional_lstm_*.png")
    print("  • Metrics: reports/tables/bidirectional_lstm_metrics.csv")


if __name__ == "__main__":
    main()
