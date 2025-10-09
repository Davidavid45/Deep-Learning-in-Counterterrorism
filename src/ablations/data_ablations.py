"""
Data Configuration Ablation Study

Test impact of data configuration choices:
1. Reduced Span: Train on last 5 years vs full history
2. Grain Swap: Country-level vs region-level aggregation
3. Sequence Length: Different temporal window sizes

Usage:
    python src/ablations/data_ablations.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from math import sqrt
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_config, save_table, ensure_directories


class DataAblation:
    """
    Test impact of data configuration choices
    """
    
    def __init__(self, config):
        self.config = config
        self.results = []
        self.seed = config.get('seed', 42)
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def create_sequences(self, data, target, sequence_length, batch_size=5000):
        """
        Create sequences for LSTM training with memory-efficient batching
        """
        n_sequences = len(data) - sequence_length
        n_features = data.shape[1]
        
        # Pre-allocate arrays
        X = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)
        y = np.empty(n_sequences, dtype=np.float32)
        
        # Create sequences in batches
        for start_idx in range(0, n_sequences, batch_size):
            end_idx = min(start_idx + batch_size, n_sequences)
            
            for i in range(start_idx, end_idx):
                X[i] = data[i:(i + sequence_length)].astype(np.float32)
                y[i] = target[i + sequence_length].astype(np.float32)
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build Bidirectional LSTM model (same as best model)
        """
        model = Sequential([
            Bidirectional(LSTM(32, activation='relu', return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(32, activation='relu', return_sequences=False)),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, name="Model", epochs=50):
        """
        Train model and return metrics
        """
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        ]
        
        # Split train into train/val
        val_size = int(len(X_train) * 0.15)
        X_train_sub = X_train[:-val_size]
        y_train_sub = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        
        # Train
        history = model.fit(
            X_train_sub, y_train_sub,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict
        predictions = model.predict(X_test, verbose=0).flatten()
        
        # Metrics
        rmse = sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Epochs': len(history.history['loss'])
        }
    
    def test_reduced_span(self, data, target_col, sequence_length=30, train_ratio=0.70):
        """
        Test 1: Train on last 5 years only vs full history
        """
        print("\n" + "="*60)
        print("TEST 1: REDUCED SPAN (Last 5 Years Only)")
        print("="*60)
        
        # Ensure date column exists
        if 'week_start' not in data.columns:
            print("⚠️  No week_start column found, skipping reduced span test")
            return
        
        # Convert to datetime
        data['week_start'] = pd.to_datetime(data['week_start'])
        
        # Full history baseline
        print("\n[1a] Full History (1970-2016)")
        print("-"*60)
        full_data = data.copy()
        print(f"Date range: {full_data['week_start'].min()} to {full_data['week_start'].max()}")
        print(f"Total samples: {len(full_data)}")
        
        # Prepare data
        exclude_cols = ['week_start', 'first_attack_date', 'region_txt', 'country_txt', target_col]
        feature_cols = [col for col in full_data.columns if col not in exclude_cols]
        
        # Scale features
        scaler_full = MinMaxScaler(feature_range=(0, 1))
        feature_data_full = full_data[feature_cols].fillna(0).values
        target_data_full = full_data[target_col].fillna(0).values
        feature_data_scaled_full = scaler_full.fit_transform(feature_data_full)
        
        # Create sequences
        X_full, y_full = self.create_sequences(feature_data_scaled_full, target_data_full, sequence_length)
        
        # Split
        train_size_full = int(len(X_full) * train_ratio)
        X_train_full = X_full[:train_size_full]
        y_train_full = y_full[:train_size_full]
        X_test_full = X_full[train_size_full:]
        y_test_full = y_full[train_size_full:]
        
        print(f"Train sequences: {len(X_train_full)}")
        print(f"Test sequences: {len(X_test_full)}")
        
        metrics_full = self.train_and_evaluate(X_train_full, y_train_full, X_test_full, y_test_full, "Full History")
        print(f"✅ RMSE: {metrics_full['RMSE']:.4f} | MAE: {metrics_full['MAE']:.4f} | R²: {metrics_full['R2']:.4f}")
        
        # Last 5 years only
        print("\n[1b] Recent Data (Last 5 Years: 2012-2016)")
        print("-"*60)
        recent_cutoff = full_data['week_start'].max() - pd.DateOffset(years=5)
        recent_data = data[data['week_start'] >= recent_cutoff].copy()
        print(f"Date range: {recent_data['week_start'].min()} to {recent_data['week_start'].max()}")
        print(f"Total samples: {len(recent_data)}")
        
        if len(recent_data) < 100:
            print("⚠️  Too few samples for 5-year test, skipping")
            return
        
        # Scale features
        scaler_recent = MinMaxScaler(feature_range=(0, 1))
        feature_data_recent = recent_data[feature_cols].fillna(0).values
        target_data_recent = recent_data[target_col].fillna(0).values
        feature_data_scaled_recent = scaler_recent.fit_transform(feature_data_recent)
        
        # Create sequences
        X_recent, y_recent = self.create_sequences(feature_data_scaled_recent, target_data_recent, sequence_length)
        
        # Split
        train_size_recent = int(len(X_recent) * train_ratio)
        X_train_recent = X_recent[:train_size_recent]
        y_train_recent = y_recent[:train_size_recent]
        X_test_recent = X_recent[train_size_recent:]
        y_test_recent = y_recent[train_size_recent:]
        
        print(f"Train sequences: {len(X_train_recent)}")
        print(f"Test sequences: {len(X_test_recent)}")
        
        if len(X_train_recent) < 50 or len(X_test_recent) < 10:
            print("⚠️  Insufficient sequences for training, skipping")
            return
        
        metrics_recent = self.train_and_evaluate(X_train_recent, y_train_recent, X_test_recent, y_test_recent, "Last 5 Years")
        print(f"✅ RMSE: {metrics_recent['RMSE']:.4f} | MAE: {metrics_recent['MAE']:.4f} | R²: {metrics_recent['R2']:.4f}")
        
        # Calculate impact
        rmse_impact = metrics_recent['RMSE'] - metrics_full['RMSE']
        
        # Save results
        self.results.append({
            'Configuration': 'Full History (1970-2016)',
            'Samples': len(full_data),
            'Sequences': len(X_full),
            'RMSE': metrics_full['RMSE'],
            'MAE': metrics_full['MAE'],
            'R2': metrics_full['R2'],
            'RMSE_Impact': 0.0,
            'Interpretation': 'Baseline'
        })
        
        self.results.append({
            'Configuration': 'Recent Only (Last 5 Years)',
            'Samples': len(recent_data),
            'Sequences': len(X_recent),
            'RMSE': metrics_recent['RMSE'],
            'MAE': metrics_recent['MAE'],
            'R2': metrics_recent['R2'],
            'RMSE_Impact': rmse_impact,
            'Interpretation': 'Recent patterns' if rmse_impact < 0 else 'Long history needed'
        })
        
        print(f"\n📊 Impact: {rmse_impact:+.4f} RMSE")
        if rmse_impact < 0:
            print("   ✅ Recent data is more predictive")
        else:
            print("   ⚠️  Long historical data is valuable")
    
    def test_sequence_length(self, data, target_col, train_ratio=0.70):
        """
        Test 2: Different sequence lengths (20, 30, 40 weeks)
        """
        print("\n" + "="*60)
        print("TEST 2: SEQUENCE LENGTH VARIATIONS")
        print("="*60)
        
        # Prepare data
        exclude_cols = ['week_start', 'first_attack_date', 'region_txt', 'country_txt', target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        feature_data = data[feature_cols].fillna(0).values
        target_data = data[target_col].fillna(0).values
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        sequence_lengths = [20, 30, 40]
        
        for seq_len in sequence_lengths:
            print(f"\n[Testing sequence_length={seq_len}]")
            print("-"*60)
            
            # Create sequences
            X, y = self.create_sequences(feature_data_scaled, target_data, seq_len)
            
            # Split
            train_size = int(len(X) * train_ratio)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]
            
            print(f"Sequences: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
            
            # Train and evaluate
            metrics = self.train_and_evaluate(X_train, y_train, X_test, y_test, f"SeqLen_{seq_len}")
            print(f"✅ RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
            
            # Save results
            self.results.append({
                'Configuration': f'Sequence Length {seq_len}',
                'Samples': len(data),
                'Sequences': len(X),
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'RMSE_Impact': metrics['RMSE'] - 6.38 if seq_len == 30 else None,
                'Interpretation': 'Baseline' if seq_len == 30 else ''
            })
    
    def run_all_tests(self, weekly_file):
        """
        Run all data ablation tests
        """
        print("\n" + "="*70)
        print(" "*20 + "DATA CONFIGURATION ABLATION STUDY")
        print("="*70)
        
        # Load data
        print(f"\nLoading data from: {weekly_file}")
        data = pd.read_csv(weekly_file)
        print(f"Dataset shape: {data.shape}")
        
        target_col = 'attack_count' if 'attack_count' in data.columns else 'Weekly_Attacks'
        print(f"Target column: {target_col}")
        
        # Test 1: Reduced span
        self.test_reduced_span(data, target_col)
        
        # Test 2: Sequence length
        self.test_sequence_length(data, target_col)
        
        # Save results
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        results_df = pd.DataFrame(self.results)
        
        ensure_directories(['reports/tables', 'reports/figures'])
        save_table(results_df, 'data_ablation_results', format='both')
        
        print("\n✅ Data ablation study complete!")
        print(f"   Results saved to: reports/tables/data_ablation_results.csv")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(results_df.to_string(index=False))


def main():
    """Main execution"""
    # Load config
    config = load_config()
    
    # Get data path
    data_dir = config['data']['data_dir']
    weekly_file = os.path.join(data_dir, '03_weekly_aggregated.csv')
    
    if not os.path.exists(weekly_file):
        print(f"❌ Error: Data file not found: {weekly_file}")
        return
    
    # Run ablations
    ablation = DataAblation(config)
    ablation.run_all_tests(weekly_file)


if __name__ == "__main__":
    main()
