"""
Architecture Ablation Study

Test different model architectures to identify optimal configuration
for terrorism incident forecasting. Compares LSTM vs GRU, layer depth,
dropout rates, and other hyperparameters.

Usage:
    python src/ablations/architecture_ablations.py --data path/to/weekly_data.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_config, save_table, ensure_directories


class ArchitectureAblation:
    """
    Test different model architecture configurations
    """
    
    def __init__(self, config):
        self.config = config
        self.results = []
        self.seed = config.get('seed', 42)
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def create_sequences(self, data, target_col, sequence_length):
        """
        Create sequences for LSTM/GRU training
        
        Args:
            data: DataFrame with features
            target_col: Name of target column
            sequence_length: Number of time steps to look back
            
        Returns:
            X: 3D array (samples, timesteps, features)
            y: 1D array (samples,)
        """
        feature_cols = [col for col in data.columns if col != target_col]
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[feature_cols].iloc[i:(i + sequence_length)].values)
            y.append(data[target_col].iloc[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, arch_config, input_shape):
        """
        Build model based on architecture configuration
        
        Args:
            arch_config: Dictionary with architecture parameters
            input_shape: Tuple (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First layer
        if arch_config['model_type'] == 'LSTM':
            model.add(LSTM(
                arch_config['units'],
                activation='relu',
                input_shape=input_shape,
                return_sequences=(arch_config['layers'] > 1)
            ))
        else:  # GRU
            model.add(GRU(
                arch_config['units'],
                activation='relu',
                input_shape=input_shape,
                return_sequences=(arch_config['layers'] > 1)
            ))
        
        # Dropout after first layer
        if arch_config['dropout'] > 0:
            model.add(Dropout(arch_config['dropout']))
        
        # Second layer (if specified)
        if arch_config['layers'] > 1:
            if arch_config['model_type'] == 'LSTM':
                model.add(LSTM(arch_config['units'], activation='relu'))
            else:
                model.add(GRU(arch_config['units'], activation='relu'))
            
            if arch_config['dropout'] > 0:
                model.add(Dropout(arch_config['dropout']))
        
        # Output layer
        if arch_config.get('regularization', False):
            model.add(Dense(1, activation='linear', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        else:
            model.add(Dense(1, activation='linear'))
        
        # Compile
        optimizer = Adam(learning_rate=arch_config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_and_evaluate(self, arch_config, X_train, y_train, X_test, y_test):
        """
        Train model with given architecture and return metrics
        
        Args:
            arch_config: Architecture configuration dictionary
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation metrics
        """
        self._set_seeds()  # Reset seeds for each model
        
        # Build model
        model = self.build_model(arch_config, (X_train.shape[1], X_train.shape[2]))
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=arch_config['batch_size'],
            validation_split=0.15,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        predictions = model.predict(X_test, verbose=0).flatten()
        
        rmse = sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        epochs_trained = len(history.history['loss'])
        
        # Count parameters
        total_params = model.count_params()
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Epochs': epochs_trained,
            'Parameters': total_params
        }
    
    def run_ablations(self, data, target_col='attack_count', sequence_length=30, train_ratio=0.70):
        """
        Run architecture ablation study
        
        Args:
            data: DataFrame with features
            target_col: Name of target column
            sequence_length: LSTM/GRU sequence length
            train_ratio: Train/test split ratio
            
        Returns:
            DataFrame: Ablation results
        """
        print("\n" + "="*60)
        print("ARCHITECTURE ABLATION STUDY")
        print("="*60)
        print(f"Dataset: {len(data)} samples")
        print(f"Target: {target_col}")
        print(f"Sequence Length: {sequence_length}")
        
        # Prepare data
        feature_cols = [col for col in data.columns if col not in ['date', target_col]]
        data_model = data[feature_cols + [target_col]].copy()
        
        # Create sequences
        print("\nCreating sequences...")
        X, y = self.create_sequences(data_model, target_col, sequence_length)
        
        # Train/test split
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"Train: {len(X_train)} sequences")
        print(f"Test: {len(X_test)} sequences")
        print(f"Features: {X_train.shape[2]}")
        
        # Define architecture configurations to test
        architectures = [
            # Your existing models from notebook
            {
                'name': 'LSTM-2Layer-64units-Dropout0.3',
                'model_type': 'LSTM',
                'layers': 2,
                'units': 64,
                'dropout': 0.3,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'regularization': False,
                'description': 'Your Model 4 architecture'
            },
            {
                'name': 'GRU-2Layer-100units-Dropout0.2-L1L2',
                'model_type': 'GRU',
                'layers': 2,
                'units': 100,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 16,
                'regularization': True,
                'description': 'Your Model 3 architecture'
            },
            
            # Ablation: Test 1-layer vs 2-layer
            {
                'name': 'LSTM-1Layer-64units-Dropout0.3',
                'model_type': 'LSTM',
                'layers': 1,
                'units': 64,
                'dropout': 0.3,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'regularization': False,
                'description': 'Single layer variant'
            },
            
            # Ablation: Test without dropout
            {
                'name': 'LSTM-2Layer-64units-NoDropout',
                'model_type': 'LSTM',
                'layers': 2,
                'units': 64,
                'dropout': 0.0,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'regularization': False,
                'description': 'No regularization'
            },
            
            # Ablation: Test different unit sizes
            {
                'name': 'LSTM-2Layer-32units-Dropout0.3',
                'model_type': 'LSTM',
                'layers': 2,
                'units': 32,
                'dropout': 0.3,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'regularization': False,
                'description': 'Smaller capacity'
            },
            {
                'name': 'LSTM-2Layer-128units-Dropout0.3',
                'model_type': 'LSTM',
                'layers': 2,
                'units': 128,
                'dropout': 0.3,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'regularization': False,
                'description': 'Larger capacity'
            },
            
            # Ablation: GRU variants
            {
                'name': 'GRU-1Layer-100units-Dropout0.2',
                'model_type': 'GRU',
                'layers': 1,
                'units': 100,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 16,
                'regularization': False,
                'description': 'Single layer GRU'
            },
            {
                'name': 'GRU-2Layer-64units-Dropout0.3',
                'model_type': 'GRU',
                'layers': 2,
                'units': 64,
                'dropout': 0.3,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'regularization': False,
                'description': 'GRU with LSTM hyperparams'
            },
        ]
        
        print(f"\nTesting {len(architectures)} architecture configurations...")
        
        # Test each architecture
        for i, arch_config in enumerate(architectures, 1):
            print("\n" + "-"*60)
            print(f"[{i}/{len(architectures)}] {arch_config['name']}")
            print("-"*60)
            print(f"Type: {arch_config['model_type']}")
            print(f"Layers: {arch_config['layers']}")
            print(f"Units: {arch_config['units']}")
            print(f"Dropout: {arch_config['dropout']}")
            print(f"Learning Rate: {arch_config['learning_rate']}")
            print(f"Batch Size: {arch_config['batch_size']}")
            print(f"Regularization: {arch_config['regularization']}")
            print(f"Note: {arch_config['description']}")
            
            # Train and evaluate
            metrics = self.train_and_evaluate(arch_config, X_train, y_train, X_test, y_test)
            
            print(f"\nResults:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  R²:   {metrics['R2']:.4f}")
            print(f"  Epochs: {metrics['Epochs']}")
            print(f"  Parameters: {metrics['Parameters']:,}")
            
            self.results.append({
                'Architecture': arch_config['name'],
                'Model_Type': arch_config['model_type'],
                'Layers': arch_config['layers'],
                'Units': arch_config['units'],
                'Dropout': arch_config['dropout'],
                'Learning_Rate': arch_config['learning_rate'],
                'Batch_Size': arch_config['batch_size'],
                'Regularization': arch_config['regularization'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'Epochs_Trained': metrics['Epochs'],
                'Parameters': metrics['Parameters'],
                'Description': arch_config['description']
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('RMSE')
        
        return results_df


def main():
    """Run architecture ablation study"""
    print("="*60)
    print("ARCHITECTURE ABLATION STUDY")
    print("Evaluating model architecture configurations")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Ensure output directories
    ensure_directories(['reports/tables'])
    
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
    
    # Run ablations
    ablation = ArchitectureAblation(config)
    results = ablation.run_ablations(
        data,
        target_col=target_col,
        sequence_length=config['model'].get('sequence_length', 30),
        train_ratio=config['split'].get('train_ratio', 0.70)
    )
    
    # Display results
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print("\nRanked by RMSE:\n")
    print(results[['Architecture', 'Model_Type', 'Layers', 'Units', 'Dropout', 'RMSE', 'MAE', 'R2']].to_string(index=False))
    
    # Save results
    save_table(results, 'architecture_ablation_results', format='both')
    
    print("\n" + "="*60)
    print("✅ Architecture ablation study complete!")
    print("="*60)
    print("\nResults saved to:")
    print("  - reports/tables/architecture_ablation_results.csv")
    print("  - reports/tables/architecture_ablation_results.tex")
    
    # Summary insights
    print("\n📊 Key Insights:")
    best = results.iloc[0]
    print(f"  • Best Architecture: {best['Architecture']}")
    print(f"    RMSE: {best['RMSE']:.4f}, MAE: {best['MAE']:.4f}, R²: {best['R2']:.4f}")
    print(f"  • Parameters: {best['Parameters']:,}")
    print(f"  • {best['Description']}")
    
    # Compare LSTM vs GRU
    lstm_best = results[results['Model_Type'] == 'LSTM'].iloc[0]
    gru_best = results[results['Model_Type'] == 'GRU'].iloc[0]
    print(f"\n  • Best LSTM: {lstm_best['Architecture']} (RMSE: {lstm_best['RMSE']:.4f})")
    print(f"  • Best GRU:  {gru_best['Architecture']} (RMSE: {gru_best['RMSE']:.4f})")
    
    diff = abs(lstm_best['RMSE'] - gru_best['RMSE'])
    if diff < 0.1:
        print(f"  → LSTM and GRU perform similarly (difference: {diff:.4f})")
    elif lstm_best['RMSE'] < gru_best['RMSE']:
        print(f"  → LSTM outperforms GRU by {diff:.4f} RMSE")
    else:
        print(f"  → GRU outperforms LSTM by {diff:.4f} RMSE")


if __name__ == "__main__":
    main()
