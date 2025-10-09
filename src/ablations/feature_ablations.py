"""
Feature Ablation Study

Systematically remove feature groups to evaluate their contribution
to model performance. This helps identify which engineered features
are most critical for terrorism incident forecasting.

Usage:
    python src/ablations/feature_ablations.py --data path/to/weekly_data.csv
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
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_config, save_table, ensure_directories


class FeatureAblation:
    """
    Test impact of removing feature groups from LSTM model
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
    
    def identify_feature_groups(self, df):
        """
        Identify feature groups in the dataset
        
        Returns:
            dict: Feature groups with column names
        """
        all_cols = [col for col in df.columns if col not in ['date', 'Year', 'Month', 'Week']]
        
        # Define feature groups based on naming patterns
        groups = {}
        
        # Temporal encodings (sin/cos features)
        temporal_cols = [col for col in all_cols if any(x in col.lower() for x in ['_sin', '_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos'])]
        if temporal_cols:
            groups['Temporal Encoding'] = temporal_cols
        
        # Holiday indicator
        holiday_cols = [col for col in all_cols if 'holiday' in col.lower()]
        if holiday_cols:
            groups['IsHoliday'] = holiday_cols
        
        # Lag features
        lag_cols = [col for col in all_cols if 'lag' in col.lower() and not any(x in col.lower() for x in ['seasonal', 'rolling'])]
        if lag_cols:
            groups['Lag Features'] = lag_cols
        
        # Seasonal lag features
        seasonal_lag_cols = [col for col in all_cols if 'seasonal_lag' in col.lower()]
        if seasonal_lag_cols:
            groups['Seasonal Lag Features'] = seasonal_lag_cols
        
        # Rolling statistics
        rolling_cols = [col for col in all_cols if 'rolling' in col.lower()]
        if rolling_cols:
            groups['Rolling Statistics'] = rolling_cols
        
        # Days since last attack
        days_since_cols = [col for col in all_cols if 'days_since' in col.lower()]
        if days_since_cols:
            groups['Days Since Last'] = days_since_cols
        
        # Geographic features (region/country text columns - for "No Geo" ablation)
        geo_cols = [col for col in all_cols if any(x in col.lower() for x in ['region_txt', 'country_txt', 'region_', 'country_'])]
        if geo_cols:
            groups['Geographic Features'] = geo_cols
        
        # Regional dummies (one-hot encoded)
        region_cols = [col for col in all_cols if col.startswith('region_') and col not in geo_cols]
        if region_cols:
            groups['Regional Dummies'] = region_cols
        
        # Attack type dummies
        attack_cols = [col for col in all_cols if col.startswith('attacktype_')]
        if attack_cols:
            groups['AttackType Dummies'] = attack_cols
        
        # Target type dummies
        target_cols = [col for col in all_cols if col.startswith('target_type_') or col.startswith('targettype_')]
        if target_cols:
            groups['Target Type Dummies'] = target_cols
        
        # Weapon type dummies
        weapon_cols = [col for col in all_cols if col.startswith('weapon_type_') or col.startswith('weapontype_')]
        if weapon_cols:
            groups['Weapon Type Dummies'] = weapon_cols
        
        # Casualties/intensity features
        casualty_cols = [col for col in all_cols if any(x in col.lower() for x in ['casualt', 'killed', 'wounded', 'nkill', 'nwound'])]
        casualty_cols = [col for col in casualty_cols if col not in lag_cols + rolling_cols]  # Exclude lags and rolling
        if casualty_cols:
            groups['Casualty Features'] = casualty_cols
        
        return groups
    
    def create_sequences(self, data, target_col, sequence_length):
        """
        Create sequences for LSTM training
        
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
        
        # Convert to numpy arrays with explicit float32 dtype
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def build_lstm_model(self, input_shape):
        """
        Build Bidirectional LSTM model (same as best model)
        
        Args:
            input_shape: Tuple (timesteps, features)
            
        Returns:
            Compiled Keras model
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
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, description="Model"):
        """
        Train LSTM and return evaluation metrics
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            description: Model description for logging
            
        Returns:
            dict: Evaluation metrics (RMSE, MAE, R²)
        """
        self._set_seeds()  # Reset seeds for each model
        
        # Build model
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
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
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Epochs': epochs_trained
        }
    
    def run_ablations(self, data, target_col='attack_count', sequence_length=30, train_ratio=0.70):
        """
        Run feature ablation study
        
        Args:
            data: DataFrame with all features
            target_col: Name of target column
            sequence_length: LSTM sequence length
            train_ratio: Train/test split ratio
            
        Returns:
            DataFrame: Ablation results
        """
        print("\n" + "="*60)
        print("FEATURE ABLATION STUDY")
        print("="*60)
        print(f"Dataset: {len(data)} samples")
        print(f"Target: {target_col}")
        print(f"Sequence Length: {sequence_length}")
        print(f"Train Ratio: {train_ratio}")
        
        # Identify feature groups
        feature_groups = self.identify_feature_groups(data)
        print(f"\nIdentified {len(feature_groups)} feature groups:")
        for name, cols in feature_groups.items():
            print(f"  - {name}: {len(cols)} features")
        
        # Prepare data - exclude date/text columns
        exclude_cols = ['date', 'week_start', 'first_attack_date', 'region_txt', 'country_txt', target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        data_model = data[feature_cols + [target_col]].copy().fillna(0)
        
        # Create sequences
        print("\nCreating sequences...")
        X, y = self.create_sequences(data_model, target_col, sequence_length)
        
        # Train/test split (chronological)
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"Train: {len(X_train)} sequences")
        print(f"Test: {len(X_test)} sequences")
        
        # Baseline: All features
        print("\n" + "-"*60)
        print("[Baseline] Training with ALL features")
        print("-"*60)
        print(f"Features: {X_train.shape[2]}")
        
        metrics_baseline = self.train_and_evaluate(X_train, y_train, X_test, y_test, "Baseline")
        
        print(f"RMSE: {metrics_baseline['RMSE']:.4f}")
        print(f"MAE:  {metrics_baseline['MAE']:.4f}")
        print(f"R²:   {metrics_baseline['R2']:.4f}")
        print(f"Epochs: {metrics_baseline['Epochs']}")
        
        self.results.append({
            'Feature Group': 'Baseline (All Features)',
            'Features Count': X_train.shape[2],
            'RMSE': metrics_baseline['RMSE'],
            'MAE': metrics_baseline['MAE'],
            'R2': metrics_baseline['R2'],
            'RMSE_Impact': 0.0,
            'MAE_Impact': 0.0,
            'Interpretation': 'Full model',
            'Epochs': metrics_baseline['Epochs']
        })
        
        # Test removing each feature group
        for i, (group_name, feature_list) in enumerate(feature_groups.items(), 1):
            print("\n" + "-"*60)
            print(f"[{i}/{len(feature_groups)}] Testing WITHOUT {group_name}")
            print("-"*60)
            
            # Create dataset without this feature group
            cols_to_keep = [col for col in feature_cols if col not in feature_list]
            data_subset = data[cols_to_keep + [target_col]].copy().fillna(0)
            
            # Create sequences
            X_sub, y_sub = self.create_sequences(data_subset, target_col, sequence_length)
            X_train_sub = X_sub[:train_size]
            X_test_sub = X_sub[train_size:]
            
            print(f"Removed: {len(feature_list)} features")
            print(f"Remaining: {X_train_sub.shape[2]} features")
            
            # Train and evaluate
            metrics = self.train_and_evaluate(X_train_sub, y_train, X_test_sub, y_test, f"w/o {group_name}")
            
            # Calculate impact
            rmse_impact = metrics['RMSE'] - metrics_baseline['RMSE']
            mae_impact = metrics['MAE'] - metrics_baseline['MAE']
            
            # Interpret impact
            if abs(rmse_impact) > 1.0:
                interpretation = "Critical feature group"
            elif abs(rmse_impact) > 0.5:
                interpretation = "Important feature group"
            elif abs(rmse_impact) > 0.2:
                interpretation = "Moderate contribution"
            else:
                interpretation = "Minimal impact"
            
            print(f"RMSE: {metrics['RMSE']:.4f} (Δ{rmse_impact:+.4f})")
            print(f"MAE:  {metrics['MAE']:.4f} (Δ{mae_impact:+.4f})")
            print(f"R²:   {metrics['R2']:.4f}")
            print(f"→ {interpretation}")
            
            self.results.append({
                'Feature Group': group_name,
                'Features Count': X_train_sub.shape[2],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'RMSE_Impact': rmse_impact,
                'MAE_Impact': mae_impact,
                'Interpretation': interpretation,
                'Epochs': metrics['Epochs']
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('RMSE_Impact', ascending=False)
        
        return results_df


def main():
    """Run feature ablation study"""
    print("="*60)
    print("FEATURE ABLATION STUDY")
    print("Evaluating contribution of feature groups")
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
    ablation = FeatureAblation(config)
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
    print("\nRanked by RMSE Impact (positive = performance degraded):\n")
    print(results[['Feature Group', 'RMSE', 'RMSE_Impact', 'MAE_Impact', 'Interpretation']].to_string(index=False))
    
    # Save results
    save_table(results, 'feature_ablation_results', format='both')
    
    print("\n" + "="*60)
    print("✅ Feature ablation study complete!")
    print("="*60)
    print("\nResults saved to:")
    print("  - reports/tables/feature_ablation_results.csv")
    print("  - reports/tables/feature_ablation_results.tex")
    
    # Summary insights
    print("\n📊 Key Insights:")
    top_3 = results.nlargest(3, 'RMSE_Impact')
    for idx, row in top_3.iterrows():
        if row['Feature Group'] != 'Baseline (All Features)':
            print(f"  • {row['Feature Group']}: {row['RMSE_Impact']:+.4f} RMSE impact")


if __name__ == "__main__":
    main()
