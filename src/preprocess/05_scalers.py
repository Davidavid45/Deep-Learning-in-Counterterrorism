"""
Data Scaling for Neural Networks

Applies MinMax or Standard scaling to numerical features
for LSTM and other neural network models.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_processed_data


def fit_scaler(data: np.ndarray, scaler_type: str = 'minmax') -> object:
    """
    Fit scaler to training data.
    
    Args:
        data: Input array to fit
        scaler_type: 'minmax' or 'standard'
        
    Returns:
        Fitted scaler object
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    scaler.fit(data)
    return scaler


def scale_sequences(X: np.ndarray, y: np.ndarray, scaler_type: str = 'minmax') -> tuple:
    """
    Scale sequence data for LSTM.
    
    Args:
        X: Input sequences (samples, timesteps, features)
        y: Target values
        scaler_type: Type of scaling
        
    Returns:
        Scaled X, y, and fitted scalers
    """
    print(f"\nScaling sequences using {scaler_type} scaler...")
    
    # Reshape X for scaling
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    # Fit and transform X
    X_scaler = fit_scaler(X_reshaped, scaler_type)
    X_scaled = X_scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Fit and transform y
    y_scaler = fit_scaler(y.reshape(-1, 1), scaler_type)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).flatten()
    
    print(f"  X scaled: {X_scaled.shape}")
    print(f"  y scaled: {y_scaled.shape}")
    print(f"  X range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"  y range: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")
    
    return X_scaled, y_scaled, X_scaler, y_scaler


def save_scalers(X_scaler: object, y_scaler: object, config: dict = None) -> None:
    """
    Save scaler objects to disk.
    
    Args:
        X_scaler: Feature scaler
        y_scaler: Target scaler
        config: Optional configuration dictionary
    """
    models_dir = 'src/models'
    os.makedirs(models_dir, exist_ok=True)
    
    scalers = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
    
    filepath = os.path.join(models_dir, 'scalers.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"\nScalers saved to {filepath}")


def load_scalers(config: dict = None) -> dict:
    """
    Load scaler objects from disk.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with X_scaler and y_scaler
    """
    filepath = os.path.join('src/models', 'scalers.pkl')
    
    with open(filepath, 'rb') as f:
        scalers = pickle.load(f)
    
    print(f"Scalers loaded from {filepath}")
    return scalers


def inverse_transform_predictions(y_pred: np.ndarray, y_scaler: object) -> np.ndarray:
    """
    Convert scaled predictions back to original scale.
    
    Args:
        y_pred: Scaled predictions
        y_scaler: Fitted target scaler
        
    Returns:
        Predictions in original scale
    """
    return y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()


def apply_scaling(config: dict) -> tuple:
    """
    Main scaling pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Scaled X, y and scalers
    """
    print("="*60)
    print("SCALING DATA FOR LSTM")
    print("="*60)
    
    # Load sequences
    data_dir = config['data']['data_dir']
    X = np.load(os.path.join(data_dir, 'X_sequences.npy'))
    y = np.load(os.path.join(data_dir, 'y_targets.npy'))
    
    print(f"\nLoaded sequences:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Scale data
    X_scaled, y_scaled, X_scaler, y_scaler = scale_sequences(X, y, scaler_type='minmax')
    
    # Save scalers
    save_scalers(X_scaler, y_scaler, config)
    
    print("\n" + "="*60)
    print("SCALING COMPLETE")
    print("="*60)
    
    return X_scaled, y_scaled, X_scaler, y_scaler


if __name__ == "__main__":
    config = load_config()
    X_scaled, y_scaled, X_scaler, y_scaler = apply_scaling(config)
    
    # Save scaled sequences
    data_dir = config['data']['data_dir']
    np.save(os.path.join(data_dir, 'X_scaled.npy'), X_scaled)
    np.save(os.path.join(data_dir, 'y_scaled.npy'), y_scaled)
    
    print(f"\nScaled data saved:")
    print(f"  X_scaled.npy: {X_scaled.shape}")
    print(f"  y_scaled.npy: {y_scaled.shape}")
