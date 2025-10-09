"""
Train/Validation/Test Split

Splits time series data chronologically to prevent data leakage.
Uses ratios from config file.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config


def split_time_series(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                      train_ratio: float, val_ratio: float, 
                      test_ratio: float) -> dict:
    """
    Split time series data chronologically.
    
    Args:
        X: Input sequences
        y: Target values
        dates: Date array for chronological ordering
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Dictionary with train/val/test splits
    """
    # Sort by date
    sort_idx = np.argsort(dates)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    dates_sorted = dates[sort_idx]
    
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    splits = {
        'X_train': X_sorted[:train_end],
        'y_train': y_sorted[:train_end],
        'dates_train': dates_sorted[:train_end],
        
        'X_val': X_sorted[train_end:val_end],
        'y_val': y_sorted[train_end:val_end],
        'dates_val': dates_sorted[train_end:val_end],
        
        'X_test': X_sorted[val_end:],
        'y_test': y_sorted[val_end:],
        'dates_test': dates_sorted[val_end:]
    }
    
    return splits


def print_split_info(splits: dict) -> None:
    """Print information about data splits."""
    print("\nData Split Summary:")
    print(f"  Training:   {len(splits['X_train']):>6} samples ({splits['dates_train'].min()} to {splits['dates_train'].max()})")
    print(f"  Validation: {len(splits['X_val']):>6} samples ({splits['dates_val'].min()} to {splits['dates_val'].max()})")
    print(f"  Test:       {len(splits['X_test']):>6} samples ({splits['dates_test'].min()} to {splits['dates_test'].max()})")
    print(f"  Total:      {len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']):>6} samples")


def perform_split(config: dict) -> dict:
    """
    Main split pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with all splits
    """
    print("="*60)
    print("SPLITTING DATA: TRAIN/VAL/TEST")
    print("="*60)
    
    # Load scaled sequences
    data_dir = config['data']['data_dir']
    X = np.load(os.path.join(data_dir, 'X_scaled.npy'))
    y = np.load(os.path.join(data_dir, 'y_scaled.npy'))
    dates = np.load(os.path.join(data_dir, 'dates.npy'), allow_pickle=True)
    
    print(f"\nLoaded data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Get split ratios from config
    train_ratio = config['split']['train_ratio']
    val_ratio = config['split']['val_ratio']
    test_ratio = config['split']['test_ratio']
    
    print(f"\nSplit ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    
    # Perform split
    splits = split_time_series(X, y, dates, train_ratio, val_ratio, test_ratio)
    
    # Print info
    print_split_info(splits)
    
    print("\n" + "="*60)
    print("SPLIT COMPLETE")
    print("="*60)
    
    return splits


if __name__ == "__main__":
    config = load_config()
    splits = perform_split(config)
    
    # Save splits
    data_dir = config['data']['data_dir']
    
    for key, value in splits.items():
        filepath = os.path.join(data_dir, f'{key}.npy')
        np.save(filepath, value)
        print(f"Saved: {key}.npy")
    
    print("\nAll splits saved successfully")
