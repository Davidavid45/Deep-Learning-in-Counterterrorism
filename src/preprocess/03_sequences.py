"""
Sequence Preparation for LSTM Models

Creates sliding window sequences from time series data
for supervised learning with LSTM networks.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_processed_data, save_processed_data


def create_sequences(data: np.ndarray, sequence_length: int, 
                     target_col_idx: int = 0) -> tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Input array (time_steps, features)
        sequence_length: Number of time steps in each sequence
        target_col_idx: Index of target column in features
        
    Returns:
        X (sequences), y (targets)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_col_idx])
    
    return np.array(X), np.array(y)


def prepare_sequences_by_group(df: pd.DataFrame, config: dict, 
                               grain_col: str) -> dict:
    """
    Create sequences for each group (region/country/attacktype).
    
    Args:
        df: Input DataFrame (aggregated time series)
        config: Configuration dictionary
        grain_col: Column to group by
        
    Returns:
        Dictionary with sequences for each group
    """
    sequence_length = config['model']['sequence_length']
    
    print(f"\nCreating sequences (length={sequence_length}) for each {grain_col}...")
    
    # Feature columns for LSTM input
    feature_cols = ['attack_count', 'total_casualties', 'total_killed', 'total_wounded']
    available_features = [col for col in feature_cols if col in df.columns]
    
    sequences_dict = {}
    groups = df[grain_col].unique()
    
    for group in groups:
        group_data = df[df[grain_col] == group].sort_values('week_start')
        
        # Skip if not enough data
        if len(group_data) <= sequence_length:
            continue
        
        # Extract features
        values = group_data[available_features].values
        
        # Create sequences
        X, y = create_sequences(values, sequence_length, target_col_idx=0)
        
        sequences_dict[group] = {
            'X': X,
            'y': y,
            'dates': group_data['week_start'].values[sequence_length:]
        }
    
    print(f"Created sequences for {len(sequences_dict)} groups")
    return sequences_dict


def combine_sequences(sequences_dict: dict) -> tuple:
    """
    Combine sequences from all groups into single arrays.
    
    Args:
        sequences_dict: Dictionary of sequences by group
        
    Returns:
        Combined X, y, and metadata
    """
    all_X, all_y, all_groups, all_dates = [], [], [], []
    
    for group, data in sequences_dict.items():
        all_X.append(data['X'])
        all_y.append(data['y'])
        all_groups.extend([group] * len(data['y']))
        all_dates.extend(data['dates'])
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"\nCombined sequences shape: X={X.shape}, y={y.shape}")
    
    return X, y, np.array(all_groups), np.array(all_dates)


def add_seasonal_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add seasonal lag features from config.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with seasonal features
    """
    seasonal_lag = config['model']['seasonal_lag_weeks']
    grain_col = df.columns[df.columns.str.contains('txt')][0]
    
    print(f"Adding seasonal lag features (lag={seasonal_lag})...")
    
    df = df.sort_values([grain_col, 'week_start'])
    df[f'attack_count_lag_{seasonal_lag}'] = df.groupby(grain_col)['attack_count'].shift(seasonal_lag)
    
    return df


def prepare_sequences(config: dict) -> tuple:
    """
    Main sequence preparation pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        X, y, groups, dates arrays
    """
    print("="*60)
    print("PREPARING SEQUENCES FOR LSTM")
    print("="*60)
    
    # Load weekly aggregated data
    df = load_processed_data("03_weekly_aggregated.csv", config)
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    print(f"Input shape: {df.shape}")
    
    # Determine grain column
    grain = config['aggregation']['weekly_grain']
    grain_col = 'region_txt' if grain == 'region' else 'country_txt' if grain == 'country' else 'attacktype1_txt'
    
    # Add seasonal features
    df = add_seasonal_features(df, config)
    
    # Remove rows with NaN in lag features
    df = df.dropna()
    
    # Create sequences by group
    sequences_dict = prepare_sequences_by_group(df, config, grain_col)
    
    # Combine all sequences
    X, y, groups, dates = combine_sequences(sequences_dict)
    
    print("\n" + "="*60)
    print("SEQUENCE PREPARATION COMPLETE")
    print("="*60)
    
    return X, y, groups, dates


if __name__ == "__main__":
    config = load_config()
    X, y, groups, dates = prepare_sequences(config)
    
    # Save sequences
    np.save(os.path.join(config['data']['data_dir'], 'X_sequences.npy'), X)
    np.save(os.path.join(config['data']['data_dir'], 'y_targets.npy'), y)
    np.save(os.path.join(config['data']['data_dir'], 'groups.npy'), groups)
    np.save(os.path.join(config['data']['data_dir'], 'dates.npy'), dates)
    
    print(f"\nSequences saved:")
    print(f"  X_sequences.npy: {X.shape}")
    print(f"  y_targets.npy: {y.shape}")
