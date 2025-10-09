"""
Label Encoding for Categorical Variables

Encodes categorical variables to numeric format and saves
encoders for inverse transformation during inference.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_processed_data, save_processed_data


def encode_categorical_columns(df: pd.DataFrame, categorical_cols: list) -> tuple:
    """
    Apply label encoding to categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of columns to encode
        
    Returns:
        Encoded DataFrame and dictionary of encoders
    """
    print(f"\nEncoding {len(categorical_cols)} categorical columns...")
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col not in df.columns:
            print(f"  Warning: {col} not found in DataFrame")
            continue
        
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
        print(f"  {col}: {len(le.classes_)} unique values")
    
    return df_encoded, encoders


def save_encoders(encoders: dict, config: dict) -> None:
    """
    Save encoder objects to disk.
    
    Args:
        encoders: Dictionary of LabelEncoder objects
        config: Configuration dictionary
    """
    models_dir = 'src/models'
    os.makedirs(models_dir, exist_ok=True)
    
    filepath = os.path.join(models_dir, 'label_encoders.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f"\nEncoders saved to {filepath}")


def load_encoders(config: dict = None) -> dict:
    """
    Load encoder objects from disk.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of LabelEncoder objects
    """
    filepath = os.path.join('src/models', 'label_encoders.pkl')
    
    with open(filepath, 'rb') as f:
        encoders = pickle.load(f)
    
    print(f"Encoders loaded from {filepath}")
    return encoders


def apply_encoding(config: dict) -> pd.DataFrame:
    """
    Main encoding pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with encoded categorical variables
    """
    print("="*60)
    print("ENCODING CATEGORICAL VARIABLES")
    print("="*60)
    
    # Load featured data
    df = load_processed_data("02_featured_data.csv", config)
    print(f"Input shape: {df.shape}")
    
    # Define categorical columns to encode
    categorical_cols = [
        'country_txt', 'region_txt', 'provstate', 'city',
        'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt', 'gname'
    ]
    
    # Encode columns
    df_encoded, encoders = encode_categorical_columns(df, categorical_cols)
    
    # Save encoders
    save_encoders(encoders, config)
    
    print("\n" + "="*60)
    print("ENCODING COMPLETE")
    print("="*60)
    
    return df_encoded


if __name__ == "__main__":
    config = load_config()
    df_encoded = apply_encoding(config)
    
    # Save encoded data
    save_processed_data(df_encoded, "04_encoded_data.csv", config)
    
    print(f"\nEncoded data shape: {df_encoded.shape}")
    print(f"Encoded columns: {[col for col in df_encoded.columns if col.endswith('_encoded')]}")
