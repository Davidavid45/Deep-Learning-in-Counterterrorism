"""
Data Cleaning Script for Global Terrorism Database

This script handles:
- Loading raw data
- Removing duplicates
- Handling missing values
- Fixing data types
- Validating geographic coordinates
- Standardizing text fields
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_raw_data, save_processed_data


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame without duplicates
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    else:
        print("No duplicates found")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values appropriately for different column types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    print("\nHandling missing values...")
    
    # Keep track of missing values
    missing_before = df.isnull().sum().sum()
    
    # Critical columns - remove rows if these are missing
    critical_cols = ['iyear', 'imonth', 'iday', 'country_txt', 'region_txt']
    for col in critical_cols:
        if col in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=[col])
            dropped = initial_len - len(df)
            if dropped > 0:
                print(f"  Dropped {dropped} rows with missing {col}")
    
    # Numeric columns - fill with 0 (casualties, coordinates if missing)
    numeric_fill_cols = ['nkill', 'nwound']
    for col in numeric_fill_cols:
        if col in df.columns and df[col].isnull().any():
            filled = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            print(f"  Filled {filled} missing values in {col} with 0")
    
    # Categorical columns - fill with 'Unknown'
    categorical_cols = ['gname', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            filled = df[col].isnull().sum()
            df[col] = df[col].fillna('Unknown')
            print(f"  Filled {filled} missing values in {col} with 'Unknown'")
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values: {missing_before} -> {missing_after}")
    
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with corrected data types
    """
    print("\nFixing data types...")
    
    # Convert date components to integers
    date_cols = ['iyear', 'imonth', 'iday']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Convert casualty columns to numeric
    casualty_cols = ['nkill', 'nwound']
    for col in casualty_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert coordinates to float
    coord_cols = ['latitude', 'longitude']
    for col in coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("Data types corrected")
    
    return df


def create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proper datetime column from year, month, day.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with date column added
    """
    print("\nCreating date column...")
    
    if all(col in df.columns for col in ['iyear', 'imonth', 'iday']):
        # Handle invalid dates (e.g., day=0)
        df['iday'] = df['iday'].replace(0, 1)
        df['imonth'] = df['imonth'].replace(0, 1)
        
        # Create date column
        df['date'] = pd.to_datetime(
            df[['iyear', 'imonth', 'iday']].rename(
                columns={'iyear': 'year', 'imonth': 'month', 'iday': 'day'}
            ),
            errors='coerce'
        )
        
        # Remove rows with invalid dates
        invalid_dates = df['date'].isnull().sum()
        if invalid_dates > 0:
            print(f"  Removed {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['date'])
        
        print(f"Date column created: {df['date'].min()} to {df['date'].max()}")
    else:
        print("Warning: Date columns not found")
    
    return df


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean geographic coordinates.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with validated coordinates
    """
    print("\nValidating coordinates...")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        initial_len = len(df)
        
        # Valid latitude: -90 to 90
        # Valid longitude: -180 to 180
        invalid_coords = (
            (df['latitude'] < -90) | (df['latitude'] > 90) |
            (df['longitude'] < -180) | (df['longitude'] > 180)
        )
        
        # Set invalid coordinates to NaN
        df.loc[invalid_coords, ['latitude', 'longitude']] = np.nan
        
        invalid_count = invalid_coords.sum()
        if invalid_count > 0:
            print(f"  Set {invalid_count} invalid coordinate pairs to NaN")
    else:
        print("Coordinate columns not found")
    
    return df


def standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize text fields (trim whitespace, handle case).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized text
    """
    print("\nStandardizing text fields...")
    
    text_cols = ['country_txt', 'region_txt', 'provstate', 'city', 
                 'gname', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']
    
    for col in text_cols:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings with Unknown
            df[col] = df[col].replace('', 'Unknown')
            df[col] = df[col].replace('nan', 'Unknown')
    
    print("Text fields standardized")
    
    return df


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns needed for analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with selected columns
    """
    print("\nSelecting relevant columns...")
    
    # Define columns to keep
    keep_cols = [
        'eventid', 'date', 'iyear', 'imonth', 'iday',
        'country_txt', 'region_txt', 'provstate', 'city',
        'latitude', 'longitude',
        'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt',
        'gname', 'nkill', 'nwound',
        'summary'
    ]
    
    # Only keep columns that exist
    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols]

    
    return df


def clean_data(config: dict) -> pd.DataFrame:
    """
    Main cleaning pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
    """
    print("="*60)
    print("STARTING DATA CLEANING PIPELINE")
    print("="*60)
    
    # Load raw data
    df = load_raw_data(config)
    print(f"\nInitial shape: {df.shape}")
    
    # Apply cleaning steps
    df = remove_duplicates(df)
    df = fix_data_types(df)
    df = create_date_column(df)
    df = handle_missing_values(df)
    df = validate_coordinates(df)
    df = standardize_text_fields(df)
    df = select_relevant_columns(df)
    
    print(f"\nFinal shape: {df.shape}")

    print("\n" + "="*60)
    print("DATA CLEANING COMPLETE")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Clean the data
    df_clean = clean_data(config)
    
    # Save cleaned data
    save_processed_data(df_clean, "01_cleaned_data.csv", config)

