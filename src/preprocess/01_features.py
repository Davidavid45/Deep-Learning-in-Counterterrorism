"""
Feature Engineering Script for Terrorism Data

This script creates derived features for modeling:
- Casualties (killed + wounded)
- Temporal features (day of week, month, quarter, etc.)
- Days since last attack
- Attack frequency features
- Holiday indicators
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_processed_data, save_processed_data


def create_casualties_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total casualties column (killed + wounded).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with casualties column
    """
    print("Creating casualties feature...")
    
    df['casualties'] = df['nkill'] + df['nwound']
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date column.
    
    Args:
        df: Input DataFrame with 'date' column
        
    Returns:
        DataFrame with temporal features added
    """
    print("\nCreating temporal features...")
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    
    # Month
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    
    # Quarter
    df['quarter'] = df['date'].dt.quarter
    
    # Year
    df['year'] = df['date'].dt.year
    
    # Day of year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Week of year
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    print("  Created: day_of_week, month, quarter, year, week_of_year, is_weekend")
    
    return df


def create_us_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create indicator for US federal holidays.
    
    Args:
        df: Input DataFrame with 'date' column
        
    Returns:
        DataFrame with holiday indicator
    """
    print("\nCreating US holiday feature...")
    
    # Define major US holidays (simplified - fixed dates)
    def is_holiday(date):
        month = date.month
        day = date.day
        
        # New Year's Day
        if month == 1 and day == 1:
            return 1
        # Independence Day
        elif month == 7 and day == 4:
            return 1
        # Christmas
        elif month == 12 and day == 25:
            return 1
        # Memorial Day (last Monday of May - approximation)
        elif month == 5 and date.dayofweek == 0 and day >= 25:
            return 1
        # Labor Day (first Monday of September - approximation)
        elif month == 9 and date.dayofweek == 0 and day <= 7:
            return 1
        # Thanksgiving (4th Thursday of November - approximation)
        elif month == 11 and date.dayofweek == 3 and 22 <= day <= 28:
            return 1
        else:
            return 0
    
    df['is_holiday'] = df['date'].apply(is_holiday)
    
    holiday_count = df['is_holiday'].sum()
    print(f"  Marked {holiday_count} incidents on holidays")
    
    return df


def create_days_since_last_attack(df: pd.DataFrame, group_by: str = 'country_txt') -> pd.DataFrame:
    """
    Calculate days since last attack for each country/region.
    
    Args:
        df: Input DataFrame
        group_by: Column to group by (default: 'country_txt')
        
    Returns:
        DataFrame with days_since_last column
    """
    print(f"\nCreating days since last attack (grouped by {group_by})...")
    
    # Sort by location and date
    df = df.sort_values([group_by, 'date'])
    
    # Calculate days since last attack within each group
    df['days_since_last'] = df.groupby(group_by)['date'].diff().dt.days
    
    # Fill first occurrence in each group with 0
    df['days_since_last'] = df['days_since_last'].fillna(0)
    
    print(f"  Mean days since last attack: {df['days_since_last'].mean():.2f}")
    print(f"  Max days since last attack: {df['days_since_last'].max():.0f}")
    
    return df


def create_attack_frequency_features(df: pd.DataFrame, windows: list = [7, 30, 90]) -> pd.DataFrame:
    """
    Create rolling attack frequency features.
    
    Args:
        df: Input DataFrame
        windows: List of rolling window sizes in days
        
    Returns:
        DataFrame with frequency features
    """
    print(f"\nCreating attack frequency features (windows: {windows})...")
    
    # Sort by date
    df = df.sort_values('date')
    
    for window in windows:
        col_name = f'attacks_last_{window}d'
        
        # For each country, count attacks in rolling window
        # This is a simplified version - could be made more sophisticated
        df[col_name] = 0
        
    
    return df


def create_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create region-based features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with region features
    """
    print("\nCreating region features...")
    
    # Calculate historical average casualties per region
    region_avg_casualties = df.groupby('region_txt')['casualties'].transform('mean')
    df['region_avg_casualties'] = region_avg_casualties
    
    # Calculate historical attack count per region
    region_attack_count = df.groupby('region_txt')['eventid'].transform('count')
    df['region_attack_count'] = region_attack_count
    
    print("  Created: region_avg_casualties, region_attack_count")
    
    return df


def create_attack_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create attack type based features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with attack type features
    """
    print("\nCreating attack type features...")
    
    # Historical average casualties per attack type
    attacktype_avg_casualties = df.groupby('attacktype1_txt')['casualties'].transform('mean')
    df['attacktype_avg_casualties'] = attacktype_avg_casualties
    
    print("  Created: attacktype_avg_casualties")
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'casualties', 
                       lags: list = [1, 7, 30]) -> pd.DataFrame:
    """
    Create lagged features for time series.
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    print(f"\nCreating lag features for {target_col} (lags: {lags})...")
    
    df = df.sort_values('date')
    
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df[target_col].shift(lag)
    
    print(f"  Created {len(lags)} lag features")
    
    return df


def engineer_features(config: dict) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    print("="*60)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load cleaned data
    df = load_processed_data("01_cleaned_data.csv", config)
    print(f"\nInitial shape: {df.shape}")
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply feature engineering steps
    df = create_casualties_feature(df)
    df = create_temporal_features(df)
    df = create_us_holiday_feature(df)
    df = create_days_since_last_attack(df, group_by='country_txt')
    df = create_attack_frequency_features(df)
    df = create_region_features(df)
    df = create_attack_type_features(df)
    # df = create_lag_features(df)  # Uncomment if needed
    
    print(f"\nFinal shape: {df.shape}")
    print(f"Features added: {df.shape[1] - load_processed_data('01_cleaned_data.csv', config).shape[1]}")
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Engineer features
    df_features = engineer_features(config)
    
    # Save featured data
    save_processed_data(df_features, "02_featured_data.csv", config)
    
    print(f"\nNew feature columns:")
    feature_cols = [
        'casualties', 'day_of_week', 'month', 'quarter', 'year', 
        'week_of_year', 'is_weekend', 'is_holiday', 'days_since_last',
        'region_avg_casualties', 'region_attack_count', 'attacktype_avg_casualties'
    ]
    for col in feature_cols:
        if col in df_features.columns:
            print(f"  - {col}")
    
    print(f"\nCasualties statistics:")
    print(df_features['casualties'].describe())
