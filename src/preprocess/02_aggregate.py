"""
Time Series Aggregation Script for Terrorism Data

This script aggregates incident-level data to time series format:
- Daily aggregation
- Weekly aggregation (primary for LSTM)
- Monthly aggregation
- Configurable grain: region, country, or attack type
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_processed_data, save_processed_data


def aggregate_daily(df: pd.DataFrame, grain: str = 'region_txt') -> pd.DataFrame:
    """
    Aggregate data to daily level.
    
    Args:
        df: Input DataFrame
        grain: Aggregation grain (region_txt, country_txt, attacktype1_txt)
        
    Returns:
        Daily aggregated DataFrame
    """
    print(f"\nAggregating to daily level by {grain}...")
    
    # Group by date and grain
    agg_dict = {
        'eventid': 'count',
        'casualties': 'sum',
        'nkill': 'sum',
        'nwound': 'sum'
    }
    
    df_daily = df.groupby(['date', grain]).agg(agg_dict).reset_index()
    
    # Rename columns
    df_daily = df_daily.rename(columns={
        'eventid': 'attack_count',
        'casualties': 'total_casualties',
        'nkill': 'total_killed',
        'nwound': 'total_wounded'
    })
    
    print(f"  Daily records: {len(df_daily):,}")
    print(f"  Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    
    return df_daily


def aggregate_weekly(df: pd.DataFrame, grain: str = 'region_txt') -> pd.DataFrame:
    """
    Aggregate data to weekly level.
    
    Args:
        df: Input DataFrame
        grain: Aggregation grain (region_txt, country_txt, attacktype1_txt)
        
    Returns:
        Weekly aggregated DataFrame
    """
    print(f"\nAggregating to weekly level by {grain}...")
    
    # Create week start date
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
    
    # Group by week and grain
    agg_dict = {
        'eventid': 'count',
        'casualties': 'sum',
        'nkill': 'sum',
        'nwound': 'sum',
        'date': 'min'  # Keep earliest date in week
    }
    
    df_weekly = df.groupby(['week_start', grain]).agg(agg_dict).reset_index()
    
    # Rename columns
    df_weekly = df_weekly.rename(columns={
        'eventid': 'attack_count',
        'casualties': 'total_casualties',
        'nkill': 'total_killed',
        'nwound': 'total_wounded',
        'date': 'first_attack_date'
    })
    
    # Add week number and year
    df_weekly['year'] = df_weekly['week_start'].dt.year
    df_weekly['week'] = df_weekly['week_start'].dt.isocalendar().week
    
    print(f"  Weekly records: {len(df_weekly):,}")
    print(f"  Date range: {df_weekly['week_start'].min()} to {df_weekly['week_start'].max()}")
    
    return df_weekly


def aggregate_monthly(df: pd.DataFrame, grain: str = 'region_txt') -> pd.DataFrame:
    """
    Aggregate data to monthly level.
    
    Args:
        df: Input DataFrame
        grain: Aggregation grain (region_txt, country_txt, attacktype1_txt)
        
    Returns:
        Monthly aggregated DataFrame
    """
    print(f"\nAggregating to monthly level by {grain}...")
    
    # Create month start date
    df['month_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Group by month and grain
    agg_dict = {
        'eventid': 'count',
        'casualties': 'sum',
        'nkill': 'sum',
        'nwound': 'sum'
    }
    
    df_monthly = df.groupby(['month_start', grain]).agg(agg_dict).reset_index()
    
    # Rename columns
    df_monthly = df_monthly.rename(columns={
        'eventid': 'attack_count',
        'casualties': 'total_casualties',
        'nkill': 'total_killed',
        'nwound': 'total_wounded'
    })
    
    # Add month and year
    df_monthly['year'] = df_monthly['month_start'].dt.year
    df_monthly['month'] = df_monthly['month_start'].dt.month
    
    print(f"  Monthly records: {len(df_monthly):,}")
    print(f"  Date range: {df_monthly['month_start'].min()} to {df_monthly['month_start'].max()}")
    
    return df_monthly


def fill_missing_periods(df: pd.DataFrame, date_col: str, grain: str, 
                        freq: str = 'W') -> pd.DataFrame:
    """
    Fill in missing time periods with zeros.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        grain: Grain column name
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        DataFrame with all periods filled
    """
    print(f"\nFilling missing {freq} periods...")
    
    # Get all unique grain values
    grain_values = df[grain].unique()
    
    # Create date range
    date_range = pd.date_range(
        start=df[date_col].min(),
        end=df[date_col].max(),
        freq=freq
    )
    
    # Create complete index
    complete_index = pd.MultiIndex.from_product(
        [date_range, grain_values],
        names=[date_col, grain]
    )
    
    # Reindex and fill missing values with 0
    df = df.set_index([date_col, grain])
    df = df.reindex(complete_index, fill_value=0)
    df = df.reset_index()
    
    print(f"  Total records after filling: {len(df):,}")
    
    return df


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Add time-based features to aggregated data.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        
    Returns:
        DataFrame with time features
    """
    print("\nAdding time features...")
    
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_year'] = df[date_col].dt.dayofyear
    
    if 'week' not in df.columns:
        df['week'] = df[date_col].dt.isocalendar().week
    
    print("  Added: year, month, quarter, day_of_year, week")
    
    return df


def add_seasonal_lags(df: pd.DataFrame, target_col: str = 'attack_count',
                     seasonal_periods: int = 52) -> pd.DataFrame:
    """
    Add seasonal lag features (e.g., same week last year).
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags for
        seasonal_periods: Number of periods for seasonal lag (52 weeks = 1 year)
        
    Returns:
        DataFrame with seasonal lag features
    """
    print(f"\nAdding seasonal lags (period={seasonal_periods})...")
    
    # Sort by grain and date
    date_col = [col for col in df.columns if 'start' in col or col == 'date'][0]
    grain_col = [col for col in df.columns if col in ['region_txt', 'country_txt', 'attacktype1_txt']][0]
    
    df = df.sort_values([grain_col, date_col])
    
    # Create seasonal lag
    lag_col_name = f'{target_col}_lag_{seasonal_periods}'
    df[lag_col_name] = df.groupby(grain_col)[target_col].shift(seasonal_periods)
    
    print(f"  Created: {lag_col_name}")
    
    return df


def calculate_rolling_statistics(df: pd.DataFrame, target_col: str = 'attack_count',
                                 windows: list = [4, 12]) -> pd.DataFrame:
    """
    Calculate rolling statistics (mean, std, etc.).
    
    Args:
        df: Input DataFrame
        target_col: Column to calculate statistics for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling statistics
    """
    print(f"\nCalculating rolling statistics (windows: {windows})...")
    
    date_col = [col for col in df.columns if 'start' in col or col == 'date'][0]
    grain_col = [col for col in df.columns if col in ['region_txt', 'country_txt', 'attacktype1_txt']][0]
    
    df = df.sort_values([grain_col, date_col])
    
    for window in windows:
        # Rolling mean
        mean_col = f'{target_col}_rolling_mean_{window}'
        df[mean_col] = df.groupby(grain_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        std_col = f'{target_col}_rolling_std_{window}'
        df[std_col] = df.groupby(grain_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    print(f"  Created rolling features for {len(windows)} windows")
    
    return df


def aggregate_data(config: Dict[str, Any]) -> tuple:
    """
    Main aggregation pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (daily_df, weekly_df, monthly_df)
    """
    print("="*60)
    print("STARTING TIME SERIES AGGREGATION PIPELINE")
    print("="*60)
    
    # Load featured data
    df = load_processed_data("02_featured_data.csv", config)
    print(f"\nInitial shape: {df.shape}")
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get aggregation grain from config
    grain = config['aggregation']['weekly_grain']
    if grain == 'region':
        grain_col = 'region_txt'
    elif grain == 'country':
        grain_col = 'country_txt'
    elif grain == 'attacktype':
        grain_col = 'attacktype1_txt'
    else:
        grain_col = 'region_txt'
    
    print(f"Aggregation grain: {grain_col}")
    
    # Create aggregations
    df_daily = aggregate_daily(df, grain_col)
    df_weekly = aggregate_weekly(df, grain_col)
    df_monthly = aggregate_monthly(df, grain_col)
    
    # Fill missing periods for weekly data (primary for LSTM)
    df_weekly = fill_missing_periods(df_weekly, 'week_start', grain_col, freq='W-MON')
    
    # Add time features
    df_weekly = add_time_features(df_weekly, 'week_start')
    
    # Add seasonal lags (from config)
    seasonal_lag = config['model']['seasonal_lag_weeks']
    df_weekly = add_seasonal_lags(df_weekly, 'attack_count', seasonal_lag)
    
    # Add rolling statistics
    df_weekly = calculate_rolling_statistics(df_weekly, 'attack_count')
    
    print("\n" + "="*60)
    print("TIME SERIES AGGREGATION COMPLETE")
    print("="*60)
    
    return df_daily, df_weekly, df_monthly


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Aggregate data
    df_daily, df_weekly, df_monthly = aggregate_data(config)
    
    # Save aggregated data
    save_processed_data(df_daily, "03_daily_aggregated.csv", config)
    save_processed_data(df_weekly, "03_weekly_aggregated.csv", config)
    save_processed_data(df_monthly, "03_monthly_aggregated.csv", config)
    

