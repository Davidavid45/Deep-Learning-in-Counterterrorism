"""
Run All Baseline Models

Executes all baseline models and creates comparison table for paper.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, save_table

# Import baseline functions directly
import importlib.util

baseline_dir = os.path.dirname(os.path.abspath(__file__))

# Load seasonal_naive
spec = importlib.util.spec_from_file_location("seasonal_naive", os.path.join(baseline_dir, "01_seasonal_naive.py"))
sn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sn_module)
run_seasonal_naive = sn_module.run_seasonal_naive

# Load moving_average
spec = importlib.util.spec_from_file_location("moving_average", os.path.join(baseline_dir, "02_moving_average.py"))
ma_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma_module)
run_moving_average = ma_module.run_moving_average

# Load arima_sarima
spec = importlib.util.spec_from_file_location("arima_sarima", os.path.join(baseline_dir, "03_arima_sarima.py"))
ar_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar_module)
run_arima = ar_module.run_arima

# Load linear_regression
spec = importlib.util.spec_from_file_location("linear_regression", os.path.join(baseline_dir, "04_linear_regression.py"))
lr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr_module)
run_linear_regression = lr_module.run_linear_regression


def run_all_baselines(y_train: np.ndarray, y_test: np.ndarray, 
                      config: dict) -> pd.DataFrame:
    """
    Run all baseline models and return results.
    
    Args:
        y_train: Training data
        y_test: Test data
        config: Configuration dictionary
        
    Returns:
        DataFrame with all baseline results
    """
    results = []
    
    print("="*60)
    print("RUNNING ALL BASELINE MODELS")
    print("="*60)
    
    # 1. Seasonal Naive
    print("\n[1/4] Seasonal Naive Baseline")
    try:
        seasonal_lag = config['model']['seasonal_lag_weeks']
        _, metrics = run_seasonal_naive(y_train, y_test, seasonal_period=seasonal_lag)
        results.append(metrics)
    except Exception as e:
        print(f"  Error: {e}")
    
    # 2. Moving Average
    print("\n[2/4] Moving Average Baseline")
    try:
        _, metrics = run_moving_average(y_train, y_test, window=4)
        results.append(metrics)
    except Exception as e:
        print(f"  Error: {e}")
    
    # 3. ARIMA/SARIMA
    print("\n[3/4] ARIMA/SARIMA Baseline")
    try:
        seasonal_lag = config['model']['seasonal_lag_weeks']
        seasonal_period = min(seasonal_lag, 12)  # Use smaller period if needed
        _, metrics = run_arima(
            y_train, y_test,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period)
        )
        results.append(metrics)
    except Exception as e:
        print(f"  Error: {e}")
        # Try simple ARIMA if SARIMA fails
        try:
            _, metrics = run_arima(y_train, y_test, order=(1, 1, 1))
            results.append(metrics)
        except:
            print("  ARIMA also failed")
    
    # 4. Linear Regression
    print("\n[4/4] Linear Regression Baseline")
    try:
        _, metrics = run_linear_regression(y_train, y_test, n_lags=4)
        results.append(metrics)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    return results_df


def print_baseline_summary(results_df: pd.DataFrame):
    """Print formatted baseline results."""
    print("\n" + "="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    print("\nPerformance Metrics:")
    print(results_df.to_string(index=False))
    
    print("\n" + "-"*60)
    best_model = results_df.iloc[0]
    print(f"\n🏆 Best Baseline: {best_model['model']}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   MAE:  {best_model['MAE']:.4f}")
    print(f"   R²:   {best_model['R2']:.4f}")
    


if __name__ == "__main__":
    # Load config
    config = load_config()
    
    print("\n" + "="*60)
    print("BASELINE EVALUATION PIPELINE")
    print("="*60)
    
    # Load real preprocessed data
    data_dir = config['data']['data_dir']
    weekly_file = os.path.join(data_dir, '03_weekly_aggregated.csv')
    
    print(f"\nLoading data from: {weekly_file}")
    df = pd.read_csv(weekly_file)
    
    # Get the target variable (attack_count)
    y = df['attack_count'].values
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(y):,}")
    print(f"  Date range: {df['week_start'].min()} to {df['week_start'].max()}")
    print(f"  Mean attacks per week: {y.mean():.2f}")
    print(f"  Std attacks per week: {y.std():.2f}")
    
    # Chronological split (no shuffling for time series!)
    train_size = int(len(y) * config['split']['train_ratio'])
    test_start = int(len(y) * (config['split']['train_ratio'] + config['split']['val_ratio']))
    
    y_train = y[:train_size]
    y_test = y[test_start:]
    
    print(f"\nSplit Info:")
    print(f"  Training: {len(y_train):,} samples ({config['split']['train_ratio']:.0%})")
    print(f"  Validation: {test_start - train_size:,} samples ({config['split']['val_ratio']:.0%})")
    print(f"  Testing: {len(y_test):,} samples ({config['split']['test_ratio']:.0%})")
    
    # Run all baselines
    results_df = run_all_baselines(y_train, y_test, config)
    
    # Print summary
    print_baseline_summary(results_df)
    
    # Save results
    save_table(results_df, 'baseline_comparison', format='both')
    
    print("\n✅ Results saved to reports/tables/")
    print("   - baseline_comparison.csv")
    print("   - baseline_comparison.tex")
