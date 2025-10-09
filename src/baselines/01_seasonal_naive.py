"""
Seasonal Naive Baseline Model

Predicts the value from the same period in the previous year.
For weekly data: uses 52-week lag (same week last year).
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config


class SeasonalNaive:
    """
    Seasonal Naive baseline: Predicts value from same period last year.
    """
    
    def __init__(self, seasonal_period: int = 52):
        """
        Initialize Seasonal Naive model.
        
        Args:
            seasonal_period: Number of periods in a season (52 for weekly, 12 for monthly)
        """
        self.seasonal_period = seasonal_period
        self.history = None
    
    def fit(self, y_train: np.ndarray) -> 'SeasonalNaive':
        """
        Fit the model (store training history).
        
        Args:
            y_train: Training data
            
        Returns:
            self
        """
        self.history = y_train.copy()
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """
        Make predictions for n_periods ahead.
        
        Args:
            n_periods: Number of periods to predict
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        for i in range(n_periods):
            # Look back seasonal_period from current position
            idx = len(self.history) - self.seasonal_period + i
            
            if idx >= 0 and idx < len(self.history):
                predictions.append(self.history[idx])
            else:
                # Fallback to mean if not enough history
                predictions.append(self.history.mean())
        
        return np.array(predictions)
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'model': 'Seasonal Naive',
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


def run_seasonal_naive(y_train: np.ndarray, y_test: np.ndarray, 
                       seasonal_period: int = 52) -> tuple:
    """
    Train and evaluate Seasonal Naive baseline.
    
    Args:
        y_train: Training data
        y_test: Test data
        seasonal_period: Seasonal period (52 for weekly, 12 for monthly)
        
    Returns:
        Predictions and metrics dictionary
    """
    print(f"\nRunning Seasonal Naive (period={seasonal_period})...")
    
    # Initialize and fit model
    model = SeasonalNaive(seasonal_period=seasonal_period)
    model.fit(y_train)
    
    # Make predictions
    y_pred = model.predict(len(y_test))
    
    # Calculate metrics
    metrics = model.score(y_test, y_pred)
    
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    return y_pred, metrics


if __name__ == "__main__":
    # Load config
    config = load_config()
    seasonal_lag = config['model']['seasonal_lag_weeks']
    
    # Load data (example with dummy data)
    print("="*60)
    print("SEASONAL NAIVE BASELINE")
    print("="*60)
    
    # Generate dummy seasonal data for testing
    np.random.seed(config['seed'])
    n = 200
    time = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * time / seasonal_lag)
    trend = 0.1 * time
    noise = np.random.normal(0, 2, n)
    y = seasonal + trend + noise + 50
    
    # Split data
    train_size = int(n * config['split']['train_ratio'])
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"\nTrain size: {len(y_train)}")
    print(f"Test size:  {len(y_test)}")
    
    # Run model
    y_pred, metrics = run_seasonal_naive(y_train, y_test, seasonal_period=seasonal_lag)
    
    print("\n" + "="*60)
    print("SEASONAL NAIVE COMPLETE")
    print("="*60)
