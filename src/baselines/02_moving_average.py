"""
Moving Average Baseline Model

Predicts by averaging the last N observations.
Simple smoothing baseline that should be easy to beat.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config


class MovingAverage:
    """
    Moving Average baseline: Predicts average of last N observations.
    """
    
    def __init__(self, window: int = 4):
        """
        Initialize Moving Average model.
        
        Args:
            window: Number of previous observations to average
        """
        self.window = window
        self.history = None
    
    def fit(self, y_train: np.ndarray) -> 'MovingAverage':
        """
        Fit the model (store training history).
        
        Args:
            y_train: Training data
            
        Returns:
            self
        """
        self.history = list(y_train.copy())
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
            # Average of last 'window' observations
            window_data = self.history[-self.window:]
            pred = np.mean(window_data)
            predictions.append(pred)
            
            # Append prediction to history for next iteration
            self.history.append(pred)
        
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
            'model': 'Moving Average',
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


def run_moving_average(y_train: np.ndarray, y_test: np.ndarray, 
                      window: int = 4) -> tuple:
    """
    Train and evaluate Moving Average baseline.
    
    Args:
        y_train: Training data
        y_test: Test data
        window: Moving average window size
        
    Returns:
        Predictions and metrics dictionary
    """
    print(f"\nRunning Moving Average (window={window})...")
    
    # Initialize and fit model
    model = MovingAverage(window=window)
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
    
    print("="*60)
    print("MOVING AVERAGE BASELINE")
    print("="*60)
    
    # Generate dummy data for testing
    np.random.seed(config['seed'])
    n = 200
    time = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * time / 52)
    trend = 0.1 * time
    noise = np.random.normal(0, 2, n)
    y = seasonal + trend + noise + 50
    
    # Split data
    train_size = int(n * config['split']['train_ratio'])
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"\nTrain size: {len(y_train)}")
    print(f"Test size:  {len(y_test)}")
    
    # Run model with different windows
    windows = [4, 8, 12]
    results = []
    
    for window in windows:
        y_pred, metrics = run_moving_average(y_train, y_test, window=window)
        results.append(metrics)
    
    # Show comparison
    print("\n" + "="*60)
    print("WINDOW SIZE COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("MOVING AVERAGE COMPLETE")
    print("="*60)
