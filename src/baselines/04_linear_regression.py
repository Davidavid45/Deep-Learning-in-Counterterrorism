"""
Linear Regression Baseline with Lag Features

Uses lagged values as predictors in a simple linear model.
Shows if linear relationships exist in the data.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config


class LaggedLinearRegression:
    """
    Linear Regression with lagged features baseline.
    """
    
    def __init__(self, n_lags: int = 4):
        """
        Initialize Linear Regression model.
        
        Args:
            n_lags: Number of lag features to create
        """
        self.n_lags = n_lags
        self.model = None
        self.last_values = None
    
    def create_features(self, y: np.ndarray) -> pd.DataFrame:
        """
        Create lagged features from time series.
        
        Args:
            y: Time series data
            
        Returns:
            DataFrame with lag features
        """
        df = pd.DataFrame({'y': y})
        
        # Create lag features
        for i in range(1, self.n_lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        
        # Drop rows with NaN (initial lags)
        df = df.dropna()
        
        return df
    
    def fit(self, y_train: np.ndarray) -> 'LaggedLinearRegression':
        """
        Fit Linear Regression model.
        
        Args:
            y_train: Training data
            
        Returns:
            self
        """
        from sklearn.linear_model import LinearRegression
        
        # Create features
        df = self.create_features(y_train)
        
        X = df.drop('y', axis=1).values
        y = df['y'].values
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Store last values for prediction
        self.last_values = y_train[-self.n_lags:].copy()
        
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
        current_lags = list(self.last_values)
        
        for _ in range(n_periods):
            # Create feature vector from last n_lags values
            X = np.array(current_lags[-self.n_lags:]).reshape(1, -1)
            
            # Predict next value
            pred = self.model.predict(X)[0]
            predictions.append(pred)
            
            # Update lags with prediction
            current_lags.append(pred)
        
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
            'model': 'Linear Regression',
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


def run_linear_regression(y_train: np.ndarray, y_test: np.ndarray,
                         n_lags: int = 4) -> tuple:
    """
    Train and evaluate Linear Regression baseline.
    
    Args:
        y_train: Training data
        y_test: Test data
        n_lags: Number of lag features
        
    Returns:
        Predictions and metrics dictionary
    """
    print(f"\nRunning Linear Regression (n_lags={n_lags})...")
    
    # Initialize and fit model
    model = LaggedLinearRegression(n_lags=n_lags)
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
    print("LINEAR REGRESSION BASELINE")
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
    
    # Run model with different lag counts
    lag_options = [4, 8, 12]
    results = []
    
    for n_lags in lag_options:
        y_pred, metrics = run_linear_regression(y_train, y_test, n_lags=n_lags)
        results.append(metrics)
    
    # Show comparison
    print("\n" + "="*60)
    print("LAG COUNT COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("LINEAR REGRESSION COMPLETE")
    print("="*60)
