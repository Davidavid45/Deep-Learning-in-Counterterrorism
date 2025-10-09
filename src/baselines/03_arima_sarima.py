"""
ARIMA/SARIMA Baseline Models

Traditional statistical time series models.
SARIMA is the strongest baseline - LSTM must beat this to justify complexity.
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config


class ARIMABaseline:
    """
    ARIMA/SARIMA baseline using statsmodels.
    """
    
    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = None):
        """
        Initialize ARIMA model.
        
        Args:
            order: (p, d, q) for ARIMA
            seasonal_order: (P, D, Q, s) for SARIMA, None for non-seasonal
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
    
    def fit(self, y_train: np.ndarray) -> 'ARIMABaseline':
        """
        Fit ARIMA/SARIMA model.
        
        Args:
            y_train: Training data
            
        Returns:
            self
        """
        try:
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(
                    y_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA
                self.model = ARIMA(y_train, order=self.order)
            
            self.model_fit = self.model.fit(disp=False)
            
        except Exception as e:
            print(f"  Warning: ARIMA fitting failed: {e}")
            # Fallback to simpler model
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y_train, order=(1, 0, 0))
            self.model_fit = self.model.fit(disp=False)
        
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """
        Make predictions for n_periods ahead.
        
        Args:
            n_periods: Number of periods to predict
            
        Returns:
            Array of predictions
        """
        forecast = self.model_fit.forecast(steps=n_periods)
        return np.array(forecast)
    
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
        
        model_name = 'SARIMA' if self.seasonal_order else 'ARIMA'
        
        return {
            'model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


def run_arima(y_train: np.ndarray, y_test: np.ndarray,
              order: tuple = (1, 1, 1), seasonal_order: tuple = None) -> tuple:
    """
    Train and evaluate ARIMA/SARIMA baseline.
    
    Args:
        y_train: Training data
        y_test: Test data
        order: (p, d, q) for ARIMA
        seasonal_order: (P, D, Q, s) for SARIMA
        
    Returns:
        Predictions and metrics dictionary
    """
    model_name = 'SARIMA' if seasonal_order else 'ARIMA'
    print(f"\nRunning {model_name}...")
    
    # Initialize and fit model
    model = ARIMABaseline(order=order, seasonal_order=seasonal_order)
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
    
    print("="*60)
    print("ARIMA/SARIMA BASELINE")
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
    
    results = []
    
    # 1. Simple ARIMA
    try:
        y_pred, metrics = run_arima(y_train, y_test, order=(1, 1, 1))
        results.append(metrics)
    except Exception as e:
        print(f"  ARIMA failed: {e}")
    
    # 2. SARIMA with seasonality
    try:
        # Note: seasonal_lag might be too large for small datasets
        # Using a smaller period for testing
        seasonal_period = min(seasonal_lag, 12)
        y_pred, metrics = run_arima(
            y_train, y_test,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period)
        )
        results.append(metrics)
    except Exception as e:
        print(f"  SARIMA failed: {e}")
    
    # Show comparison
    if results:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("ARIMA/SARIMA COMPLETE")
    print("="*60)
    print("\nNote: For production use, tune hyperparameters using grid search")
