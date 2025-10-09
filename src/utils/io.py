import os
import yaml
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw terrorism data from CSV.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with raw data
    """
    data_dir = config['data']['data_dir']
    raw_file = config['data']['raw_file']
    encoding = config['data']['encoding']
    
    filepath = os.path.join(data_dir, raw_file)
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    
    print(f"Loaded {len(df)} rows from {raw_file}")
    return df


def save_processed_data(df: pd.DataFrame, 
                       filename: str,
                       config: Dict[str, Any]) -> None:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        config: Configuration dictionary
    """
    data_dir = config['data']['data_dir']
    filepath = os.path.join(data_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filename}")


def load_processed_data(filename: str,
                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load processed data from CSV.
    
    Args:
        filename: Input filename
        config: Configuration dictionary
        
    Returns:
        DataFrame with processed data
    """
    data_dir = config['data']['data_dir']
    filepath = os.path.join(data_dir, filename)
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filename}")
    return df


def save_model(model, filename: str, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        filename: Output filename
        config: Optional configuration dictionary
    """
    # Create models directory if it doesn't exist
    models_dir = 'src/models'
    os.makedirs(models_dir, exist_ok=True)
    
    filepath = os.path.join(models_dir, filename)
    
    # Save based on model type
    if hasattr(model, 'save'):  # Keras/TensorFlow model
        model.save(filepath)
    else:  # Scikit-learn or other pickle-able models
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filename: str):
    """
    Load trained model from disk.
    
    Args:
        filename: Model filename
        
    Returns:
        Loaded model object
    """
    from tensorflow import keras
    
    filepath = os.path.join('src/models', filename)
    
    # Try loading as Keras model first
    try:
        model = keras.models.load_model(filepath)
    except:
        # Otherwise load as pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def save_predictions(predictions: np.ndarray,
                    actual: np.ndarray,
                    filename: str,
                    config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save predictions alongside actual values.
    
    Args:
        predictions: Model predictions
        actual: Actual values
        filename: Output filename
        config: Optional configuration dictionary
    """
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predictions
    })
    
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")


def save_metrics(metrics: Dict[str, float],
                filename: str,
                config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save evaluation metrics to file.
    
    Args:
        metrics: Dictionary of metric names and values
        filename: Output filename
        config: Optional configuration dictionary
    """
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"Metrics saved to {filepath}")


def save_figure(fig, filename: str, dpi: int = 300) -> None:
    """
    Save matplotlib figure to reports/figures directory.
    
    Args:
        fig: Matplotlib figure object or filename string
        filename: Output filename (e.g., 'lstm_loss.png') - used if fig is a figure object
        dpi: Resolution for publication (default 300)
    """
    import matplotlib.pyplot as plt
    
    figures_dir = 'reports/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Handle both old (filename only) and new (fig, filename) signatures
    if isinstance(fig, str):
        # Old signature: save_figure('filename.png')
        filepath = os.path.join(figures_dir, fig)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    else:
        # New signature: save_figure(fig, 'filename.png')
        filepath = os.path.join(figures_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    print(f"Figure saved to {filepath}")


def save_table(df: pd.DataFrame, filename: str, format: str = 'csv') -> None:
    """
    Save DataFrame as table to reports/tables directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename (without extension)
        format: 'csv', 'latex', or 'both'
    """
    tables_dir = 'reports/tables'
    os.makedirs(tables_dir, exist_ok=True)
    
    # Remove extension if provided
    base_name = filename.split('.')[0]
    
    if format in ['csv', 'both']:
        filepath = os.path.join(tables_dir, f'{base_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"Table saved to {filepath}")
    
    if format in ['latex', 'both']:
        filepath = os.path.join(tables_dir, f'{base_name}.tex')
        df.to_latex(filepath, index=False)
        print(f"Table saved to {filepath}")


def ensure_directories(directories: list = None) -> None:
    """
    Create necessary project directories if they don't exist.
    
    Args:
        directories: List of directory paths to create. If None, uses defaults.
    """
    if directories is None:
        directories = [
            'src/models',
            'results',
            'reports/figures',
            'reports/tables'
        ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories ready")


if __name__ == "__main__":
    # Test the functions
    config = load_config()
    print("Config loaded successfully")
    print(f"Data directory: {config['data']['data_dir']}")
    ensure_directories()