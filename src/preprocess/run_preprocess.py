"""
Preprocessing Pipeline Runner

Runs all preprocessing steps in order:
1. Data cleaning
2. Feature engineering  
3. Time series aggregation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import ensure_directories


def main():
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Ensure directories exist
    print("Setting up directories...")
    ensure_directories()
    print("✅ Directories ready\n")
    
    # Get the directory where preprocessing scripts are
    preprocess_dir = Path(__file__).parent
    
    # Preprocessing scripts in order
    scripts = [
        ('00_clean.py', 'Data Cleaning'),
        ('01_features.py', 'Feature Engineering'),
        ('02_aggregate.py', 'Time Series Aggregation'),
    ]
    
    for i, (script, name) in enumerate(scripts, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(scripts)}] {name}")
        print('='*60 + "\n")
        
        script_path = preprocess_dir / script
        result = os.system(f'python {script_path}')
        
        if result != 0:
            print(f"\n❌ Error in {script}")
            print("Stopping preprocessing.")
            sys.exit(1)
        
        print(f"\n✅ Completed: {name}")
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print("\nOutput files created:")
    print("  - 01_cleaned_data.csv")
    print("  - 02_featured_data.csv")
    print("  - 03_weekly_aggregated.csv")



if __name__ == "__main__":
    main()
