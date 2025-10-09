"""
Master Runner for Ablation Studies

Executes all ablation experiments and generates comprehensive reports.

Usage:
    python src/ablations/run_ablations.py
    
    # Run specific ablation only:
    python src/ablations/run_ablations.py --feature-only
    python src/ablations/run_ablations.py --architecture-only
    python src/ablations/run_ablations.py --data-only
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_config, ensure_directories


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_prerequisites():
    """Check if required data files exist"""
    config = load_config()
    data_dir = config['data']['data_dir']
    weekly_file = os.path.join(data_dir, '03_weekly_aggregated.csv')
    
    if not os.path.exists(weekly_file):
        print("\n❌ Error: Preprocessed data not found")
        print(f"Expected: {weekly_file}")
        print("\nPlease run preprocessing first:")
        print("  python src/preprocess/run_preprocess.py")
        return False
    
    print(f"✓ Found preprocessed data: {weekly_file}")
    return True


def run_feature_ablations():
    """Run feature ablation study"""
    print_banner("RUNNING FEATURE ABLATIONS")
    
    from src.ablations.feature_ablations import main as feature_main
    
    try:
        feature_main()
        return True
    except Exception as e:
        print(f"\n❌ Feature ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_architecture_ablations():
    """Run architecture ablation study"""
    print_banner("RUNNING ARCHITECTURE ABLATIONS")
    
    from src.ablations.architecture_ablations import main as architecture_main
    
    try:
        architecture_main()
        return True
    except Exception as e:
        print(f"\n❌ Architecture ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_data_ablations():
    """Run data configuration ablation study"""
    print_banner("RUNNING DATA CONFIGURATION ABLATIONS")
    
    from src.ablations.data_ablations import main as data_main
    
    try:
        data_main()
        return True
    except Exception as e:
        print(f"\n❌ Data ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary_report():
    """Generate combined summary of all ablation results"""
    print_banner("GENERATING SUMMARY REPORT")
    
    tables_dir = Path(project_root) / 'reports' / 'tables'
    
    feature_file = tables_dir / 'feature_ablation_results.csv'
    architecture_file = tables_dir / 'architecture_ablation_results.csv'
    data_file = tables_dir / 'data_ablation_results.csv'
    
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("ABLATION STUDIES SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append("")
    
    # Feature ablation summary
    if feature_file.exists():
        summary_lines.append("FEATURE ABLATION RESULTS")
        summary_lines.append("-"*60)
        
        df = pd.read_csv(feature_file)
        baseline = df[df['Feature Group'].str.contains('Baseline', case=False, na=False)]
        
        if not baseline.empty:
            baseline_rmse = baseline['RMSE'].values[0]
            summary_lines.append(f"Baseline RMSE: {baseline_rmse:.4f}")
            summary_lines.append("")
            
            # Top impactful features
            df_sorted = df[~df['Feature Group'].str.contains('Baseline', case=False, na=False)]
            df_sorted = df_sorted.sort_values('RMSE_Impact', ascending=False)
            
            summary_lines.append("Top 5 Most Impactful Feature Groups:")
            for idx, (i, row) in enumerate(df_sorted.head(5).iterrows(), 1):
                impact = row['RMSE_Impact']
                summary_lines.append(f"  {idx}. {row['Feature Group']}: {impact:+.4f} RMSE impact")
            
            summary_lines.append("")
        
        summary_lines.append(f"Full results: {feature_file}")
        summary_lines.append("")
    else:
        summary_lines.append("⚠️  Feature ablation results not found")
        summary_lines.append("")
    
    # Architecture ablation summary
    if architecture_file.exists():
        summary_lines.append("ARCHITECTURE ABLATION RESULTS")
        summary_lines.append("-"*60)
        
        df = pd.read_csv(architecture_file)
        df_sorted = df.sort_values('RMSE')
        
        summary_lines.append("Top 5 Best Architectures:")
        for idx, (i, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            arch_name = row['Architecture']
            rmse = row['RMSE']
            mae = row['MAE']
            summary_lines.append(f"  {idx}. {arch_name}")
            summary_lines.append(f"     RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        summary_lines.append("")
        summary_lines.append(f"Full results: {architecture_file}")
        summary_lines.append("")
    else:
        summary_lines.append("⚠️  Architecture ablation results not found")
        summary_lines.append("")
    
    # Data configuration ablation summary
    if data_file.exists():
        summary_lines.append("DATA CONFIGURATION ABLATION RESULTS")
        summary_lines.append("-"*60)
        
        df = pd.read_csv(data_file)
        
        summary_lines.append("Data Configuration Results:")
        for idx, (i, row) in enumerate(df.iterrows(), 1):
            config = row['Configuration']
            rmse = row['RMSE']
            samples = row.get('Samples', 'N/A')
            summary_lines.append(f"  {idx}. {config}")
            summary_lines.append(f"     RMSE: {rmse:.4f}, Samples: {samples}")
        
        summary_lines.append("")
        summary_lines.append(f"Full results: {data_file}")
        summary_lines.append("")
    else:
        summary_lines.append("⚠️  Data configuration ablation results not found")
        summary_lines.append("")
    

    # Print summary
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary
    summary_file = tables_dir / 'ablation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main runner"""
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--feature-only', action='store_true', 
                       help='Run only feature ablations')
    parser.add_argument('--architecture-only', action='store_true',
                       help='Run only architecture ablations')
    parser.add_argument('--data-only', action='store_true',
                       help='Run only data configuration ablations')
    args = parser.parse_args()
    
    print_banner("ABLATION STUDIES - MASTER RUNNER")
    print("Deep Learning in Counterterrorism Project")
    print("\nThis will run ablation experiments to evaluate:")
    print("  • Feature group contributions")
    print("  • Architecture configurations")
    print("  • Data configuration choices")
    print("\n⏱️  Estimated time: 4-8 hours total (can run overnight)")
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Ensure output directories
    ensure_directories(['reports/tables'])
    
    # Run ablations
    success_count = 0
    total_count = 0
    
    if not args.architecture_only and not args.data_only:
        total_count += 1
        if run_feature_ablations():
            success_count += 1
    
    if not args.feature_only and not args.data_only:
        total_count += 1
        if run_architecture_ablations():
            success_count += 1
    
    if not args.feature_only and not args.architecture_only:
        total_count += 1
        if run_data_ablations():
            success_count += 1
    
    # Generate summary
    if success_count > 0:
        generate_summary_report()
    
    # Final status
    print_banner("ABLATION STUDIES COMPLETE")
    print(f"Successfully completed: {success_count}/{total_count} studies")
    
    if success_count < total_count:
        print("\n⚠️  Some ablation studies failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All ablation studies completed successfully!")
        print("\nResults available in:")
        print("  • reports/tables/feature_ablation_results.csv")
        print("  • reports/tables/architecture_ablation_results.csv")
        print("  • reports/tables/data_ablation_results.csv")
        print("  • reports/tables/ablation_summary.txt")


if __name__ == "__main__":
    main()
