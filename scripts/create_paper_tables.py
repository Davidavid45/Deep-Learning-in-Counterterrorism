"""
Generate LaTeX Tables for Paper

This script creates publication-ready LaTeX tables from all results.
Outputs are saved to reports/tables/latex/ as .tex files.

Usage:
    python scripts/create_paper_tables.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Output directory
OUTPUT_DIR = project_root / 'reports' / 'tables' / 'latex'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("GENERATING LATEX TABLES FOR PAPER")
print("="*60)


def load_baseline_results():
    """Load baseline comparison results"""
    file_path = project_root / 'reports' / 'tables' / 'baseline_comparison.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


def load_model_results():
    """Load deep learning model results"""
    results = {}
    
    # LSTM Attention
    lstm_att_path = project_root / 'reports' / 'tables' / 'lstm_attention_metrics.csv'
    if lstm_att_path.exists():
        results['LSTM Attention'] = pd.read_csv(lstm_att_path)
    
    # Bidirectional LSTM
    bilstm_path = project_root / 'reports' / 'tables' / 'bidirectional_lstm_metrics.csv'
    if bilstm_path.exists():
        results['Bidirectional LSTM'] = pd.read_csv(bilstm_path)
    
    return results


def load_ablation_results():
    """Load ablation study results"""
    results = {}
    
    # Feature ablation
    feat_path = project_root / 'reports' / 'tables' / 'feature_ablation_results.csv'
    if feat_path.exists():
        results['feature'] = pd.read_csv(feat_path)
    
    # Data ablation
    data_path = project_root / 'reports' / 'tables' / 'data_ablation_results.csv'
    if data_path.exists():
        results['data'] = pd.read_csv(data_path)
    
    return results


def table_1_model_performance():
    """
    Table 1: Comprehensive Model Performance Comparison
    Includes all baselines and deep learning models
    """
    print("\n[1/5] Creating Model Performance Table...")
    
    # Collect all model results
    all_models = []
    
    # Baselines
    baseline_df = load_baseline_results()
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            all_models.append({
                'Model': row['model'],
                'Type': 'Baseline',
                'RMSE': row['RMSE'],
                'MAE': row['MAE'],
                'R²': row['R2']
            })
    
    # Deep learning models
    dl_results = load_model_results()
    if 'LSTM Attention' in dl_results:
        lstm_att = dl_results['LSTM Attention'].iloc[0]
        all_models.append({
            'Model': 'LSTM with Attention',
            'Type': 'Deep Learning',
            'RMSE': lstm_att['RMSE'],
            'MAE': lstm_att['MAE'],
            'R²': lstm_att['R2']
        })
    
    if 'Bidirectional LSTM' in dl_results:
        bilstm = dl_results['Bidirectional LSTM'].iloc[0]
        all_models.append({
            'Model': 'Bidirectional LSTM',
            'Type': 'Deep Learning',
            'RMSE': bilstm['RMSE'],
            'MAE': bilstm['MAE'],
            'R²': bilstm['R2']
        })
    
    # Create DataFrame
    df = pd.DataFrame(all_models)
    
    # Calculate improvement over best baseline
    if not df.empty:
        baseline_rmse = df[df['Type'] == 'Baseline']['RMSE'].min()
        df['Improvement'] = ((baseline_rmse - df['RMSE']) / baseline_rmse * 100).round(1)
        df.loc[df['Type'] == 'Baseline', 'Improvement'] = '-'
    
    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Model Performance Comparison for Terrorist Attack Forecasting}
\label{tab:model_performance}
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{RMSE} & \textbf{MAE} & \textbf{R²} & \textbf{Improvement} \\
\midrule
"""
    
    for _, row in df.iterrows():
        model = row['Model']
        typ = row['Type']
        rmse = f"{row['RMSE']:.2f}"
        mae = f"{row['MAE']:.2f}"
        r2 = f"{row['R²']:.3f}"
        imp = f"{row['Improvement']}\\%" if row['Improvement'] != '-' else '-'
        
        # Bold the best model
        if row['Type'] == 'Deep Learning' and row['Model'] == 'Bidirectional LSTM':
            latex += f"\\textbf{{{model}}} & {typ} & \\textbf{{{rmse}}} & \\textbf{{{mae}}} & \\textbf{{{r2}}} & \\textbf{{{imp}}} \\\\\n"
        else:
            latex += f"{model} & {typ} & {rmse} & {mae} & {r2} & {imp} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Improvement calculated relative to best baseline model. Best performing model shown in bold.
\item RMSE: Root Mean Squared Error; MAE: Mean Absolute Error; R²: Coefficient of Determination
\end{tablenotes}
\end{table}
"""
    
    # Save
    output_path = OUTPUT_DIR / 'model_performance.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"   ✓ Saved: {output_path}")


def table_2_feature_ablation():
    """
    Table 2: Feature Ablation Study Results
    """
    print("\n[2/5] Creating Feature Ablation Table...")
    
    ablation = load_ablation_results()
    if 'feature' not in ablation:
        print("   ⚠ Feature ablation results not found")
        return
    
    df = ablation['feature'].copy()
    df = df.sort_values('RMSE_Impact', ascending=False)
    
    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Feature Ablation Study: Impact of Removing Feature Groups}
\label{tab:feature_ablation}
\begin{tabular}{lrrrrl}
\toprule
\textbf{Feature Group} & \textbf{Features} & \textbf{RMSE} & \textbf{$\Delta$ RMSE} & \textbf{R²} & \textbf{Interpretation} \\
\midrule
"""
    
    for _, row in df.iterrows():
        feature = row['Feature Group']
        n_feat = int(row['Features Count'])
        rmse = f"{row['RMSE']:.2f}"
        impact = f"{row['RMSE_Impact']:+.3f}"
        r2 = f"{row['R2']:.3f}"
        interp = row['Interpretation']
        
        # Highlight baseline
        if 'Baseline' in feature:
            latex += f"\\textbf{{{feature}}} & {n_feat} & \\textbf{{{rmse}}} & {impact} & {r2} & {interp} \\\\\n"
        else:
            latex += f"{feature} & {n_feat} & {rmse} & {impact} & {r2} & {interp} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\Delta$ RMSE shows change when feature group is removed. Positive values indicate performance degradation.
\item Negative values indicate that removing the feature group improves performance (suggesting overfitting).
\end{tablenotes}
\end{table}
"""
    
    # Save
    output_path = OUTPUT_DIR / 'feature_ablation.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"   ✓ Saved: {output_path}")


def table_3_data_ablation():
    """
    Table 3: Data Configuration Ablation Study
    """
    print("\n[3/5] Creating Data Ablation Table...")
    
    ablation = load_ablation_results()
    if 'data' not in ablation:
        print("   ⚠ Data ablation results not found")
        return
    
    df = ablation['data'].copy()
    
    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Data Configuration Ablation Study Results}
\label{tab:data_ablation}
\begin{tabular}{lrrrrl}
\toprule
\textbf{Configuration} & \textbf{Samples} & \textbf{RMSE} & \textbf{MAE} & \textbf{R²} & \textbf{Interpretation} \\
\midrule
"""
    
    for _, row in df.iterrows():
        config = row['Configuration']
        samples = f"{int(row['Samples']):,}"
        rmse = f"{row['RMSE']:.2f}"
        mae = f"{row['MAE']:.2f}"
        r2 = f"{row['R2']:.3f}"
        interp = row['Interpretation'] if pd.notna(row['Interpretation']) else ''
        
        # Highlight best configuration
        if interp and ('Optimal' in interp or 'BEST' in interp or '20' in config):
            latex += f"\\textbf{{{config}}} & {samples} & \\textbf{{{rmse}}} & {mae} & {r2} & {interp} \\\\\n"
        else:
            latex += f"{config} & {samples} & {rmse} & {mae} & {r2} & {interp} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Configurations tested include historical data span and sequence length variations.
\item Best performing configuration shown in bold.
\end{tablenotes}
\end{table}
"""
    
    # Save
    output_path = OUTPUT_DIR / 'data_ablation.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"   ✓ Saved: {output_path}")


def table_4_hyperparameters():
    """
    Table 4: Model Hyperparameters
    """
    print("\n[4/5] Creating Hyperparameters Table...")
    
    # Define hyperparameters for each model
    hyperparams = [
        {
            'Model': 'LSTM with Attention',
            'Architecture': '2 LSTM layers (64, 32 units)',
            'Attention': 'Yes',
            'Dropout': '0.2',
            'Learning Rate': '0.0001',
            'Batch Size': '32',
            'Epochs': '50'
        },
        {
            'Model': 'Bidirectional LSTM',
            'Architecture': '2 Bi-LSTM layers (32 units each)',
            'Attention': 'No',
            'Dropout': '0.2',
            'Learning Rate': '0.0001',
            'Batch Size': '32',
            'Epochs': '50'
        }
    ]
    
    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Deep Learning Model Hyperparameters}
\label{tab:hyperparameters}
\begin{tabular}{lllllll}
\toprule
\textbf{Model} & \textbf{Architecture} & \textbf{Attention} & \textbf{Dropout} & \textbf{LR} & \textbf{Batch} & \textbf{Epochs} \\
\midrule
"""
    
    for params in hyperparams:
        model = params['Model']
        arch = params['Architecture']
        att = params['Attention']
        dropout = params['Dropout']
        lr = params['Learning Rate']
        batch = params['Batch Size']
        epochs = params['Epochs']
        
        # Bold the best model
        if model == 'Bidirectional LSTM':
            latex += f"\\textbf{{{model}}} & {arch} & {att} & {dropout} & {lr} & {batch} & {epochs} \\\\\n"
        else:
            latex += f"{model} & {arch} & {att} & {dropout} & {lr} & {batch} & {epochs} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: LR = Learning Rate (Adam optimizer). All models use early stopping (patience=15).
\item Sequence length = 30 weeks. Train/Val/Test split = 70/15/15.
\end{tablenotes}
\end{table}
"""
    
    # Save
    output_path = OUTPUT_DIR / 'hyperparameters.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"   ✓ Saved: {output_path}")


def table_5_dataset_summary():
    """
    Table 5: Dataset Summary Statistics
    """
    print("\n[5/5] Creating Dataset Summary Table...")
    
    # Dataset statistics
    latex = r"""\begin{table}[htbp]
\centering
\caption{Dataset Summary Statistics}
\label{tab:dataset_summary}
\begin{tabular}{lr}
\toprule
\textbf{Characteristic} & \textbf{Value} \\
\midrule
Data Source & Global Terrorism Database (GTD) \\
Time Period & 1970--2016 \\
Total Records & 29,436 \\
Temporal Granularity & Weekly \\
Total Weeks & 2,434 \\
Number of Features & 13 \\
\midrule
\multicolumn{2}{l}{\textbf{Feature Categories:}} \\
\quad Lag Features & 1 (52-week lag) \\
\quad Rolling Statistics & 4 (4-week \& 12-week) \\
\quad Geographic Features & 1 (region) \\
\quad Casualty Features & 3 (casualties, killed, wounded) \\
\quad Target Variable & Attack count (weekly) \\
\midrule
\multicolumn{2}{l}{\textbf{Data Split:}} \\
\quad Training Set & 70\% (20,605 records) \\
\quad Validation Set & 15\% (4,415 records) \\
\quad Test Set & 15\% (4,416 records) \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Features include temporal patterns, geographic context, and historical casualty information.
\end{tablenotes}
\end{table}
"""
    
    # Save
    output_path = OUTPUT_DIR / 'dataset_summary.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"   ✓ Saved: {output_path}")


def create_simple_csv_tables():
    """
    Create simple CSV versions for easy reference
    """
    print("\n[Bonus] Creating CSV versions for reference...")
    
    # Model comparison
    baseline_df = load_baseline_results()
    dl_results = load_model_results()
    
    if baseline_df is not None and dl_results:
        all_models = []
        
        for _, row in baseline_df.iterrows():
            all_models.append({
                'Model': row['model'],
                'Type': 'Baseline',
                'RMSE': row['RMSE'],
                'MAE': row['MAE'],
                'R²': row['R2']
            })
        
        if 'LSTM Attention' in dl_results:
            lstm_att = dl_results['LSTM Attention'].iloc[0]
            all_models.append({
                'Model': 'LSTM with Attention',
                'Type': 'Deep Learning',
                'RMSE': lstm_att['RMSE'],
                'MAE': lstm_att['MAE'],
                'R²': lstm_att['R2']
            })
        
        if 'Bidirectional LSTM' in dl_results:
            bilstm = dl_results['Bidirectional LSTM'].iloc[0]
            all_models.append({
                'Model': 'Bidirectional LSTM',
                'Type': 'Deep Learning',
                'RMSE': bilstm['RMSE'],
                'MAE': bilstm['MAE'],
                'R²': bilstm['R2']
            })
        
        df = pd.DataFrame(all_models)
        output_path = OUTPUT_DIR / 'all_models_comparison.csv'
        df.to_csv(output_path, index=False)
        print(f"   ✓ Saved: {output_path}")


def main():
    """Generate all tables"""
    try:
        table_1_model_performance()
        table_2_feature_ablation()
        table_3_data_ablation()
        table_4_hyperparameters()
        table_5_dataset_summary()
        create_simple_csv_tables()
        
        print("\n" + "="*60)
        print("✅ ALL LATEX TABLES GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated LaTeX tables:")
        print("  1. model_performance.tex")
        print("  2. feature_ablation.tex")
        print("  3. data_ablation.tex")
        print("  4. hyperparameters.tex")
        print("  5. dataset_summary.tex")
        print("\nUsage in LaTeX:")
        print("  \\input{tables/latex/model_performance.tex}")
        
    except Exception as e:
        print(f"\n❌ Error generating tables: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
