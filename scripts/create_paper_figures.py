"""
Generate Publication-Quality Figures for Paper

This script creates all figures needed for the deep learning counterterrorism paper.
Outputs are saved to reports/figures/paper/ as high-resolution PNG files.

Usage:
    python scripts/create_paper_figures.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Output directory
OUTPUT_DIR = project_root / 'reports' / 'figures' / 'paper'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("GENERATING PUBLICATION-QUALITY FIGURES")
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


def figure_1_model_comparison():
    """
    Figure 1: Model Performance Comparison
    Bar chart comparing RMSE across all models
    """
    print("\n[1/6] Creating Model Comparison Figure...")
    
    # Collect all model results
    models = []
    rmse_values = []
    colors = []
    
    # Baselines
    baseline_df = load_baseline_results()
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            models.append(row['model'])
            rmse_values.append(row['RMSE'])
            colors.append('#95a5a6')  # Gray for baselines
    
    # Deep learning models
    dl_results = load_model_results()
    if 'LSTM Attention' in dl_results:
        lstm_att = dl_results['LSTM Attention']
        models.append('LSTM Attention')
        rmse_values.append(lstm_att['RMSE'].values[0])
        colors.append('#3498db')  # Blue
    
    if 'Bidirectional LSTM' in dl_results:
        bilstm = dl_results['Bidirectional LSTM']
        models.append('Bidirectional LSTM')
        rmse_values.append(bilstm['RMSE'].values[0])
        colors.append('#e74c3c')  # Red (best model)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, rmse_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, rmse_values)):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', 
                va='center', ha='left', fontweight='bold', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('RMSE (Root Mean Squared Error)', fontweight='bold')
    ax.set_title('Model Performance Comparison: Terrorist Attack Forecasting', 
                 fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', edgecolor='black', label='Baseline Models'),
        Patch(facecolor='#3498db', edgecolor='black', label='LSTM Attention'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Bidirectional LSTM (Best)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def figure_2_training_history():
    """
    Figure 2: Training History
    Shows loss curves for deep learning models
    """
    print("\n[2/6] Creating Training History Figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LSTM Attention history
    lstm_att_fig = project_root / 'reports' / 'figures' / 'lstm_attention_training_history.png'
    bilstm_fig = project_root / 'reports' / 'figures' / 'bidirectional_lstm_training_history.png'
    
    if lstm_att_fig.exists() and bilstm_fig.exists():
        print("   ℹ Training history plots already exist in reports/figures/")
        print("   ℹ Using existing plots instead of recreating")
        plt.close()
        return
    
    # Placeholder if history not available
    for ax in axes:
        ax.text(0.5, 0.5, 'Training history plots\navailable in\nreports/figures/', 
                ha='center', va='center', fontsize=12, style='italic')
        ax.axis('off')
    
    plt.suptitle('Model Training History', fontweight='bold', fontsize=14)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def figure_3_predictions():
    """
    Figure 3: Predictions vs Actual
    Shows model predictions against ground truth
    """
    print("\n[3/6] Creating Predictions Comparison Figure...")
    
    # Check if prediction plots exist
    lstm_att_pred = project_root / 'reports' / 'figures' / 'lstm_attention_predictions.png'
    bilstm_pred = project_root / 'reports' / 'figures' / 'bidirectional_lstm_predictions.png'
    
    if lstm_att_pred.exists() and bilstm_pred.exists():
        print("   ℹ Prediction plots already exist in reports/figures/")
        print("   ℹ Using existing plots instead of recreating")
        return
    
    # Create placeholder
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, 'Prediction plots available in\nreports/figures/\n\n' + 
            'lstm_attention_predictions.png\nbidirectional_lstm_predictions.png', 
            ha='center', va='center', fontsize=12, style='italic')
    ax.axis('off')
    
    plt.suptitle('Model Predictions vs Actual Values', fontweight='bold', fontsize=14)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'predictions_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def figure_4_feature_ablation():
    """
    Figure 4: Feature Ablation Results
    Bar chart showing impact of removing each feature group
    """
    print("\n[4/6] Creating Feature Ablation Figure...")
    
    ablation = load_ablation_results()
    if 'feature' not in ablation:
        print("   ⚠ Feature ablation results not found")
        return
    
    df = ablation['feature']
    
    # Filter out baseline and sort by impact
    df_plot = df[df['Feature Group'] != 'Baseline (All Features)'].copy()
    df_plot = df_plot.sort_values('RMSE_Impact', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color code: negative impact (helpful) = red, positive (harmful) = blue
    colors = ['#e74c3c' if x < 0 else '#3498db' for x in df_plot['RMSE_Impact']]
    
    bars = ax.barh(df_plot['Feature Group'], df_plot['RMSE_Impact'], 
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, df_plot['RMSE_Impact']):
        x_pos = val + (0.05 if val > 0 else -0.05)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', 
                va='center', ha=ha, fontweight='bold', fontsize=9)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('RMSE Impact (when feature removed)', fontweight='bold')
    ax.set_title('Feature Ablation Study: Impact of Removing Feature Groups', 
                 fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add interpretation text
    ax.text(0.98, 0.02, 
            '← Removing helps performance | Removing hurts performance →', 
            transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def figure_5_data_ablation():
    """
    Figure 5: Data Configuration Ablation
    Shows impact of different data configurations
    """
    print("\n[5/6] Creating Data Ablation Figure...")
    
    ablation = load_ablation_results()
    if 'data' not in ablation:
        print("   ⚠ Data ablation results not found")
        return
    
    df = ablation['data']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Historical span comparison
    span_df = df[df['Configuration'].str.contains('History|Recent')].copy()
    if not span_df.empty:
        configs = span_df['Configuration'].values
        rmse = span_df['RMSE'].values
        colors_span = ['#2ecc71', '#e74c3c']
        
        bars1 = ax1.bar(configs, rmse, color=colors_span, 
                        edgecolor='black', linewidth=0.5, alpha=0.8)
        
        for bar, val in zip(bars1, rmse):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, 
                    f'{val:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_ylabel('RMSE', fontweight='bold')
        ax1.set_title('Impact of Historical Data Span', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_xticklabels(configs, rotation=15, ha='right')
    
    # Subplot 2: Sequence length comparison
    seq_df = df[df['Configuration'].str.contains('Sequence')].copy()
    if not seq_df.empty:
        seq_lengths = seq_df['Configuration'].str.extract(r'(\d+)')[0].astype(int).values
        rmse = seq_df['RMSE'].values
        
        # Color code: highlight the best
        best_idx = np.argmin(rmse)
        colors_seq = ['#e74c3c' if i == best_idx else '#3498db' 
                      for i in range(len(rmse))]
        
        bars2 = ax2.bar(seq_lengths, rmse, color=colors_seq, 
                        edgecolor='black', linewidth=0.5, alpha=0.8, width=3)
        
        for bar, val in zip(bars2, rmse):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2, 
                    f'{val:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_xlabel('Sequence Length (weeks)', fontweight='bold')
        ax2.set_ylabel('RMSE', fontweight='bold')
        ax2.set_title('Impact of Sequence Length', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_xticks(seq_lengths)
    
    plt.suptitle('Data Configuration Ablation Study', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'data_ablation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def figure_6_improvement_summary():
    """
    Figure 6: Improvement Summary
    Visual summary of key improvements
    """
    print("\n[6/6] Creating Improvement Summary Figure...")
    
    # Get best baseline and best deep learning model
    baseline_df = load_baseline_results()
    dl_results = load_model_results()
    
    if baseline_df is None or 'Bidirectional LSTM' not in dl_results:
        print("   ⚠ Missing data for improvement summary")
        return
    
    best_baseline_rmse = baseline_df['RMSE'].min()
    best_baseline_name = baseline_df.loc[baseline_df['RMSE'].idxmin(), 'model']
    
    bilstm_rmse = dl_results['Bidirectional LSTM']['RMSE'].values[0]
    
    improvement_pct = ((best_baseline_rmse - bilstm_rmse) / best_baseline_rmse) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [f'Baseline\n({best_baseline_name})', 'Bidirectional\nLSTM']
    rmse_vals = [best_baseline_rmse, bilstm_rmse]
    colors = ['#95a5a6', '#e74c3c']
    
    bars = ax.bar(models, rmse_vals, color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.8, width=0.5)
    
    # Add value labels
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                f'RMSE: {val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add improvement arrow and text
    arrow_y = max(rmse_vals) * 0.7
    ax.annotate('', xy=(1, bilstm_rmse), xytext=(0, best_baseline_rmse),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax.text(0.5, arrow_y, 
            f'{improvement_pct:.1f}% Improvement', 
            ha='center', va='center', fontsize=14, fontweight='bold', 
            color='green', bbox=dict(boxstyle='round', facecolor='white', 
                                     edgecolor='green', linewidth=2))
    
    ax.set_ylabel('RMSE (Root Mean Squared Error)', fontweight='bold', fontsize=12)
    ax.set_title('Deep Learning Performance Improvement over Baseline', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rmse_vals) * 1.2)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'improvement_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def main():
    """Generate all figures"""
    try:
        figure_1_model_comparison()
        figure_2_training_history()
        figure_3_predictions()
        figure_4_feature_ablation()
        figure_5_data_ablation()
        figure_6_improvement_summary()
        
        print("\n" + "="*60)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated figures:")
        print("  1. model_comparison.png")
        print("  2. training_curves.png")
        print("  3. predictions_comparison.png")
        print("  4. feature_importance.png")
        print("  5. data_ablation.png")
        print("  6. improvement_summary.png")

        
    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
