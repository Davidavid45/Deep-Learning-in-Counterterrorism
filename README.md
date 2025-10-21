# Deep Learning for Terrorism Forecasting# Deep Learning in Counterterrorism
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17409540.svg)](https://doi.org/10.5281/zenodo.17409540)
Time series forecasting of global terrorist attacks using deep learning architectures on the Global Terrorism Database (1970-2016).

## Project Overview

This project leverages advanced deep learning techniques, including LSTM (Long Short-Term Memory) and GCN (Graph Convolutional Networks), to analyze and predict terrorist activity based on historical data. By capturing both temporal and relational patterns, the models provide actionable insights for improving security planning and resource allocation.

## Overview

## Key Features

This project applies LSTM-based deep learning models to predict weekly terrorist attack counts. Using 46 years of historical data, we achieve a 35.4% improvement over traditional time series baselines through bidirectional sequence processing and comprehensive feature engineering.- Forecasting weekly terrorist activity trends with high accuracy (MAE: 67.6).

- Classifying terrorist attack types with an accuracy of 85%.

**Best Model Performance:**- Visualizing regional hotspots and patterns in global terrorism data.

- RMSE: 6.38 attacks/week- Data preprocessing for feature engineering and dimensionality reduction.

- R²: 0.540 (explains 54% of variance)

- 35.4% improvement over Linear Regression baseline## Dataset

- **Source**: [Global Terrorism Database (GTD)](https://www.start.umd.edu/gtd)

## Features- The dataset includes information on global terrorist incidents from 1970 to 2016.


- **Temporal forecasting** with LSTM and Bidirectional LSTM architectures## Technologies Used

- **Feature engineering** including lag features, rolling statistics, and casualty metrics- **Python**: Data analysis, preprocessing, and model development.

- **Ablation studies** to validate model design choices- **TensorFlow/Keras**: Building and training LSTM models.

- **Baseline comparisons** with traditional methods (Linear Regression, SARIMA, Moving Average)- **NetworkX**: Graph-based data representation for GCN.

- **Publication-ready visualizations** and LaTeX tables- **Matplotlib/Seaborn**: Data visualization.

- **Pandas/Numpy**: Data manipulation.

## Dataset

## LICENSE

**Source:** [Global Terrorism Database (GTD)](https://www.start.umd.edu/gtd)  This project uses the **Global Terrorism Database (GTD)** by the National Consortium for the Study of Terrorism and Responses to Terrorism (START), University of Maryland. Use of the GTD is governed by START’s **Terms of Use**: https://www.start.umd.edu/gtd-terms

**Time Period:** 1970-2016 (46 years)  

**Records:** 29,436 weekly aggregated observations  **Required citation**  

**Features:** 13 engineered features including temporal, geographic, and casualty information> Global Terrorism Database (GTD) [2023]. National Consortium for the Study of Terrorism and Responses to Terrorism (START), University of Maryland. Available at https://www.start.umd.edu/gtd. (Accessed: 2023-10-08)

> Source: https://www.start.umd.edu/gtd  

> **Note:** Download the GTD dataset from [START website](https://www.start.umd.edu/gtd) and place it in `data/raw/` before running preprocessing.

## Project Structure

```
├── configs/              # Configuration files
├── data/                 # Dataset location (not included in repo)
├── reports/
│   ├── figures/          # Training plots and predictions
│   └── tables/           # Results tables (CSV and LaTeX)
├── scripts/              # Paper figure and table generation
├── src/
│   ├── baselines/        # Traditional forecasting models
│   ├── models/           # Deep learning models
│   ├── preprocess/       # Data preprocessing pipeline
│   ├── ablations/        # Ablation study scripts
│   └── utils/            # Helper functions
├── Makefile              # Automation commands
└── requirements.txt      # Python dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/Davidavid45/Deep-Learning-in-Counterterrorism.git
cd Deep-Learning-in-Counterterrorism

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

## Usage

### Quick Start

```bash
# 1. Run preprocessing (requires GTD data in data/raw/)
make preprocess

# 2. Train best model
make train

# 3. Generate paper assets
make paper-assets
```

### Step-by-Step

```bash
# Preprocess data
cd src/preprocess
python run_preprocess.py

# Run baseline models
cd src/baselines
python run_all_baselines.py

# Train deep learning models
python src/models/bidirectional_lstm_weekly.py
python src/models/lstm_attention_weekly.py

# Run ablation studies
python src/ablations/run_ablations.py

# Generate figures and tables
python scripts/create_paper_figures.py
python scripts/create_paper_tables.py
```

## Models

### Baseline Methods
- Linear Regression: RMSE 9.89
- Seasonal Naive: RMSE 9.96
- Moving Average: RMSE 10.09
- SARIMA: RMSE 11.52

### Deep Learning Models
- **Bidirectional LSTM** (Best): RMSE 6.38, R² 0.540
- LSTM with Attention: RMSE 9.19, R² 0.046

## Key Findings

1. **Long historical data is critical:** Using only 5 years degrades performance by 126%
2. **Optimal sequence length:** 20 weeks provides best balance
3. **Bidirectional processing essential:** Captures both past context and future trends
4. **Feature importance:** Lag features most critical, followed by casualty and geographic data

## Results

All results are saved in `reports/`:
- **Figures:** `reports/figures/paper/` - Publication-quality PNG files
- **Tables:** `reports/tables/latex/` - Ready-to-use LaTeX tables
- **Metrics:** `reports/tables/` - CSV files with detailed metrics


## Citation

If you use this work, please cite:

```bibtex
@misc{deep-learning-counterterrorism,
  author = Oluwasegun Adegoke,
  title = {Deep Learning for Terrorism Forecasting},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/Deep-Learning-in-Counterterrorism}
}
```

```bibtex
@misc{adegoke2025gtdforecast,
  title   = {Predicting the Unpredictable: Reproducible BiLSTM Forecasting of Incident Counts in the Global Terrorism Database (GTD)},
  author  = {Oluwasegun Adegoke},
  year    = {2025},
  eprint  = {ARXIV_ID_HERE},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
}
```

## Data Citation

This project uses the Global Terrorism Database:

> National Consortium for the Study of Terrorism and Responses to Terrorism (START). (2023). Global Terrorism Database [Data file]. Retrieved from https://www.start.umd.edu/gtd

**Terms of Use:** https://www.start.umd.edu/gtd-terms

The Global Terrorism Database is subject to START's Terms of Use.
