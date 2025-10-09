# Baseline Models for Terrorism Incident Forecasting

This module implements traditional time series forecasting methods to establish performance benchmarks for deep learning models. These baselines provide critical context for evaluating whether LSTM/GRU architectures offer meaningful improvements over classical statistical approaches.

## Overview

Four baseline models are implemented to represent different forecasting paradigms:
- **Seasonal methods**: Capture periodic patterns in terrorism incidents
- **Smoothing methods**: Filter noise from volatile time series
- **Statistical models**: Industry-standard ARIMA/SARIMA approaches
- **Linear models**: Simple supervised learning with lagged features

## Implemented Models

### 1. Seasonal Naive Forecasting (`01_seasonal_naive.py`)
**Methodology**: Uses the value from the corresponding period in the previous year (52-week lag) as the forecast.

- **Rationale**: Terrorism incidents may exhibit annual seasonality due to weather patterns, religious holidays, or political cycles
- **Advantages**: Simple, computationally efficient, effective for strongly seasonal data
- **Limitations**: Cannot adapt to trends or regime changes; assumes year-over-year similarity
- **Complexity**: O(1) prediction time

### 2. Moving Average (`02_moving_average.py`)
**Methodology**: Forecasts using the arithmetic mean of the N most recent observations (window=4 weeks).

- **Rationale**: Smooths short-term volatility in attack frequency while preserving recent trends
- **Advantages**: Robust to outliers, simple implementation, minimal assumptions
- **Limitations**: Introduces lag in trend detection; no seasonality modeling; fixed window may not adapt to regime changes
- **Complexity**: O(N) where N is window size

### 3. ARIMA/SARIMA (`03_arima_sarima.py`)
**Methodology**: Seasonal Autoregressive Integrated Moving Average with parameters ARIMA(1,1,1) × (1,1,1)₁₂.

- **Rationale**: Industry-standard statistical approach combining autoregression, differencing, and moving average components with seasonal terms
- **Advantages**: Captures both trend and seasonality; well-established theoretical foundation; interpretable coefficients
- **Limitations**: Assumes linear relationships; univariate (ignores external factors); computationally expensive; may be unstable for volatile series
- **Complexity**: O(n³) for fitting, O(1) for prediction

### 4. Linear Regression with Lagged Features (`04_linear_regression.py`)
**Methodology**: Ordinary least squares regression using 4 lagged attack counts as predictors.

- **Rationale**: Exploits autocorrelation in attack frequency while maintaining model interpretability
- **Advantages**: Transparent feature importance; fast training and inference; can incorporate external regressors
- **Limitations**: Linear assumption may miss nonlinear temporal dynamics; fixed lag structure; no explicit seasonality
- **Complexity**: O(n·p²) for fitting where p is number of lags

**Outputs**:
1. Console summary with performance metrics for all models
2. `reports/tables/baseline_comparison.csv` - Machine-readable results
3. `reports/tables/baseline_comparison.tex` - Publication-ready LaTeX table


## Empirical Results

### Dataset Characteristics
- **Total samples**: 29,436 weekly aggregated records (1969-2016)
- **Mean weekly attacks**: 5.79 ± 14.82 (high variance indicates forecasting difficulty)
- **Split**: 70% train (20,605), 15% validation (4,415), 15% test (4,416)
- **Temporal ordering preserved**: Chronological split prevents data leakage

### Baseline Performance on Test Set

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Interpretation |
|-------|---------|--------|-------|----------------|
| **Linear Regression** | **9.89** | **5.61** | **-0.10** | Best baseline; autocorrelation features effective |
| Seasonal Naive | 9.96 | 5.62 | -0.12 | Strong annual patterns detected |
| Moving Average | 10.09 | 7.92 | -0.15 | Smoothing insufficient for volatility |
| SARIMA | 11.52 | 9.96 | -0.50 | Model instability on irregular series |

**Key Findings**:
- Negative R² values indicate high intrinsic unpredictability (terrorist attacks are rare, volatile events)
- Linear Regression emerges as the benchmark to beat (**RMSE: 9.89**)
- SARIMA underperforms despite complexity, suggesting non-stationary dynamics
- Best baseline explains only ~10% better than naive mean predictor

### Target for Deep Learning Models

**Success Criterion**: LSTM/GRU models should achieve:
- **Minimum**: RMSE < 9.89 (beat Linear Regression)
- **Publication-worthy**: RMSE < 8.40 (15% improvement)
- **Strong result**: RMSE < 7.90 (20% improvement)

**Justification**: Deep learning is warranted if:
1. Improvement ≥ 15% over best baseline
2. Captures nonlinear temporal dependencies missed by linear models
3. Demonstrates statistical significance (confidence intervals do not overlap)

## Configuration Parameters

All baseline models respect hyperparameters defined in `configs/config.yaml`:

```yaml
model:
  sequence_length: 30           # LSTM lookback (not used by baselines)
  seasonal_lag_weeks: 52        # Seasonal Naive period
  batch_size: 32               # LSTM batch size (not used by baselines)
  epochs: 100                  # LSTM epochs (not used by baselines)

split:
  train_ratio: 0.70            # Used by all models
  val_ratio: 0.15              # Used by all models  
  test_ratio: 0.15             # Used by all models

seed: 42                       # Random seed for reproducibility
```

**Model-Specific Parameters**:
- Seasonal Naive: `seasonal_lag_weeks = 52` (annual cycle)
- Moving Average: `window = 4` (4-week smoothing)
- SARIMA: `order = (1,1,1)`, `seasonal_order = (1,1,1,12)` (quarterly seasonality)
- Linear Regression: `n_lags = 4` (autoregressive features)

## Implementation Notes

### Data Requirements
- **Input format**: 1D numpy array of weekly attack counts
- **Minimum length**: 104 observations (2 years) for SARIMA seasonality fitting
- **Split strategy**: Chronological (respects temporal ordering)
- **Preprocessing**: Data must be aggregated to weekly grain before baseline evaluation

### Computational Considerations
- **Seasonal Naive**: Instant (<0.1s for 29K samples)
- **Moving Average**: Near-instant (<0.5s for 29K samples)
- **Linear Regression**: Fast (~1-2s for 29K samples with sklearn)
- **SARIMA**: Slow (30-60s for 29K samples; may require grid search)

### Known Limitations
1. **SARIMA stability**: May fail to converge on highly irregular time series or with insufficient data
2. **Univariate assumption**: All models ignore potentially useful covariates (region, attack type, etc.)
3. **Linear dynamics**: Except SARIMA, models assume linear relationships
4. **Point forecasts only**: No uncertainty quantification (confidence intervals)

## Reproducibility

All baselines use `random_state=42` where applicable and follow deterministic procedures. Results should be exactly reproducible across runs given identical:
- Input data (`03_weekly_aggregated.csv`)
- Configuration (`configs/config.yaml`)
- Python environment (see `requirements.txt`)




