# Condensed: AutoGluon Time Series - Forecasting Quick Start

Summary: This tutorial demonstrates AutoGluon's time series forecasting implementation, focusing on the TimeSeriesDataFrame and TimeSeriesPredictor components. It covers essential tasks including loading time series data in long format, training models with different quality presets (fast to best), and generating probabilistic forecasts with uncertainty estimates. Key functionalities include handling multiple time series, automatic model selection from a diverse pool (baseline, statistical, tree-based, and deep learning models), customizable prediction horizons, and model evaluation through leaderboards. The tutorial is particularly useful for implementing automated time series forecasting pipelines with minimal code while maintaining flexibility in model selection and training parameters.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoGluon Time Series - Quick Start Guide

## Key Components
- `TimeSeriesDataFrame`: Stores multiple time series datasets
- `TimeSeriesPredictor`: Handles model fitting, tuning, and forecasting

## Implementation Details

### 1. Setup and Data Loading
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data and convert to TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

**Required Data Format:**
- Long format with columns for:
  - Unique ID (`item_id`)
  - Timestamp
  - Target value

### 2. Model Training
```python
predictor = TimeSeriesPredictor(
    prediction_length=48,  # Forecast horizon
    path="autogluon-m4-hourly",  # Save path
    target="target",
    eval_metric="MASE"
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600
)
```

**Preset Options:**
- `fast_training`
- `medium_quality`
- `high_quality`
- `best_quality`

**Models Included in Medium Quality:**
- Baselines: Naive, SeasonalNaive
- Statistical: ETS, Theta
- Tree-based: RecursiveTabular, DirectTabular
- Deep Learning: TemporalFusionTransformer
- Weighted ensemble

### 3. Generating Forecasts
```python
predictions = predictor.predict(train_data)
```

**Output Features:**
- Probabilistic forecasts
- Mean predictions
- Quantile forecasts (uncertainty estimates)

### 4. Model Evaluation
```python
predictor.leaderboard(test_data)
```

## Best Practices
1. Choose prediction length based on data frequency
2. Use higher quality presets for better accuracy (requires more time)
3. Provide sufficient training time limit based on dataset size
4. Consider validation scores when selecting models

## Important Notes
- AutoGluon generates individual forecasts for each time series without modeling inter-series interactions
- Higher scores in leaderboard indicate better performance (MASE scores are multiplied by -1)
- TimeSeriesDataFrame inherits from pandas.DataFrame, supporting all its methods