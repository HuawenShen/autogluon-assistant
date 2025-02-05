# Condensed: AutoGluon Time Series - Forecasting Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# AutoGluon Time Series - Forecasting Quick Start

## Key Components
- `TimeSeriesDataFrame`: Stores multiple time series data
- `TimeSeriesPredictor`: Handles model fitting, tuning, and forecasting

## Data Preparation

```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data
df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")

# Convert to TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

**Important**: Data must be in long format with three essential columns:
- Unique ID for each time series
- Timestamp
- Target value

## Model Training

```python
predictor = TimeSeriesPredictor(
    prediction_length=48,  # Forecast horizon (48 hours)
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)
```

**Key Configuration Options**:
- `prediction_length`: Number of future timesteps to forecast
- `presets`: Model selection strategy
  - "medium_quality": Includes baselines, statistical models, tree-based models, and deep learning
  - Other options: "fast_training", "high_quality", "best_quality"
- `time_limit`: Maximum training duration in seconds

## Generating Forecasts

```python
# Generate probabilistic forecasts
predictions = predictor.predict(train_data)

# Visualize forecasts
test_data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv")
predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)

# Evaluate model performance
predictor.leaderboard(test_data)
```

**Forecast Features**:
- Produces probabilistic forecasts with quantiles
- Automatically selects best performing model
- Evaluates performance using specified metric (MASE)

## Models Included (medium_quality preset):
- Baselines: Naive, SeasonalNaive
- Statistical: ETS, Theta
- Tree-based: RecursiveTabular, DirectTabular
- Deep Learning: TemporalFusionTransformer
- Weighted ensemble of above models

**Note**: Higher quality presets typically produce better forecasts but require longer training times.

This tutorial demonstrates core functionality for time series forecasting with AutoGluon. For advanced features, refer to the in-depth forecasting tutorial.