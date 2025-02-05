# Condensed: Forecasting Time Series - In Depth

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details while maintaining key concepts:

# Forecasting Time Series - In Depth

## Key Concepts

### Probabilistic Time Series Forecasting
- Time series: Sequence of measurements at regular intervals
- Main goal: Predict future values given past observations
- Two types of forecasts:
  - Mean forecast: Expected future values
  - Quantile forecast: Distribution quantiles (e.g., P10, P90)

```python
# Configure custom quantiles
predictor = TimeSeriesPredictor(quantile_levels=[0.05, 0.5, 0.95])
```

### Additional Information Types

1. Static Features
```python
# Create TimeSeriesDataFrame with static features
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,
)

# Alternative: Attach static features to existing DataFrame
train_data.static_features = static_features_df
```

2. Time-varying Covariates
- Known covariates: Known for entire forecast horizon (e.g., holidays)
- Past covariates: Known only up to forecast start (e.g., related metrics)

```python
# Specify covariates when creating predictor
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"],
).fit(train_data)

# Generate predictions with known covariates
predictions = predictor.predict(train_data, known_covariates=known_covariates)
```

### Holiday Features Implementation
```python
import holidays

def add_holiday_features(ts_df, country_holidays, 
                        include_individual_holidays=True,
                        include_holiday_indicator=True):
    ts_df = ts_df.copy()
    timestamps = ts_df.index.get_level_values("timestamp")
    country_holidays_df = pd.get_dummies(pd.Series(country_holidays)).astype(float)
    holidays_df = country_holidays_df.reindex(timestamps.date).fillna(0)
    if include_individual_holidays:
        ts_df[holidays_df.columns] = holidays_df.values
    if include_holiday_indicator:
        ts_df["Holiday"] = holidays_df.max(axis=1).values
    return ts_df
```

## Data Format Requirements

### Minimum Length Requirements
- Default settings: `>= max(prediction_length + 1, 5) + prediction_length`
- With custom validation: `>= max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size`

### Handling Irregular Data
```python
# Specify frequency when creating predictor
predictor = TimeSeriesPredictor(freq="D").fit(df_irregular)

# Or manually convert frequency
df_regular = df_irregular.convert_frequency(freq="D")
```

## Evaluation

### Train-Test Split
```python
train_data, test_data = data.train_test_split(prediction_length)
predictor = TimeSeriesPredictor(prediction_length=prediction_length, 
                               eval_metric="MASE").fit(train_data)
score = predictor.evaluate(test_data)
```

### Multi-window Backtesting
```python
from autogluon.timeseries.splitter import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(prediction_length=prediction_length, 
                                  num_val_windows=3)
for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):
    score = predictor.evaluate(val_split)
```

## Model Configuration

### Basic Configuration
```python
# Using presets
predictor.fit(train_data, presets="medium_quality")

# With time limit
predictor.fit(train_data, time_limit=3600)  # 1 hour
```

### Manual Model Selection
```python
predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {},
        "Theta": [
            {"decomposition_type": "additive"},
            {"seasonal_period": 1},
        ],
    }
)
```

### Hyperparameter Tuning
```python
from autogluon.common import space

predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {
            "hidden_size": space.Int(20, 100),
            "dropout_rate": space.Categorical(0.1, 0.3),
        },
    },
    hyperparameter_tune_kwargs="auto",
    enable_ensemble=False,
)
```

This condensed version maintains all critical implementation details while removing redundant explanations and supplementary examples.