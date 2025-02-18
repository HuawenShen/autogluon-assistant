# Condensed: Forecasting Time Series - In Depth

Summary: This tutorial covers comprehensive time series forecasting implementation using AutoGluon, focusing on three main areas: (1) core probabilistic forecasting with static features and quantile predictions, (2) handling covariates (both known and past) and holiday features with specific data formatting requirements, and (3) model configuration and selection across local, global, and ensemble approaches. It provides code examples for data preprocessing, handling irregular/missing data, evaluation methods including backtesting, and model configuration with different quality presets. Key functionalities include TimeSeriesPredictor setup, static feature integration, covariate handling, holiday feature implementation, and hyperparameter tuning capabilities, making it valuable for tasks ranging from basic time series forecasting to complex multi-series predictions with additional features.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and concepts:

# Forecasting Time Series - Core Concepts and Implementation

## Probabilistic Time Series Forecasting

### Key Components
- **Time Series**: Sequential measurements at regular intervals
- **Forecast Horizon**: Future period to predict, set via `prediction_length`
- **Output Types**:
  1. Mean forecast (expected values)
  2. Quantile forecasts (distribution percentiles)

```python
# Custom quantile configuration
predictor = TimeSeriesPredictor(quantile_levels=[0.05, 0.5, 0.95])
```

## Working with Additional Information

### Static Features
Static features are time-independent attributes for each time series.

#### Implementation Example:
```python
# Load data and static features
df = pd.read_csv("path_to_timeseries_data.csv")
static_features_df = pd.read_csv("path_to_static_features.csv")

# Create TimeSeriesDataFrame with static features
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,
)

# Alternative: Attach static features to existing TimeSeriesDataFrame
train_data.static_features = static_features_df
```

### Static Feature Types
- **Categorical**: 
  - Datatypes: `object`, `string`, `category`
  - Example: domain, store_id, product_category
- **Continuous**: 
  - Datatypes: `int`, `float`
  - Example: numerical attributes

#### Converting Feature Types:
```python
# Convert numeric to categorical
train_data.static_features["store_id"] = train_data.static_features["store_id"].astype("category")
```

### Important Notes
1. Static features must include all `item_id`s present in training data
2. Prediction data must include same static features with matching column names/types
3. Non-supported datatypes are ignored

### Basic Training
```python
predictor = TimeSeriesPredictor(prediction_length=14).fit(train_data)
```

This condensed version maintains all critical implementation details while removing redundant explanations and focusing on practical code examples and configurations.

Here's the condensed tutorial focusing on key implementation details and concepts:

# Time Series Covariates and Data Format in AutoGluon

## Types of Covariates

AutoGluon supports two types of covariates:
1. **Known covariates**: Available for entire forecast horizon (holidays, day of week, promotions)
2. **Past covariates**: Only known up to forecast start (sales, temperature, transformed target series)

## Implementation Example

```python
# Adding covariates to TimeSeriesDataFrame
# Past covariate example
train_data["log_target"] = np.log(train_data["target"])

# Known covariate example (weekend indicator)
WEEKEND_INDICES = [5, 6]
timestamps = train_data.index.get_level_values("timestamp")
train_data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)

# Initialize predictor with covariates
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"],
).fit(train_data)
```

### Making Predictions with Known Covariates

```python
# Generate future known covariates
future_index = get_forecast_horizon_index_ts_dataframe(train_data, prediction_length=14)
future_timestamps = future_index.get_level_values("timestamp")
known_covariates = pd.DataFrame(index=future_index)
known_covariates["weekend"] = future_timestamps.weekday.isin(WEEKEND_INDICES).astype(float)

# Predict
predictions = predictor.predict(train_data, known_covariates=known_covariates)
```

## Holiday Features Implementation

```python
def add_holiday_features(
    ts_df: TimeSeriesDataFrame,
    country_holidays: dict,
    include_individual_holidays: bool = True,
    include_holiday_indicator: bool = True
) -> TimeSeriesDataFrame:
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

### Using Holiday Features

```python
# Add holidays
train_data_with_holidays = add_holiday_features(train_data, country_holidays)

# Initialize predictor with holiday features
holiday_columns = train_data_with_holidays.columns.difference(train_data.columns)
predictor = TimeSeriesPredictor(..., known_covariates_names=holiday_columns)
```

## Important Requirements

1. **Known Covariates Requirements**:
   - Must include all columns listed in `predictor.known_covariates_names`
   - `item_id` index must include all training data item ids
   - `timestamp` index must cover prediction_length steps into future

2. **Time Series Length Requirements**:
   - Default: Length ≥ max(prediction_length + 1, 5) + prediction_length
   - With custom validation: Length ≥ max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size

Note: Time series in the dataset can have different lengths.

Here's the condensed tutorial focusing on key implementation details and practices:

# Handling Irregular Data and Missing Values

## Irregular Time Series Data
```python
# Example of irregular data handling
predictor = TimeSeriesPredictor(..., freq="D").fit(df_irregular)
```
- Use `freq` parameter to specify desired frequency (e.g., "D" for daily)
- AutoGluon automatically handles frequency conversion and missing values

## Manual Data Processing
```python
# Convert frequency manually
df_regular = df_irregular.convert_frequency(freq="D")

# Fill missing values
df_filled = df_regular.fill_missing_values()  # Default: forward + backward filling
df_filled = df_regular.fill_missing_values(method="constant", value=0.0)  # For zero-demand cases
```

# Evaluation Methods

## Basic Train-Test Split
```python
# Split data for evaluation
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Train and evaluate
predictor = TimeSeriesPredictor(prediction_length=prediction_length, 
                               eval_metric="MASE").fit(train_data)
predictor.evaluate(test_data)
```

## Backtesting with Multiple Windows
```python
from autogluon.timeseries.splitter import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(prediction_length=prediction_length, 
                                  num_val_windows=3)
for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):
    score = predictor.evaluate(val_split)
```

## Validation Process
```python
# Multiple validation windows
predictor = TimeSeriesPredictor(...)
predictor.fit(train_data, num_val_windows=3)

# Custom validation set
predictor.fit(train_data=train_data, tuning_data=my_validation_dataset)
```

### Key Points:
1. Evaluation always uses the last `prediction_length` timesteps
2. Multi-window backtesting provides more accurate performance estimates but requires longer time series
3. `num_val_windows` increases validation robustness but also increases training time
4. Time series must be at least `(num_val_windows + 1) * prediction_length` long for multiple validation windows

### Best Practices:
- Use single-window backtesting for short time series
- Consider zero-filling for demand forecasting applications
- Use multiple validation windows when data length permits
- Specify appropriate frequency when handling irregular data

Here's the condensed version focusing on key implementation details and concepts:

# AutoGluon Forecasting Models and Configuration

## Available Model Types

### 1. Local Models
- Fit separately to each time series
- Include: `ETS`, `AutoARIMA`, `Theta`, `SeasonalNaive`
- Best for individual time series forecasting

### 2. Global Models
- Learn from entire training set with multiple time series
- Neural network models from GluonTS:
  ```python
  - DeepAR
  - PatchTST
  - DLinear
  - TemporalFusionTransformer
  ```
- Includes pre-trained models (Chronos)
- Tabular models: `RecursiveTabular`, `DirectTabular`

### 3. Ensemble Models
- `WeightedEnsemble` combines predictions from other models
- Can be disabled with `enable_ensemble=False`

## TimeSeriesPredictor Configuration

### Basic Configuration
```python
predictor = TimeSeriesPredictor()
predictor.fit(train_data, 
              presets="medium_quality",
              time_limit=3600)  # in seconds
```

### Preset Quality Levels
| Preset | Description | Fit Time |
|--------|-------------|-----------|
| `fast_training` | Simple statistical + baseline models | 0.5x |
| `medium_quality` | Fast models + TFT + Chronos-Bolt | 1x |
| `high_quality` | Advanced DL + ML models | 3x |
| `best_quality` | Extended cross-validation | 6x |

### Manual Model Configuration
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

# Exclude specific models
predictor.fit(
    train_data,
    presets="high_quality",
    excluded_model_types=["AutoETS", "AutoARIMA"]
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
    enable_ensemble=False
)
```

Important HPO Configuration:
- Uses Ray Tune for deep learning models
- Random search for other models
- Required keys in `hyperparameter_tune_kwargs`:
  - `num_trials`: number of configurations
  - `searcher`: use "random"
  - `scheduler`: use "local"

⚠️ **Warning**: HPO significantly increases training time with often modest performance gains.