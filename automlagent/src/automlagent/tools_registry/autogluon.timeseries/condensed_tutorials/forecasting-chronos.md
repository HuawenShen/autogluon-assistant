# Condensed: Forecasting with Chronos

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Forecasting with Chronos

## Key Concepts
- Chronos models are pretrained on large collections of time series data for zero-shot forecasting
- Two model families available:
  - Chronos-Bolt⚡️ (new, faster, CPU-compatible): `bolt_tiny`, `bolt_mini`, `bolt_small`, `bolt_base`
  - Original Chronos (GPU required for larger models): `chronos_tiny` through `chronos_large`

## Basic Usage

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data
data = TimeSeriesDataFrame.from_path("path_to_data.csv")

# Split data
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Zero-shot forecasting
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, 
    presets="bolt_small"
)

# Generate predictions
predictions = predictor.predict(train_data)
```

## Fine-tuning

```python
# Compare zero-shot vs fine-tuned models
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,
    enable_ensemble=False,
)

# Evaluate models
predictor.leaderboard(test_data)
```

Custom fine-tuning parameters:
```python
predictor.fit(
    hyperparameters={
        "Chronos": {
            "fine_tune": True,
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 2000
        }
    }
)
```

## Handling Covariates

For datasets with known covariates:

```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="target_column",
    known_covariates_names=["covariate1", "covariate2"]
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot without covariates
            {
                "model_path": "bolt_small",
                "ag_args": {"name_suffix": "ZeroShot"},
            },
            # With covariate regressor
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ]
    }
)
```

## Important Notes

Hardware Requirements:
- Original Chronos (small and larger): GPU required
- Chronos-Bolt: Works on both CPU and GPU
- Recommended: g5.2xlarge/p3.2xlarge with 16GB+ GPU memory, 32GB+ RAM

Best Practices:
- Use Chronos-Bolt for better performance and CPU compatibility
- Apply target scaling when using covariate regressors
- Compare multiple model configurations using leaderboard evaluation
- Consider fine-tuning for improved accuracy on specific datasets

The tutorial preserves all essential implementation details while removing redundant explanations and maintaining clear structure for practical usage.