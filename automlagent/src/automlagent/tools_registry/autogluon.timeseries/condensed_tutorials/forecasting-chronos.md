# Condensed: Forecasting with Chronos

Summary: This tutorial covers implementing time series forecasting using AutoGluon's Chronos models, focusing on both zero-shot and fine-tuned approaches. It demonstrates how to use Chronos-Bolt models (faster variants) and original Chronos models, incorporate covariates through regressor integration, and perform model fine-tuning. Key implementations include basic model setup, covariate handling, and fine-tuning configurations. The tutorial helps with tasks like time series prediction, model comparison, and optimization of forecasting accuracy. Notable features include CPU/GPU compatibility options, covariate integration techniques, customizable fine-tuning parameters, and visualization capabilities. It provides practical code examples for data preparation, model training, prediction generation, and performance evaluation.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Forecasting with Chronos in AutoGluon

## Key Concepts
- Chronos models are pretrained on large collections of time series data
- Two variants available:
  - Chronos-Bolt⚡️ (new, faster, more accurate)
  - Original Chronos models

## Implementation Details

### Installation
```python
!pip install autogluon.timeseries
```

### Model Presets
- **Chronos-Bolt Models** (Recommended):
  - `bolt_tiny`, `bolt_mini`, `bolt_small`, `bolt_base`
  - Can run on both CPU and GPU
  
- **Original Chronos Models**:
  - `chronos_tiny`, `chronos_mini`, `chronos_small`, `chronos_base`, `chronos_large`
  - Models `small` and above require GPU

### Basic Implementation
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data
data = TimeSeriesDataFrame.from_path("your_data_path.csv")

# Split data
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Create and fit predictor
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, 
    presets="bolt_small"
)

# Generate predictions
predictions = predictor.predict(train_data)
```

## Important Notes
1. Chronos models don't actually "fit" to data - computation happens during inference
2. Prediction computation scales linearly with number of time series
3. The `fit` method primarily:
   - Infers time series frequency
   - Saves predictor state
   - Handles internal configurations

## Best Practices
- Use Chronos-Bolt models for better performance and CPU compatibility
- Can combine Chronos with other time series models using presets:
  - `medium_quality`
  - `high_quality`
  - `best_quality`

## Visualization
```python
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],
    max_history_length=200
)
```

Here's the condensed tutorial section focusing on key implementation details and concepts:

# Incorporating Covariates with Chronos-Bolt

## Key Concepts
- Chronos is primarily a univariate model, but can incorporate external data through covariate regressors
- Covariate regressors are tabular models that predict target values using known covariates
- The process: regressor predictions are subtracted from target, then univariate model forecasts residuals

## Implementation Details

### 1. Data Setup
```python
data = TimeSeriesDataFrame.from_path("path_to_grocery_sales_data")
prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length=prediction_length)
```

### 2. Predictor Configuration
```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot configuration (without covariates)
            {
                "model_path": "bolt_small",
                "ag_args": {"name_suffix": "ZeroShot"},
            },
            # Configuration with CatBoost regressor
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=60,
)
```

## Important Configurations
- `known_covariates_names`: List of covariate columns to use
- `covariate_regressor`: Specifies the regressor type (e.g., "CAT" for CatBoost)
- `target_scaler`: Recommended when using covariate regressors for better accuracy

## Best Practices
1. Always apply target scaling when using covariate regressors
2. Use meaningful name suffixes to distinguish models in the leaderboard
3. Consider both zero-shot and regressor-based configurations for comparison

## Performance Note
Models using covariates typically produce more accurate forecasts compared to univariate-only approaches, as demonstrated by the test set evaluation.

Here's the condensed version focusing on the key implementation details and insights:

# Model Comparison and Best Practices

## Performance Comparison
```python
# Sample results table showing model performance
| model                            | score_test | score_val | pred_time_test | pred_time_val | fit_time_marginal |
|----------------------------------|------------|-----------|----------------|---------------|-------------------|
| ChronosWithRegressor[bolt_small] | -0.268969  | -0.358048| 0.881176       | 0.916053      | 1.004376         |
| ChronosZeroShot[bolt_small]      | -0.318562  | -0.452296| 0.859930       | 0.844927      | 0.019435         |
```

## Key Implementation Notes

1. **Model Selection**
   - Covariates may not always improve model performance
   - Zero-shot models might achieve better accuracy in some cases
   - Always evaluate multiple models on held-out data

2. **AutoGluon Integration**
   - Use AutoGluon's presets for automatic model selection:
     - `"high_quality"` preset
     - `"best_quality"` preset
   - These presets automatically compare and select the best performing model

## Best Practices
- Always compare models with and without covariates
- Use validation data to select the best performing model
- Consider prediction time requirements when selecting models
- Let AutoGluon handle model selection when possible through its quality presets

This condensed version maintains the critical implementation details while focusing on practical aspects of model selection and evaluation.

Here's the condensed tutorial section focusing on key implementation details and practices:

# Fine-tuning Chronos Models

## Key Implementation Details

### Basic Fine-tuning Setup
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot configuration
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            # Fine-tuned configuration
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,
    enable_ensemble=False,
)
```

### Custom Fine-tuning Parameters
```python
predictor.fit(
    ...,
    hyperparameters={
        "Chronos": {
            "fine_tune": True,
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 2000
        }
    }
)
```

## Important Notes

1. **Model Evaluation**: 
   - Use `predictor.leaderboard(test_data)` to compare model variants
   - Scores are reported in "higher is better" format (error metrics are multiplied by -1)

2. **Hardware Requirements**:
   - Recommended: AWS g5.2xlarge or p3.2xlarge instances
   - GPU: Minimum 16GB GPU memory
   - RAM: Minimum 32GB main memory
   - CPU: Possible but slower runtime

3. **Performance Characteristics**:
   - Zero-shot performance comparable to statistical baselines
   - Fine-tuned models typically achieve better accuracy
   - Comparable to deep learning models like TemporalFusionTransformer or PatchTST

## Best Practices

1. Start with zero-shot inference for quick baseline results
2. Use fine-tuning to improve accuracy on specific datasets
3. Adjust fine-tuning parameters (learning rate, steps) for optimal performance
4. Monitor validation scores to assess model improvements

## Support Resources
- AutoGluon Discord server
- GitHub repositories (AutoGluon and Chronos)
- Chronos documentation in Forecasting Model Zoo