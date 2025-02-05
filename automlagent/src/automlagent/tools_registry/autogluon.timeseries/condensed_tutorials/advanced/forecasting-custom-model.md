# Condensed: Adding a custom time series forecasting model

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# Adding a Custom Time Series Forecasting Model

This tutorial demonstrates how to create a custom forecasting model compatible with AutoGluon's TimeSeriesPredictor.

## Key Requirements
- Custom model must subclass `AbstractTimeSeriesModel`
- Must implement `_fit` and `_predict` methods
- Optionally implement `preprocess` for custom data preprocessing

## Installation

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system
!pip install -q neuralforecast==2.0
```

## Custom Model Implementation

Key components of the custom model wrapper:

```python
class NHITSModel(AbstractTimeSeriesModel):
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True

    def preprocess(self, data, known_covariates=None, is_train=False, **kwargs):
        # Handle missing values
        data = data.fill_missing_values()
        data = data.fill_missing_values(method="constant", value=0.0)
        return data, known_covariates

    def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
        # Model configuration and training
        default_model_params = {
            'input_size': 2 * self.prediction_length,
            'scaler_type': "standard",
            # ... other parameters
        }
        model_params = default_model_params | self._get_model_params()
        model = NHITS(h=self.prediction_length, **model_params)
        self.nf = NeuralForecast(models=[model], freq=self.freq)
        
        train_df, static_df = self._to_neuralforecast_format(train_data)
        self.nf.fit(train_df, static_df=static_df)

    def _predict(self, data, known_covariates=None, **kwargs):
        # Generate predictions
        df, static_df = self._to_neuralforecast_format(data)
        predictions = self.nf.predict(df, static_df=static_df)
        # Format predictions as required
        return TimeSeriesDataFrame(predictions)
```

## Critical Implementation Requirements

1. Input Data Constraints:
- Sorted index by (item_id, timestamp)
- Regular frequency timestamps
- May contain missing values (NaN)
- Covariates must not have missing values

2. Output Requirements:
- Return TimeSeriesDataFrame
- Include columns for mean and quantile forecasts
- No missing values or gaps
- Match prediction_length and frequency

## Usage Examples

### 1. Standalone Mode (for development)

```python
# Configure model
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)

# Train and predict
model.fit(train_data=data, time_limit=20)
predictions = model.predict(past_data, known_covariates)
```

### 2. Integration with TimeSeriesPredictor

```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)

# Train with multiple models including custom model
predictor.fit(
    train_data,
    hyperparameters={
        "Naive": {},
        "Chronos": {"model_path": "bolt_small"},
        NHITSModel: {},
    },
    time_limit=120,
)
```

### Multiple Configurations

```python
predictor.fit(
    train_data,
    hyperparameters={
        NHITSModel: [
            {},  # default
            {"input_size": 20},
            {"scaler_type": "robust"},
        ]
    },
    time_limit=60,
)
```

## Best Practices
- Implement time_limit handling in _fit
- Don't modify input data in-place
- Handle edge cases (all NaNs, single observations)
- Lazy import dependencies inside _fit
- Properly handle categorical and numerical features

This implementation can serve as a template for adding other custom forecasting models to AutoGluon.