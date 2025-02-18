# Condensed: Adding a custom time series forecasting model

Summary: This tutorial demonstrates how to implement a custom NHITS (Neural Hierarchical Interpolation for Time Series) model wrapper for AutoGluon's time series forecasting framework. It provides implementation details for handling missing values, configuring model parameters, GPU acceleration, and time limits. The tutorial covers key techniques for integrating deep learning models with AutoGluon, including data preprocessing, model initialization, and training workflows. It specifically helps with tasks like subclassing AbstractTimeSeriesModel, implementing required methods (_fit, _predict, preprocess), and properly handling covariates (known, past, and static). Key features include automatic GPU detection, quantile-based forecasting, missing value handling, and support for multiple hyperparameter configurations when used with TimeSeriesPredictor.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details:

```python
# Key Implementation: NHITS Model Wrapper for AutoGluon
class NHITSModel(AbstractTimeSeriesModel):
    # Model capability flags
    _supports_known_covariates = True
    _supports_past_covariates = True
    _supports_static_features = True

    def preprocess(self, data, known_covariates=None, is_train=False, **kwargs):
        """Handle missing values preprocessing"""
        data = data.fill_missing_values()  # Forward/backward fill
        data = data.fill_missing_values(method="constant", value=0.0)  # Fill remaining with 0
        return data, known_covariates

    def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
        """Core model fitting implementation"""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS

        # Default configuration
        default_model_params = {
            'loss': MQLoss(quantiles=self.quantile_levels),
            'input_size': 2 * self.prediction_length,
            'scaler_type': "standard",
            'enable_progress_bar': False,
            'enable_model_summary': False,
            'logger': False,
            'accelerator': "cpu",
            'start_padding_enabled': True,
            'futr_exog_list': self.metadata.known_covariates_real,
            'hist_exog_list': self.metadata.past_covariates_real,
            'stat_exog_list': self.metadata.static_features_real
        }

        # GPU support
        if torch.cuda.is_available():
            default_model_params.update({
                'accelerator': "gpu",
                'devices': 1
            })

        # Time limit handling
        if time_limit:
            default_model_params['max_time'] = {"seconds": time_limit}

        # Merge with user params
        model_params = default_model_params | self._get_model_params()
        
        # Initialize and fit model
        model = NHITS(h=self.prediction_length, **model_params)
        self.nf = NeuralForecast(models=[model], freq=self.freq)
        
        train_df, static_df = self._to_neuralforecast_format(train_data)
        self.nf.fit(
            train_df,
            static_df=static_df,
            id_col="item_id",
            time_col="timestamp",
            target_col=self.target
        )
```

### Key Points:

1. **Data Preprocessing**:
   - Handles missing values through forward/backward fill
   - Falls back to zero-filling for completely missing series
   - Converts data to NeuralForecast format

2. **Model Configuration**:
   - Supports known, past, and static covariates
   - Automatically handles GPU if available
   - Respects time limits during training
   - Uses quantile-based loss function

3. **Important Notes**:
   - Only supports real-valued covariates (no categorical)
   - Requires data preprocessing due to NaN handling limitations
   - Implements time series length flexibility with padding

4. **Best Practices**:
   - Lazy imports inside methods for better dependency handling
   - Proper time limit implementation
   - Clear separation of default and user parameters
   - Handles GPU acceleration automatically

### Usage Warning:
- Categorical covariates are not supported and will be dropped
- All missing values must be handled in preprocessing
- Time series data must be in correct format with proper column names

Here's the condensed tutorial focusing on key implementation details and practices:

# Adding Custom Time Series Forecasting Models in AutoGluon

## Key Implementation Requirements

1. Create a subclass of `AbstractTimeSeriesModel`
2. Implement required methods:
   - `_fit`
   - `_predict`
   - `preprocess` (if custom preprocessing needed)

## Critical Code Components

```python
class NHITSModel(AbstractTimeSeriesModel):
    def __init__(
        self,
        prediction_length: int,
        target: str,
        metadata: dict,
        freq: Optional[str] = None,
        quantile_levels: Optional[list] = None,
        # ... other parameters
    ):
        super().__init__(
            prediction_length=prediction_length,
            freq=freq,
            metadata=metadata,
            target=target,
        )
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
```

## Important Usage Patterns

### 1. Standalone Mode
```python
# Initialize model
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)

# Train
model.fit(train_data=data, time_limit=20)

# Predict
predictions = model.predict(past_data, known_covariates)
```

### 2. Integration with TimeSeriesPredictor
```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)

# Use custom model alongside built-in models
predictor.fit(
    train_data,
    hyperparameters={
        "Naive": {},
        "Chronos": {"model_path": "bolt_small"},
        "ETS": {},
        NHITSModel: {},  # Custom model
    },
    time_limit=120,
)
```

## Multiple Hyperparameter Configurations
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

## Best Practices and Warnings

1. Custom model implementations rely on private AutoGluon APIs that may change
2. Update custom implementations when upgrading AutoGluon versions
3. When using with TimeSeriesPredictor, it handles:
   - Model configuration
   - Data preprocessing
   - Time limits

## Prerequisites
```python
!pip install neuralforecast==2.0
!pip install autogluon.timeseries
```

This implementation allows the custom model to be:
- Trained and tuned alongside default models
- Evaluated using standard metrics
- Used with predictor features like feature importance
- Configured with multiple hyperparameter sets