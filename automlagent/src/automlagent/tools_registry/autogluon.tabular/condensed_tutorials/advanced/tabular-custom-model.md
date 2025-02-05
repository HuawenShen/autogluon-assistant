# Condensed: Adding a custom model to AutoGluon

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information:

# Adding a Custom Model to AutoGluon

## Key Concepts
- Custom models must inherit from `AbstractModel` class
- Required method implementations: `_fit` and `_preprocess`
- Models can be used standalone or integrated with TabularPredictor

## Core Implementation

### Basic Custom Model Structure
```python
class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model_cls = RandomForestRegressor if self.problem_type in ['regression', 'softclass'] else RandomForestClassifier
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'n_jobs': -1,
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

## Usage Patterns

### Standalone Usage
```python
# Data preparation
X_clean = feature_generator.fit_transform(X)
y_clean = label_cleaner.transform(y)

# Model training
custom_model = CustomRandomForestModel()
custom_model.fit(X=X_clean, y=y_clean)
```

### With TabularPredictor
```python
# Basic usage
custom_hyperparameters = {CustomRandomForestModel: [{}, {'max_depth': 10}]}
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)

# With hyperparameter tuning
custom_hyperparameters_hpo = {CustomRandomForestModel: {
    'max_depth': space.Int(lower=5, upper=30),
    'max_features': space.Real(lower=0.1, upper=1.0),
    'criterion': space.Categorical('gini', 'entropy'),
}}
predictor = TabularPredictor(label=label).fit(
    train_data,
    hyperparameters=custom_hyperparameters_hpo,
    hyperparameter_tune_kwargs='auto',
    time_limit=20
)
```

## Important Implementation Notes
- Models must handle data preprocessing in `_preprocess` method
- Support for different problem types (regression/classification) should be implemented
- Model parameters should be defined in `_set_default_params`
- Valid data types should be specified in `_get_default_auxiliary_params`

## Advanced Features
- Bagging support through `BaggedEnsembleModel`
- Integration with AutoGluon's ensemble stacking
- Hyperparameter optimization support
- Can be combined with default AutoGluon models

For more advanced implementations, refer to source code of existing models like LGBModel or RFModel.