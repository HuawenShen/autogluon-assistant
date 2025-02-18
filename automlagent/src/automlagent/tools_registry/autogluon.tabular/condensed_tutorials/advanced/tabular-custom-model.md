# Condensed: Adding a custom model to AutoGluon

Summary: This tutorial provides implementation guidance for adding custom models to AutoGluon's TabularPredictor framework. It demonstrates how to create models inheriting from AbstractModel class, implement essential methods like _preprocess and _fit, and integrate with AutoGluon's ecosystem. Key techniques covered include feature preprocessing using LabelEncoderFeatureGenerator, handling both classification and regression tasks, model serialization, bagged ensemble implementation, and hyperparameter tuning. The tutorial shows how to leverage AutoGluon's built-in features like AutoMLPipelineFeatureGenerator and TabularPredictor for production deployments, while offering reference implementations for advanced features such as time limits, GPU support, and custom HPO through examples from existing models like LGBModel and RFModel.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on the key implementation aspects:

# Adding a Custom Model to AutoGluon

## Key Points
- Custom models must inherit from `AbstractModel` class
- Models need to follow AutoGluon's API to work with other models
- Implementation requires understanding of core functionality and interfaces

## Reference Implementations for Advanced Features

| Feature | Reference Model |
| ------- | -------------- |
| Time limits & early stopping | `LGBModel`, `RFModel` |
| Memory usage limits | `LGBModel`, `RFModel` |
| Sample weights | `LGBModel` |
| Validation & metrics | `LGBModel` |
| GPU support | `LGBModel` |
| Non-serializable models | `NNFastAiTabularModel` |
| Advanced problem types | `RFModel` |
| Text features | `TextPredictorModel` |
| Image features | `ImagePredictorModel` |
| Lazy imports | `LGBModel` |
| Custom HPO | `LGBModel` |

## Best Practices
1. Review the [AbstractModel source code](https://auto.gluon.ai/stable/_modules/autogluon/core/models/abstract/abstract_model.html) before implementation
2. Study reference implementations for specific features needed
3. Start with basic functionality and gradually add advanced features
4. Ensure compatibility with AutoGluon's existing model ecosystem

## Prerequisites
- Understanding of AutoGluon basics ([Quick Start Guide](../tabular-quick-start.ipynb))
- Familiarity with model implementation patterns
- Knowledge of the specific ML algorithm to be implemented

This condensed version maintains the critical implementation guidance while removing introductory and redundant content.

Here's the condensed version focusing on key implementation details:

# Custom Model Implementation in AutoGluon

## Key Implementation Details

### CustomRandomForestModel Class
```python
class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
```

### Critical Methods

1. **_preprocess**
```python
def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
    X = super()._preprocess(X, **kwargs)
    if is_train:
        self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
        self._feature_generator.fit(X=X)
    if self._feature_generator.features_in:
        X = X.copy()
        X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
    return X.fillna(0).to_numpy(dtype=np.float32)
```

2. **_fit**
```python
def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    model_cls = RandomForestRegressor if self.problem_type in ['regression', 'softclass'] else RandomForestClassifier
    X = self.preprocess(X, is_train=True)
    params = self._get_model_params()
    self.model = model_cls(**params)
    self.model.fit(X, y)
```

### Configuration Methods

1. **Default Parameters**
```python
def _set_default_params(self):
    default_params = {
        'n_estimators': 300,
        'n_jobs': -1,
        'random_state': 0,
    }
```

2. **Auxiliary Parameters**
```python
def _get_default_auxiliary_params(self):
    return {
        'valid_raw_types': ['int', 'float', 'category']
    }
```

## Data Preprocessing

### Label Cleaning
```python
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)
```

## Best Practices
1. Import model dependencies inside methods for modularity
2. Always call preprocess on data during fit and predict
3. Handle missing values appropriately (using fillna or algorithm-specific handling)
4. Specify valid data types in auxiliary parameters
5. Use label cleaning for classification tasks

## Important Notes
- The model must implement `_fit` and `_preprocess` methods
- Categorical features are handled using LabelEncoderFeatureGenerator
- The implementation supports both classification and regression
- Data type validation is handled through auxiliary parameters

Here's the condensed tutorial focusing on key implementation details and practices:

# Custom Model Implementation in AutoGluon - Part 3

## Feature Cleaning
```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

# Clean features using AutoGluon's feature generator
feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)
```

**Note**: AutoMLPipelineFeatureGenerator doesn't handle:
- Missing value imputation for numeric features
- Feature scaling
- One-hot encoding for categoricals

## Model Training and Prediction
```python
# Train model
custom_model = CustomRandomForestModel()
custom_model.fit(X=X_clean, y=y_clean)

# Save/Load model
# custom_model.save()
# custom_model = CustomRandomForestModel.load(path=load_path)

# Predict
X_test_clean = feature_generator.transform(X_test)
y_pred = custom_model.predict(X_test_clean)
y_pred_orig = label_cleaner.inverse_transform(y_pred)
```

## Bagged Ensemble Implementation
```python
from autogluon.core.models import BaggedEnsembleModel

bagged_custom_model = BaggedEnsembleModel(CustomRandomForestModel())
bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local'  # Required for class not in separate module
bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)
```

## Integration with TabularPredictor

### Basic Training
```python
from autogluon.tabular import TabularPredictor

# Train multiple versions with different hyperparameters
custom_hyperparameters = {
    CustomRandomForestModel: [
        {}, 
        {'max_depth': 10}, 
        {'max_features': 0.9, 'max_depth': 20}
    ]
}
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)
```

### Hyperparameter Tuning
```python
from autogluon.common import space

# Define search space
custom_hyperparameters_hpo = {
    CustomRandomForestModel: {
        'max_depth': space.Int(lower=5, upper=30),
        'max_features': space.Real(lower=0.1, upper=1.0),
        'criterion': space.Categorical('gini', 'entropy'),
    }
}

# Train with HPO
predictor = TabularPredictor(label=label).fit(
    train_data,
    hyperparameters=custom_hyperparameters_hpo,
    hyperparameter_tune_kwargs='auto',
    time_limit=20
)
```

## Key Best Practices
1. Use AutoMLPipelineFeatureGenerator for consistent feature preprocessing
2. Consider bagging for improved model performance
3. Implement proper model saving/loading functionality
4. Use TabularPredictor for production deployments
5. Leverage hyperparameter tuning for optimal performance

This condensed version maintains all critical implementation details while removing verbose explanations and redundant code examples.

Here's the condensed version of the final chunk focusing on key implementation details:

# Adding Custom Model with Tuned Hyperparameters

## Key Implementation Steps

1. Add tuned custom model to default models:
```python
# Add custom model with optimized hyperparameters
custom_hyperparameters = get_hyperparameter_config('default')
custom_hyperparameters[CustomRandomForestModel] = best_model_info['hyperparameters']
```

2. Train predictor with custom configuration:
```python
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters
)
```

## Important Variations

- For enhanced performance, can use custom model in stack ensemble:
```python
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters,
    presets='best_quality'  # Enables stack ensemble
)
```

## Best Practices

1. Evaluate model performance using leaderboard:
```python
predictor.leaderboard(test_data)
```

2. Consider contributing custom models via PR to AutoGluon repository

## Additional Resources

- Basic tutorials:
  - "Predicting Columns in a Table - Quick Start"
  - "Predicting Columns in a Table - In Depth"
- Advanced custom models: "Adding a custom model to AutoGluon (Advanced)"

This concludes the implementation of custom models in AutoGluon's TabularPredictor.