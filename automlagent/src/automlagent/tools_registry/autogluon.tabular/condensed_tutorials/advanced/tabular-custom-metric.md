# Condensed: Adding a custom metric to AutoGluon

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information:

# Adding a Custom Metric to AutoGluon

## Key Concepts
- Custom metrics must be defined in separate Python files and imported to ensure serializability
- Metrics require specific parameters to work with AutoGluon's Scorer system
- Custom metrics can be used for validation scores, model ensembling, and hyperparameter tuning

## Implementation Requirements

### Setup
```python
import numpy as np
import sklearn.metrics
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset, TabularPredictor

# Generate sample data
rng = np.random.default_rng(seed=42)
y_true = rng.integers(low=0, high=2, size=10)
y_pred = rng.integers(low=0, high=2, size=10)
```

### Creating Custom Metrics

Key Parameters for `make_scorer`:
- `name`: Identifier for the metric
- `score_func`: The actual metric function
- `optimum`: Best possible value for the metric
- `greater_is_better`: Whether higher scores are better
- `needs_*`: Type of predictions required (`needs_pred`, `needs_proba`, `needs_class`, `needs_threshold`, `needs_quantile`)

1. Accuracy Metric Example:
```python
ag_accuracy_scorer = make_scorer(
    name='accuracy',
    score_func=sklearn.metrics.accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True
)
```

2. Mean Squared Error Example:
```python
ag_mean_squared_error_scorer = make_scorer(
    name='mean_squared_error',
    score_func=sklearn.metrics.mean_squared_error,
    optimum=0,
    greater_is_better=False
)
```

3. ROC AUC Example:
```python
ag_roc_auc_scorer = make_scorer(
    name='roc_auc',
    score_func=sklearn.metrics.roc_auc_score,
    optimum=1,
    greater_is_better=True,
    needs_threshold=True
)
```

### Using Custom Metrics with TabularPredictor

```python
# Load data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'

# Method 1: Using extra_metrics in leaderboard
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters='toy')
predictor.leaderboard(test_data, extra_metrics=[ag_roc_auc_scorer, ag_accuracy_scorer])

# Method 2: Using eval_metric during initialization
predictor_custom = TabularPredictor(label=label, eval_metric=ag_roc_auc_scorer).fit(train_data, hyperparameters='toy')
predictor_custom.leaderboard(test_data)
```

## Important Notes
- Custom metrics must be pickleable for parallel training
- Define metrics in separate Python files (e.g., `my_metrics.py`)
- Scores are internally converted to `greater_is_better=True` format
- Error values are in `lower_is_better` format
- Custom metric functions must accept `y_true` and `y_pred` as numpy arrays and return a float

## Best Practices
- Correctly specify `greater_is_better` to ensure proper optimization
- Choose appropriate `needs_*` parameter based on metric type
- Set correct `optimum` value for error/score calculations
- Consider contributing useful custom metrics to AutoGluon via PR