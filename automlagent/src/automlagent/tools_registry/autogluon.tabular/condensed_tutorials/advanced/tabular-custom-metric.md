# Condensed: Adding a custom metric to AutoGluon

Summary: This tutorial demonstrates how to implement custom evaluation metrics in AutoGluon using the make_scorer function. It covers the technical implementation of creating serializable custom metrics, including essential parameters like score_func, optimum, and greater_is_better, along with various needs_* flags for different prediction types. The tutorial helps with tasks like integrating custom accuracy and MSE metrics into TabularPredictor for model evaluation and training. Key features include proper metric serialization techniques, parameter configurations for different metric types, and best practices for implementing score functions that work with numpy arrays and return float values.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Adding Custom Metrics to AutoGluon

## Key Implementation Details

### Creating a Custom Metric
Custom metrics must be defined in a separate Python file and imported to ensure they are serializable (pickleable).

```python
from autogluon.core.metrics import make_scorer

# Basic structure for creating a custom scorer
custom_scorer = make_scorer(
    name='metric_name',
    score_func=metric_function,
    optimum=optimal_value,
    greater_is_better=True/False,
    needs_pred/needs_proba/needs_class/needs_threshold/needs_quantile=True
)
```

### Critical Parameters for make_scorer
- `name`: Identifier for the scorer
- `score_func`: Function that calculates the metric
- `optimum`: Best possible value for the metric
- `greater_is_better`: Whether higher scores are better
- `needs_*`: Specify prediction type required:
  - `needs_pred`: Regular predictions (regression)
  - `needs_proba`: Probability predictions
  - `needs_class`: Class predictions
  - `needs_threshold`: Continuous decision certainty (binary classification)
  - `needs_quantile`: Quantile predictions

## Implementation Examples

### Custom Accuracy Metric
```python
ag_accuracy_scorer = make_scorer(
    name='accuracy',
    score_func=sklearn.metrics.accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True
)
```

### Custom MSE Metric
```python
def mse_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

ag_mse_scorer = make_scorer(
    name='mean_squared_error',
    score_func=mse_func,
    optimum=0,
    greater_is_better=False
)
```

### Using Custom Metrics with TabularPredictor
```python
# Using extra_metrics in leaderboard
predictor.leaderboard(test_data, extra_metrics=[custom_scorer])

# Using custom eval_metric during training
predictor = TabularPredictor(
    label=label, 
    eval_metric=custom_scorer
).fit(train_data)
```

## Important Warnings and Best Practices

1. Custom metrics must be defined in separate Python files to be serializable
2. Non-serializable metrics will crash during parallel training
3. Ensure `greater_is_better` is set correctly to avoid optimizing for worst models
4. Only one `needs_*` parameter can be True at a time
5. Score functions must accept `y_true` and `y_pred` as numpy arrays and return a float
6. AutoGluon Scorers internally convert all metrics to `greater_is_better=True` format