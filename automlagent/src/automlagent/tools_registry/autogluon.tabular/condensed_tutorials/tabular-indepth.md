# Condensed: AutoGluon Tabular - In Depth

Summary: This tutorial provides comprehensive implementation guidance for AutoGluon's tabular machine learning capabilities, covering model training, optimization, and deployment. It demonstrates techniques for hyperparameter configuration, model ensembling, decision threshold calibration, prediction workflows, and model interpretation. The tutorial specifically addresses performance optimization through methods like model persistence, inference speed constraints, and memory management. Key functionalities include configuring neural networks and gradient boosting models, implementing stacking/bagging, feature importance analysis, and various optimization strategies for both training and inference phases. It's particularly useful for tasks involving automated machine learning pipeline setup, model performance optimization, and production deployment of tabular models.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# AutoGluon Tabular - Core Implementation Details

## Key Setup
```python
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common import space

# Load data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
```

## Critical Hyperparameter Configuration

### Neural Network Configuration
```python
nn_options = {
    'num_epochs': 10,
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),
}
```

### Gradient Boosting Configuration
```python
gbm_options = {
    'num_boost_round': 100,
    'num_leaves': space.Int(lower=26, upper=66, default=36),
}
```

### HPO Settings
```python
hyperparameters = {
    'GBM': gbm_options,
    'NN_TORCH': nn_options,
}

hyperparameter_tune_kwargs = {
    'num_trials': 5,
    'scheduler': 'local',
    'searcher': 'auto',
}
```

## Model Training
```python
predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(
    train_data,
    time_limit=2*60,  # 2 minutes
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

## Important Notes and Best Practices:

1. **Hyperparameter Tuning Warning**: Not recommended for most cases. Using `presets="best_quality"` typically achieves better performance.

2. **Validation Data**:
   - AutoGluon can automatically select validation data using stratified sampling
   - Manual validation data specification useful only when test distribution differs from training

3. **Performance Estimation**:
   - Validation metrics may be optimistic
   - Always evaluate on separate test data never seen during `fit()`

4. **Customization Options**:
   - Can specify single values or search spaces for hyperparameters
   - Unspecified parameters use AutoGluon defaults
   - Use `verbosity=3` for detailed training output

5. **Time Management**:
   - Example uses minimal settings for quick demonstration
   - Increase `time_limit`, `num_epochs`, and `num_boost_round` for better performance
   - Adjust `subsample_size` based on available computational resources

Here's the condensed version focusing on key implementation details and practices:

# Model Ensembling and Decision Threshold Calibration

## Stacking and Bagging Implementation

### Key Configuration Options
```python
predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    num_bag_folds=5,      # Number of folds for bagging
    num_bag_sets=1,       # Number of bagging iterations
    num_stack_levels=1    # Number of stacking levels
)
```

### Best Practices
- Don't provide `tuning_data` when using stacking/bagging
- Use `auto_stack=True` for automatic optimization of stacking/bagging parameters
- Increasing `num_bag_sets` may improve accuracy but increases training time and resource usage

### Advanced Configuration
```python
predictor = TabularPredictor(label=label, eval_metric='balanced_accuracy').fit(
    train_data, 
    auto_stack=True,
    calibrate_decision_threshold=False
)
```

## Decision Threshold Calibration

### Implementation
```python
# Manual calibration
calibrated_threshold = predictor.calibrate_decision_threshold()
predictor.set_decision_threshold(calibrated_threshold)

# Automatic calibration during training
predictor.fit(..., calibrate_decision_threshold=True)
```

### Key Features
- Can calibrate for specific metrics ('f1', 'balanced_accuracy', 'mcc')
- Default setting is `calibrate_decision_threshold="auto"`
- Affects binary classification metrics like F1 and balanced accuracy

### Prediction Methods
```python
# Standard prediction
y_pred = predictor.predict(test_data)

# Custom threshold prediction
y_pred_custom = predictor.predict(test_data, decision_threshold=0.8)

# Probability prediction
y_pred_proba = predictor.predict_proba(test_data)
```

### Important Notes
- Threshold calibration may improve one metric while degrading others
- Best to use default `"auto"` calibration in most cases
- Can specify different thresholds during prediction without retraining

Here's the condensed tutorial section focusing on key implementation details and practices:

# Prediction and Model Interpretation

## Loading and Making Predictions

```python
# Load saved predictor
predictor = TabularPredictor.load(save_path)

# Get required feature columns
predictor.features()

# Single prediction
datapoint = test_data_nolabel.iloc[[0]]  # Use [[0]] not [0] to get DataFrame
prediction = predictor.predict(datapoint)

# Get probability predictions
prob_predictions = predictor.predict_proba(datapoint)
```

## Model Selection and Evaluation

```python
# View best model
predictor.model_best

# Evaluate all models
predictor.leaderboard(test_data)

# Detailed leaderboard with extra metrics
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'])

# Use specific model for prediction
model_name = predictor.model_names()[0]
prediction = predictor.predict(datapoint, model=model_name)
```

### Important Notes:
- Metrics like `log_loss` are shown in `higher_is_better` form (negative values)
- `log_loss` can be `-inf` when used as secondary metric due to zero probabilities
- Avoid using `log_loss` as secondary metric for model comparison

## Model Evaluation Methods

```python
# Evaluate predictions with probabilities
y_pred_proba = predictor.predict_proba(test_data_nolabel)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)

# Shorthand evaluation (when labels are in test_data)
perf = predictor.evaluate(test_data)
```

## Feature Importance

```python
# Get feature importance scores
importance_scores = predictor.feature_importance(test_data)
```

### Key Points:
- Uses permutation-shuffling method
- Negative scores indicate potentially harmful features
- Useful for global model interpretation
- Local explanations available via Shapley values (separate examples)

### Best Practices:
1. Consider removing features with non-positive importance scores
2. Use feature importance for model interpretability
3. For deployment, copy entire `save_path` folder to new machine
4. Consider model-specific tradeoffs between accuracy and inference speed

Here's the condensed version focusing on key implementation details for accelerating inference in AutoGluon:

# Accelerating Inference in AutoGluon

## Key Optimization Methods (In Priority Order)

### With Bagging Enabled:
1. refit_full (8x-160x speedup)
2. persist (up to 10x speedup)
3. infer_limit (up to 50x speedup)

### Without Bagging:
1. persist
2. infer_limit

## Implementation Details

### 1. Model Persistence
```python
# Load all models into memory
predictor.persist()

# Make predictions
for i in range(num_test):
    datapoint = test_data_nolabel.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)

# Free memory
predictor.unpersist()
```

### 2. Inference Speed Constraints
```python
# Configure inference limits
predictor_infer_limit = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data=train_data,
    time_limit=30,
    infer_limit=0.00005,  # 0.05 ms per row
    infer_limit_batch_size=10000,  # batch size for inference
)
```

## Critical Parameters
- `infer_limit`: Time in seconds to predict 1 row
- `infer_limit_batch_size`: Batch size for inference calculations
  - Use 10000 for batch inference
  - Use 1 for online inference (harder to optimize)

## Best Practices
1. If bagging is enabled, always call `refit_full` after fitting
2. Persist models when making repeated predictions
3. Use batch inference when possible (larger `infer_limit_batch_size`)
4. Consider hardware optimization before complex manual tuning

## Important Warnings
- Inference limits below 0.02s with `infer_limit_batch_size=1` are difficult to achieve
- Model persistence requires sufficient memory
- Manual preprocessing and hyperparameter tuning are not recommended unless necessary

This condensed version maintains all critical implementation details while focusing on practical usage and best practices.

Here's the condensed version focusing on key implementation details and practices:

# Optimizing Model Inference Speed

## Testing Inference Speed
```python
# Test inference speed with batch prediction
test_data_batch = test_data.sample(infer_limit_batch_size, replace=True, ignore_index=True)

time_start = time.time()
predictor_infer_limit.predict(test_data_batch)
time_end = time.time()

infer_time_per_row = (time_end - time_start) / len(test_data_batch)
rows_per_second = 1 / infer_time_per_row
```

## Optimization Techniques

### 1. Using Smaller Ensembles
```python
# Create alternative ensembles with different speed-accuracy tradeoffs
additional_ensembles = predictor.fit_weighted_ensemble(expand_pareto_frontier=True)

# Select and use a faster model
model_for_prediction = additional_ensembles[0]
predictions = predictor.predict(test_data, model=model_for_prediction)
```

### 2. Collapsing Bagged Ensembles
```python
# Collapse multiple bagged models into single model
refit_model_map = predictor.refit_full()
```
**Key Benefit**: Reduces memory/latency requirements but may impact accuracy

### 3. Model Distillation
```python
# Train smaller models to mimic ensemble behavior
student_models = predictor.distill(time_limit=30)
preds_student = predictor.predict(test_data_nolabel, model=student_models[0])
```

### 4. Using Lightweight Configurations
```python
# Option 1: Use lightweight presets
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, 
    presets=['good_quality', 'optimize_for_deployment'],
    time_limit=30
)


...(truncated)