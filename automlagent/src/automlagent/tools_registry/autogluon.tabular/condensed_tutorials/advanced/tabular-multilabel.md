# Condensed: Predicting Multiple Columns in a Table (Multi-Label Prediction)

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information:

# Predicting Multiple Columns in a Table (Multi-Label Prediction)

## Overview
This tutorial demonstrates how to predict multiple columns (labels) in a table using AutoGluon. The approach uses separate TabularPredictor instances for each label, with optional correlation handling between labels.

## Key Implementation

### MultilabelPredictor Class
The core implementation uses a custom class that manages multiple TabularPredictors:

```python
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path

class MultilabelPredictor:
    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, 
                 consider_labels_correlation=True, **kwargs):
        # [Previous implementation code remains the same]
```

### Key Parameters
- `labels`: List of column names to predict
- `problem_types`: Prediction type for each label (regression/classification)
- `eval_metrics`: Evaluation metric for each label
- `consider_labels_correlation`: Whether to account for label correlations
- `path`: Save location for models

## Usage Example

### 1. Data Preparation
```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500
train_data = train_data.sample(n=subsample_size, random_state=0)

labels = ['education-num','education','class']
problem_types = ['regression','multiclass','binary']
eval_metrics = ['mean_absolute_error','accuracy','accuracy']
save_path = 'agModels-predictEducationClass'
```

### 2. Training
```python
multi_predictor = MultilabelPredictor(labels=labels, 
                                    problem_types=problem_types, 
                                    eval_metrics=eval_metrics, 
                                    path=save_path)
multi_predictor.fit(train_data, time_limit=5)
```

### 3. Prediction and Evaluation
```python
# Load test data
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data = test_data.sample(n=subsample_size, random_state=0)
test_data_nolab = test_data.drop(columns=labels)

# Make predictions
predictions = multi_predictor.predict(test_data_nolab)

# Evaluate
evaluations = multi_predictor.evaluate(test_data)
```

## Best Practices

1. Performance Optimization:
   - Use `presets='best_quality'` for best predictive performance
   - Set `consider_labels_correlation=False` if using individual predictors

2. Memory Management:
   - For memory issues: Adjust fit() arguments as per tabular-indepth tutorial
   - For faster inference: Use `presets = ['good_quality', 'optimize_for_deployment']`

3. Model Persistence:
   ```python
   # Save
   multi_predictor.save()
   
   # Load
   multi_predictor = MultilabelPredictor.load(save_path)
   ```

4. Individual Predictor Access:
   ```python
   predictor_single = multi_predictor.get_predictor('label_name')
   ```