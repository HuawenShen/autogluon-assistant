# Condensed: How to use AutoGluon for Kaggle competitions

*This is a condensed version that preserves essential implementation details and context.*

# Using AutoGluon for Kaggle Competitions

## Setup and Data Preparation
1. Install requirements:
```bash
pip install kaggle
```

2. Set up Kaggle API:
- Create account at kaggle.com/account
- Download API token to `~/.kaggle/kaggle.json`

3. Download competition data:
```bash
kaggle competitions download -c [COMPETITION]
```

## Core Implementation Steps

### 1. Data Loading and Preprocessing
```python
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

# Load data
directory = '~/IEEEfraud/'
label = 'isFraud'
eval_metric = 'roc_auc'
save_path = directory + 'AutoGluonModels/'

# Load and merge data
train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')
train_data = pd.merge(train_transaction, train_identity, 
                     on='TransactionID', how='left')
```

### 2. Model Training
```python
# Train model with best quality preset
predictor = TabularPredictor(label=label, 
                           eval_metric=eval_metric, 
                           path=save_path, 
                           verbosity=3).fit(
    train_data, 
    presets='best_quality', 
    time_limit=3600
)

results = predictor.fit_summary()
```

### 3. Prediction Generation
```python
# Load and prepare test data
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, 
                    on='TransactionID', how='left')

# Generate predictions
y_predproba = predictor.predict_proba(test_data, as_multiclass=False)
```

### 4. Submission Creation
```python
# Create submission file
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.to_csv(directory+'my_submission.csv', index=False)
```

## Key Best Practices

1. **Evaluation Metrics**:
- Specify competition-specific evaluation metric in `TabularPredictor`
- For classification tasks, verify prediction class alignment:
```python
predictor.positive_class  # binary classification
predictor.class_labels    # multiclass classification
```

2. **Model Optimization**:
- Use `presets='best_quality'` for maximum accuracy
- Increase `time_limit` for better results
- Focus on feature engineering over hyperparameter tuning

3. **Time-Series Data**:
- For temporal data, reserve recent samples as validation set
- Ensure train/test data preprocessing steps match exactly

## Submission
```bash
kaggle competitions submit -c [COMPETITION] -f [FILE] -m ["MESSAGE"]
```

## Common Pitfalls
- Verify data permissions and file access
- Ensure consistent data preprocessing between train and test
- Validate prediction format matches competition requirements
- Check Kaggle API setup for submission issues