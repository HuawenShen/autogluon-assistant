# Condensed: AutoMM Presets

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the AutoMM Presets tutorial:

# AutoMM Presets

This tutorial covers AutoMM's preset configurations for simplifying hyperparameter selection in deep learning models.

## Key Concepts
- AutoMM offers three main preset levels: `medium_quality`, `high_quality`, and `best_quality`
- Each preset can be extended with HPO (hyperparameter optimization) by adding `_hpo` suffix
- Presets balance trade-offs between model performance, training speed, and resource requirements

## Implementation

### Setup


```python
!pip install autogluon.multimodal
import warnings
warnings.filterwarnings('ignore')
```

### Data Preparation
Using Stanford Sentiment Treebank (SST) dataset for binary classification of movie reviews:

```python
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000
train_data = train_data.sample(n=subsample_size, random_state=0)
```

### Preset Usage

#### 1. Medium Quality
Best for fast training and inference:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="medium_quality")
predictor.fit(
    train_data=train_data,
    time_limit=20
)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

#### 2. High Quality
Balanced performance and speed:

```python
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="high_quality")
predictor.fit(
    train_data=train_data,
    time_limit=20
)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

#### 3. Best Quality
Optimized for maximum performance:

```python
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="best_quality")
predictor.fit(train_data=train_data, time_limit=180)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

### Viewing Preset Configurations

To inspect preset hyperparameters:

```python
from autogluon.multimodal.presets import get_automm_presets

hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(
    problem_type="default", 
    presets="high_quality"
)
```

## Important Notes

1. Resource Requirements:
   - `medium_quality`: Lowest resource requirements
   - `high_quality`: Moderate resources needed
   - `best_quality`: Requires high-end GPUs with large memory

2. HPO Options:
   - Available presets: `medium_quality_hpo`, `high_quality_hpo`, `best_quality_hpo`
   - HPO tunes key parameters like model backbone, batch size, learning rate, epochs, and optimizer

3. Time Considerations:
   - Training time increases significantly from medium to best quality
   - Adjust time_limit parameter based on dataset size and preset choice

For customization options, refer to the Customize AutoMM tutorial.