# Condensed: AutoMM Presets

Summary: This tutorial demonstrates how to implement AutoMM's preset configurations (medium_quality, high_quality, and best_quality) for multimodal machine learning tasks using AutoGluon. It covers the implementation of MultiModalPredictor with different preset options, including their HPO variants for hyperparameter optimization. The tutorial helps with tasks involving model configuration, training setup, and performance tuning, showing how to balance between speed and accuracy requirements. Key features include preset-specific configurations, time limit handling, resource management, and evaluation metrics, making it particularly useful for developers needing to implement scalable multimodal learning solutions with different computational constraints.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# AutoMM Presets Tutorial

## Key Concepts
AutoMM provides three preset configurations to simplify hyperparameter setup:
- `medium_quality`: Fast training/inference, smaller models
- `high_quality`: Balanced performance and speed
- `best_quality`: Maximum performance, higher computational requirements
- HPO variants: Add `_hpo` suffix for hyperparameter optimization (e.g., `medium_quality_hpo`)

## Implementation

### Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
import warnings
warnings.filterwarnings('ignore')
```

### Medium Quality Implementation
```python
predictor = MultiModalPredictor(
    label='label', 
    eval_metric='acc', 
    presets="medium_quality"
)
predictor.fit(
    train_data=train_data,
    time_limit=20  # seconds
)
```

### High Quality Implementation
```python
predictor = MultiModalPredictor(
    label='label', 
    eval_metric='acc', 
    presets="high_quality"
)
predictor.fit(
    train_data=train_data,
    time_limit=20
)
```

### Best Quality Implementation
```python
predictor = MultiModalPredictor(
    label='label', 
    eval_metric='acc', 
    presets="best_quality"
)
predictor.fit(
    train_data=train_data,
    time_limit=180  # requires longer training time
)
```

## Important Configurations

### Viewing Preset Configurations
```python
from autogluon.multimodal.presets import get_automm_presets

# Get preset details
hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(
    problem_type="default", 
    presets="high_quality"
)
```

## Best Practices and Warnings

1. Time Limits:
   - Medium/High quality: Can work with shorter time limits
   - Best quality: Requires significantly more training time

2. Resource Requirements:
   - Best quality preset needs high-end GPUs with large memory
   - Medium quality works well with limited resources

3. HPO Considerations:
   - HPO variants tune key parameters including:
     - Model backbone
     - Batch size
     - Learning rate
     - Max epoch
     - Optimizer type

4. Performance vs Speed:
   - Medium quality: Fastest but lower performance
   - High quality: Balanced option
   - Best quality: Highest performance but slowest

## Evaluation
```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

For customization options, refer to the Customize AutoMM documentation.