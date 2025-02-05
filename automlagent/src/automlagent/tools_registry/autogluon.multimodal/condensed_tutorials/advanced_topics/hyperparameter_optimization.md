# Condensed: Hyperparameter Optimization in AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Hyperparameter Optimization in AutoMM

## Overview
This tutorial demonstrates hyperparameter optimization (HPO) in AutoMM for improving model performance through automated parameter tuning.

## Dataset Setup
```python
from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = './ag_automm_tutorial_hpo'
train_data, test_data = shopee_dataset(download_dir)
train_data = train_data.sample(frac=0.5)
```
Dataset contains clothing images with categories: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

## Basic Model Training
Baseline implementation without HPO:
```python
from autogluon.multimodal import MultiModalPredictor
predictor_regular = MultiModalPredictor(label="label")
predictor_regular.fit(
    train_data=train_data,
    hyperparameters = {"model.timm_image.checkpoint_name": "ghostnet_100"}
)
```

## Hyperparameter Optimization Implementation

### Key Components
1. **Search Space Definition**:
```python
hyperparameters = {
    "optimization.learning_rate": tune.uniform(0.00005, 0.005),
    "optimization.optim_type": tune.choice(["adamw", "sgd"]),
    "optimization.max_epochs": tune.choice(["10", "20"]),
    "model.timm_image.checkpoint_name": tune.choice(["swin_base_patch4_window7_224", "convnext_base_in22ft1k"])
}
```

2. **HPO Configuration**:
- Searcher: `"random"` or `"bayes"`
- Scheduler: `"FIFO"` or `"ASHA"`
- `num_trials`: Number of HPO iterations
- `num_to_keep`: Checkpoints per trial (minimum 1)

### Implementation Example
```python
from ray import tune

predictor_hpo = MultiModalPredictor(label="label")

hyperparameters = {
    "optimization.learning_rate": tune.uniform(0.00005, 0.001),
    "model.timm_image.checkpoint_name": tune.choice(["ghostnet_100",
                                                   "mobilenetv3_large_100"])
}
hyperparameter_tune_kwargs = {
    "searcher": "bayes",
    "scheduler": "ASHA",
    "num_trials": 2,
    "num_to_keep": 3,
}

predictor_hpo.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

### Evaluation
```python
scores_hpo = predictor_hpo.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores_hpo["accuracy"])
```

## Important Notes
- HPO uses Ray Tune backend
- Supports both Ray Tune and AutoGluon search spaces
- Best trial parameters are automatically selected based on validation accuracy
- More customization options available in the [Customize AutoMM](customization.ipynb) guide

This condensed version maintains all critical implementation details while removing redundant explanations and focusing on practical usage.