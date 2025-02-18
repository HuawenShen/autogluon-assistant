# Condensed: Hyperparameter Optimization in AutoMM

Summary: This tutorial demonstrates hyperparameter optimization (HPO) implementation in AutoGluon's MultiModalPredictor, specifically for image classification tasks. It covers the integration of Ray Tune for HPO, showing how to define search spaces for parameters like learning rate and model checkpoints, configure searchers (random/bayes) and schedulers (FIFO/ASHA), and implement the optimization process. The tutorial helps with tasks involving automated model tuning and provides code for both standard model fitting and HPO-enhanced training. Key features include customizable search spaces, multiple optimization strategies, checkpoint management, and integration of popular model architectures like GhostNet and MobileNetV3.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and practices:

# Hyperparameter Optimization in AutoMM

## Key Implementation Details

### Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
from ray import tune

# Load and prepare data
train_data, test_data = shopee_dataset(download_dir)
```

### Standard Model Fitting
```python
predictor_regular = MultiModalPredictor(label="label")
predictor_regular.fit(
    train_data=train_data,
    hyperparameters={"model.timm_image.checkpoint_name": "ghostnet_100"}
)
```

### HPO Implementation
```python
predictor_hpo = MultiModalPredictor(label="label")

# Define search space
hyperparameters = {
    "optimization.learning_rate": tune.uniform(0.00005, 0.001),
    "model.timm_image.checkpoint_name": tune.choice([
        "ghostnet_100",
        "mobilenetv3_large_100"
    ])
}

# Configure HPO settings
hyperparameter_tune_kwargs = {
    "searcher": "bayes",  # Options: random, bayes
    "scheduler": "ASHA",  # Options: FIFO, ASHA
    "num_trials": 2,
    "num_to_keep": 3,  # Number of checkpoints to keep per trial
}

# Fit with HPO
predictor_hpo.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

## Critical Configurations

### Supported HPO Parameters
- `optimization.learning_rate`
- `optimization.optim_type`
- `optimization.max_epochs`
- `model.timm_image.checkpoint_name`

### Search Strategy Options
1. **Searcher Types**:
   - `random`
   - `bayes`

2. **Scheduler Types**:
   - `FIFO`
   - `ASHA`

## Best Practices

1. Define appropriate search spaces based on domain knowledge
2. Balance number of trials with available computing resources
3. Keep sufficient checkpoints (`num_to_keep >= 1`)
4. Use Bayesian optimization for better search efficiency
5. Monitor best trial results during training

## Important Notes
- HPO helps find optimal hyperparameter combinations for better model performance
- Uses Ray Tune in the backend for optimization
- Supports both Ray Tune and AutoGluon search spaces
- Validation accuracy is used to select the best hyperparameter combination

For more customization options, refer to the AutoMM customization documentation.