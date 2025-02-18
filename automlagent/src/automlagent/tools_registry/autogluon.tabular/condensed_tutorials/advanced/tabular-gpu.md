# Condensed: Training models with GPU support

Summary: This tutorial demonstrates GPU-enabled model training in AutoGluon, focusing on resource allocation at multiple levels (predictor, bagged model, and base model). It covers implementation techniques for configuring GPU usage through the TabularPredictor API, including basic single-GPU allocation and model-specific GPU assignments. Key functionalities include hierarchical resource management, CUDA toolkit integration, special LightGBM GPU installation requirements, and advanced configurations for distributed training. The tutorial helps with tasks like optimizing GPU resource allocation, setting up parallel training processes, and managing computational resources across different model components in AutoGluon's machine learning pipeline.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Training Models with GPU Support in AutoGluon

## Basic GPU Usage
```python
# Basic GPU allocation
predictor = TabularPredictor(label=label).fit(
    train_data,
    num_gpus=1  # Allocate 1 GPU for Tabular Predictor
)

# Model-specific GPU allocation
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # CPU training
        {'ag_args_fit': {'num_gpus': 1}}   # GPU training
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    num_gpus=1,
    hyperparameters=hyperparameters
)
```

## Important Notes
- CUDA toolkit required for GPU training
- LightGBM requires special installation for GPU support:
  ```bash
  pip uninstall lightgbm -y
  pip install lightgbm --install-option=--gpu
  ```

## Advanced Resource Allocation
Three levels of resource control:
1. Predictor level: `num_cpus`, `num_gpus`
2. Bagged model level: `ag_args_ensemble: ag_args_fit`
3. Base model level: `ag_args_fit`

### Example Configuration
```python
predictor.fit(
    num_cpus=32,
    num_gpus=4,
    hyperparameters={'NN_TORCH': {}},
    num_bag_folds=2,
    ag_args_ensemble={
        'ag_args_fit': {
            'num_cpus': 10,
            'num_gpus': 2,
        }
    },
    ag_args_fit={
        'num_cpus': 4,
        'num_gpus': 0.5,
    },
    hyperparameter_tune_kwargs={
        'searcher': 'random',
        'scheduler': 'local',
        'num_trials': 2
    }
)
```

### Resource Allocation Rules
- Bagged model resources must be ≤ total predictor resources
- Base model resources must be ≤ bagged model resources
- Resources are divided among parallel training processes