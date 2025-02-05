# Condensed: Training models with GPU support

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Training Models with GPU Support in AutoGluon

## Basic GPU Usage

GPU acceleration requires CUDA toolkit installation. Basic GPU usage:

```python
predictor = TabularPredictor(label=label).fit(
    train_data,
    num_gpus=1,  # Allocate 1 GPU for TabularPredictor
)
```

To enable GPU for specific models only:

```python
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # CPU training
        {'ag_args_fit': {'num_gpus': 1}}   # GPU training
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    num_gpus=1,
    hyperparameters=hyperparameters, 
)
```

## Multi-modal Models

For multi-modal models (tabular, text, image), configure GPU usage with:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

## LightGBM GPU Support

LightGBM requires special installation for GPU support:
1. Uninstall existing: `pip uninstall lightgbm -y`
2. Install GPU version: `pip install lightgbm --install-option=--gpu`

If this fails, follow the [official guide](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) for source installation.

## Advanced Resource Allocation

Resource allocation hierarchy:
- `num_cpus` and `num_gpus`: Total resources for TabularPredictor
- `ag_args_ensemble.ag_args_fit`: Resources for bagged models
- `ag_args_fit`: Resources for individual base models

Example of detailed resource allocation:

```python
predictor.fit(
    num_cpus=32,
    num_gpus=4,
    hyperparameters={
        'NN_TORCH': {},
    },
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
    }
    hyperparameter_tune_kwargs={
        'searcher': 'random',
        'scheduler': 'local',
        'num_trials': 2
    }
)
```

Key points for the above configuration:
- Total resources: 32 CPUs, 4 GPUs
- Per bagged model: 10 CPUs, 2 GPUs
- Per base model: 4 CPUs, 0.5 GPUs
- Runs 2 HPO trials in parallel
- Each trial runs 2 folds in parallel
- Total usage: 16 CPUs, 2 GPUs (4 models training in parallel)