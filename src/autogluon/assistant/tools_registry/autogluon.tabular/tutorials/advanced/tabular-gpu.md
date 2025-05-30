Summary: This tutorial demonstrates GPU integration in AutoGluon's TabularPredictor, covering implementation techniques for multi-level resource allocation (predictor, bagged model, and base model levels). It helps with tasks involving GPU-accelerated model training, particularly for LightGBM and neural networks. Key features include configuring single/multiple GPU usage, model-specific GPU allocation, proper CUDA toolkit setup, and hierarchical resource management with specific allocation rules. The tutorial provides practical code examples for both basic and advanced GPU configurations, making it valuable for optimizing machine learning workflows with GPU acceleration.

# Training models with GPU support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/advanced/tabular-gpu.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/advanced/tabular-gpu.ipynb)



Training with GPU can significantly speed up base algorithms, and is a necessity for text and vision models where training without GPU is infeasibly slow. 
CUDA toolkit is required for GPU training. Please refer to the [official documentation](https://docs.nvidia.com/cuda/) for the installation instructions.

```python
predictor = TabularPredictor(label=label).fit(
    train_data,
    num_gpus=1,  # Grant 1 gpu for the entire Tabular Predictor
)
```


To enable GPU acceleration on only specific models, the same parameter can be passed into model `hyperparameters`:

```python
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # Train with CPU
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU. This amount needs to be <= total num_gpus granted to TabularPredictor
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    num_gpus=1,
    hyperparameters=hyperparameters, 
)
```


## Multi-modal

In [Multimodal Data Tables: Tabular, Text, and Image](../tabular-multimodal.ipynb) tutorial we presented how to train an ensemble which can utilize tabular, text and images. 
If available GPUs don't have enough VRAM to fit the default model, or it is needed to speedup testing, different backends can be used:

Regular configuration is retrieved like this:


```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
hyperparameters
```

## Enabling GPU for LightGBM

The default installation of LightGBM does not support GPU training, however GPU support can be enabled via a special install. If `num_gpus` is set, the following warning will be displayed:

```
Warning: GPU mode might not be installed for LightGBM, GPU training raised an exception. Falling back to CPU training...Refer to LightGBM GPU documentation: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-versionOne possible method is:	pip uninstall lightgbm -y	pip install lightgbm --install-option=--gpu
```


If the suggested commands do not work, uninstall existing lightgbm `pip uninstall -y lightgbm` and install from sources following the instructions in the [official guide](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html). The
optional [Install Python Interface](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#install-python-interface-optional) section is also required to make it work with AutoGluon.

## Advanced Resource Allocation

Most of the time, you would only need to set `num_cpus` and `num_gpus` at the predictor `fit` level to control the total resources you granted to the TabularPredictor.
However, if you want to have more detailed control, we offer the following options.

`ag_args_ensemble: ag_args_fit: { RESOURCES }` allows you to control the total resources granted to a bagged model.
If using parallel folding strategy, individual base model's resources will be calculated respectively.
This value needs to be <= total resources granted to TabularPredictor
This parameter will be ignored if bagging model is not enabled.

`ag_args_fit: { RESOURCES }` allows you to control the total resources granted to a single base model.
This value needs to be <= total resources granted to TabularPredictor and <= total resources granted to a bagged model if applicable.

As an example, consider the following scenario

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


We train 2 HPO trials, which trains 2 folds in parallel at the same time. The total resources granted to the TabularPredictor is 32 cpus and 4 gpus.

For a bagged model, we grant 10 cpus and 2 gpus.
This means we would run two HPO trials in parallel, each granted 10 cpus and 2 gpus -> 20 cpus and 4 gpus in total.

We also specified that for an individual model base we want 4 cpus and 0.5 gpus and we can train two folds in parallel according to the bagged level resources -> 8 cpus and 1 gpus for a bagged model -> 16 cpus and 2 gpus when two trials running in parallel.

Therefore, we will use 16 cpus and 2 gpus in total and have two trials of bagged model running in parallel each running two folds in parallel -> 4 models training in parallel.
