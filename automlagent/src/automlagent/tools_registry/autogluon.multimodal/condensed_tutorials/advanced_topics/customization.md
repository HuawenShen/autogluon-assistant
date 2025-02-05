# Condensed: Customize AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# AutoMM Configuration Guide

This guide covers key configuration options for AutoMM's optimization, environment, model, and data settings.

## Optimization Settings

### Learning Rate and Optimizer
```python
# Base learning rate
predictor.fit(hyperparameters={"optimization.learning_rate": 1.0e-4})

# Optimizer type (adam, adamw, sgd)
predictor.fit(hyperparameters={"optimization.optim_type": "adamw"})

# Weight decay regularization
predictor.fit(hyperparameters={"optimization.weight_decay": 1.0e-3})
```

### Learning Rate Strategies
```python
# Layer-wise decay (later layers can have larger learning rates)
predictor.fit(hyperparameters={"optimization.lr_decay": 0.9})

# Two-stage learning rate (head layer multiplier)
predictor.fit(hyperparameters={"optimization.lr_mult": 10})

# Learning rate schedule (cosine_decay, polynomial_decay, linear_decay)
predictor.fit(hyperparameters={"optimization.lr_schedule": "cosine_decay"})
```

### Training Control
```python
# Maximum epochs
predictor.fit(hyperparameters={"optimization.max_epochs": 10})

# Early stopping patience
predictor.fit(hyperparameters={"optimization.patience": 10})

# Validation check frequency
predictor.fit(hyperparameters={"optimization.val_check_interval": 0.5})

# Gradient clipping
predictor.fit(hyperparameters={
    "optimization.gradient_clip_algorithm": "norm",
    "optimization.gradient_clip_val": 1
})
```

## Environment Settings

### Hardware Configuration
```python
# GPU usage
predictor.fit(hyperparameters={"env.num_gpus": -1})  # -1 for all available

# Batch size settings
predictor.fit(hyperparameters={
    "env.per_gpu_batch_size": 8,
    "env.batch_size": 128
})

# Precision settings (16-mixed, bf16-mixed, 32, 64)
predictor.fit(hyperparameters={"env.precision": "16-mixed"})
```

### Training Distribution
```python
# Distribution strategy (dp, ddp, ddp_spawn)
predictor.fit(hyperparameters={"env.strategy": "ddp_spawn"})

# Worker configuration
predictor.fit(hyperparameters={
    "env.num_workers": 2,
    "env.num_workers_evaluation": 2
})
```

## Model Settings

### Model Selection
```python
# Choose model types
predictor.fit(hyperparameters={
    "model.names": ["hf_text", "timm_image", "clip", "categorical_mlp"]
})

# Text model configuration
predictor.fit(hyperparameters={
    "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
    "model.hf_text.pooling_mode": "cls",
    "model.hf_text.max_text_len": 512
})

# Image model configuration
predictor.fit(hyperparameters={
    "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
    "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"]
})
```

## Data Processing

### Data Handling
```python
# Missing data strategy
predictor.fit(hyperparameters={
    "data.image.missing_value_strategy": "zero"
})

# Numerical data processing
predictor.fit(hyperparameters={
    "data.numerical.scaler_with_mean": True,
    "data.numerical.scaler_with_std": True
})

# Label processing
predictor.fit(hyperparameters={
    "data.label.numerical_label_preprocessing": "standardscaler"
})
```

### Data Augmentation
```python
# Mixup configuration
predictor.fit(hyperparameters={
    "data.mixup.turn_on": True,
    "data.mixup.mixup_alpha": 0.8,
    "data.mixup.prob": 1.0
})
```

## Critical Notes:
- Always test configuration changes on a small dataset first
- Monitor memory usage when adjusting batch sizes
- Consider hardware limitations when setting precision and worker counts
- Validate model compatibility when changing architectures
- Test data augmentation strategies for your specific use case

This condensed version maintains all essential implementation details while removing redundant explanations and supplementary information not critical for implementation.