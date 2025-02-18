# Condensed: Customize AutoMM

Summary: This tutorial provides comprehensive implementation guidance for AutoMM (Auto Multi-Modal) model configurations, covering optimization techniques, model architectures, and data processing. It details how to configure learning rates, optimizers, and training controls; implement efficient fine-tuning strategies like LoRA and IA3; set up GPU and distributed training environments; customize model architectures for text (HuggingFace), image (TIMM, MMDetection), and multi-modal tasks (CLIP); handle data preprocessing including missing values, transformations, and augmentations (Mixup/Cutmix); and implement knowledge distillation. Key functionalities include mixed-precision training, gradient management, model checkpointing, and various data preprocessing strategies for numerical, categorical, and text data.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and configurations:

# AutoMM Customization Guide

## Key Optimization Configurations

### Learning Rate Related
```python
# Basic learning rate
predictor.fit(hyperparameters={
    "optimization.learning_rate": 1.0e-4,  # Default
    "optimization.lr_schedule": "cosine_decay",  # Options: polynomial_decay, linear_decay
    "optimization.lr_decay": 0.9,  # For layerwise decay
    "optimization.lr_mult": 1,  # For two-stage learning rates
    "optimization.lr_choice": "layerwise_decay"  # Options: two_stages, ""
})

# Learning rate warmup
predictor.fit(hyperparameters={
    "optimization.warmup_steps": 0.1  # Percentage of steps for warmup
})
```

### Optimizer Settings
```python
predictor.fit(hyperparameters={
    "optimization.optim_type": "adamw",  # Options: adam, sgd
    "optimization.weight_decay": 1.0e-3
})
```

### Training Control
```python
predictor.fit(hyperparameters={
    "optimization.max_epochs": 10,
    "optimization.max_steps": -1,  # -1 to disable
    "optimization.patience": 10,
    "optimization.val_check_interval": 0.5  # Float [0.0, 1.0] or int
})
```

### Gradient Management
```python
predictor.fit(hyperparameters={
    "optimization.gradient_clip_algorithm": "norm",  # Options: value
    "optimization.gradient_clip_val": 1,
    "optimization.track_grad_norm": -1  # -1 for no tracking, 2 for 2-norm
})
```

### Model Checkpointing
```python
predictor.fit(hyperparameters={
    "optimization.top_k": 3,  # Number of checkpoints to consider
    "optimization.top_k_average_method": "greedy_soup"  # Options: uniform_soup, best
})
```

## Important Notes:
- Use `lr_decay` for layer-wise learning rates (earlier layers have smaller rates)
- `max_steps` and `max_epochs` work together - training stops when either is reached
- Validation checks frequency can be controlled by float (fraction of epoch) or int (fixed batches)
- Gradient clipping supports both value-based and norm-based approaches
- Model averaging supports three strategies: greedy_soup, uniform_soup, and best

This condensed version maintains all critical implementation details while removing explanatory text and redundant examples.

Here's the condensed tutorial content focusing on key implementation details and practices:

# AutoMM Hyperparameter Configuration Guide - Part 2

## Optimization Parameters

### optimization.top_k_average_method
```python
# Average top k checkpoints uniformly
predictor.fit(hyperparameters={"optimization.top_k_average_method": "uniform_soup"})
```

### optimization.efficient_finetune
Parameter-efficient finetuning options:
- `bit_fit`: Bias parameters only
- `norm_fit`: Normalization + bias parameters  
- `lora`: LoRA Adaptors
- `lora_bias`: LoRA + bias parameters
- `lora_norm`: LoRA + normalization + bias
- `ia3`: IA3 algorithm
- `ia3_bias`: IA3 + bias
- `ia3_norm`: IA3 + normalization + bias

```python
# Example configurations
predictor.fit(hyperparameters={"optimization.efficient_finetune": "bit_fit"})
predictor.fit(hyperparameters={"optimization.efficient_finetune": "ia3_bias"})
```

## Environment Configuration

### GPU and Batch Size Settings
```python
# GPU configuration
predictor.fit(hyperparameters={
    "env.num_gpus": -1,  # Use all available GPUs
    "env.per_gpu_batch_size": 8,  # Batch size per GPU
    "env.batch_size": 128,  # Total batch size
    "env.eval_batch_size_ratio": 4  # Multiplier for eval batch size
})
```

### Precision and Workers
```python
# Training precision options
predictor.fit(hyperparameters={
    "env.precision": "16-mixed",  # Mixed precision training
    "env.num_workers": 2,  # Training dataloader workers
    "env.num_workers_evaluation": 2  # Evaluation dataloader workers
})
```

### Distributed Training
```python
# Strategy options: "dp", "ddp", "ddp_spawn"
predictor.fit(hyperparameters={"env.strategy": "ddp_spawn"})
```

### Model Compilation
```python
# Torch compile settings
predictor.fit(hyperparameters={
    "env.compile.turn_on": True,
    "env.compile.mode": "default",
    "env.compile.dynamic": True,
    "env.compile.backend": "inductor"
})
```

## Important Notes:
- Mixed precision (`16-mixed`) can provide 3x speedup on modern GPUs
- Worker count doesn't always improve performance with `ddp_spawn` strategy
- Model compilation is recommended for large models and long training sessions
- Batch size accumulation occurs if `env.batch_size` > `env.per_gpu_batch_size * env.num_gpus`

This condensed version maintains all critical implementation details while removing redundant examples and explanations.

Here's the condensed tutorial content focusing on key implementation details and configurations:

# AutoMM Model Configuration Guide

## Core Model Selection
```python
# Select specific model types
predictor.fit(hyperparameters={
    "model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"] # default
    # OR
    "model.names": ["hf_text"] # text only
    "model.names": ["timm_image"] # image only 
    "model.names": ["clip"] # CLIP only
})
```

## Text Model Configurations (HuggingFace)

### Key Parameters
```python
# Text backbone selection
predictor.fit(hyperparameters={
    "model.hf_text.checkpoint_name": "google/electra-base-discriminator" # default
})

# Pooling configuration
predictor.fit(hyperparameters={
    "model.hf_text.pooling_mode": "cls" # default, or "mean"
})

# Tokenizer selection
predictor.fit(hyperparameters={
    "model.hf_text.tokenizer_name": "hf_auto" # default
    # Options: "bert", "electra", "clip"
})
```

### Text Processing Settings
```python
# Text length and segmentation
predictor.fit(hyperparameters={
    "model.hf_text.max_text_len": 512, # default, use -1 for model max
    "model.hf_text.insert_sep": True, # default, insert SEP tokens between columns
    "model.hf_text.text_segment_num": 2, # default number of text segments
})

# Text chunking and augmentation
predictor.fit(hyperparameters={
    "model.hf_text.stochastic_chunk": False, # default
    "model.hf_text.text_aug_detect_length": 10, # min length for augmentation
    "model.hf_text.text_trivial_aug_maxscale": 0, # default, set 0.1 to enable
})

# Memory optimization
predictor.fit(hyperparameters={
    "model.hf_text.gradient_checkpointing": False # default
})
```

## FT-Transformer Configurations

### Architecture Settings
```python
predictor.fit(hyperparameters={
    # Model structure
    "model.ft_transformer.num_blocks": 3, # default
    "model.ft_transformer.token_dim": 192, # default
    "model.ft_transformer.hidden_size": 192, # default
    "model.ft_transformer.ffn_hidden_size": 192, # default
    
    # Pre-trained initialization
    "model.ft_transformer.checkpoint_name": None # default
    # Can use local path or URL for checkpoint
})
```

Important Notes:
- Use gradient checkpointing for memory optimization with large models
- Text augmentation only occurs when text length ≥ text_aug_detect_length
- FT-Transformer's FFN hidden size defaults to model hidden size (unlike standard transformer's 4x)
- Careful consideration needed when modifying text segmentation and tokenization parameters

Here's the condensed tutorial focusing on key implementation details and configurations:

# Image Model Configurations

### Checkpoint Selection
```python
# Swin Transformer (default)
predictor.fit(hyperparameters={
    "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224"
})

# ViT Base
predictor.fit(hyperparameters={
    "model.timm_image.checkpoint_name": "vit_base_patch32_224"
})
```

### Image Transforms
```python
# Training transforms
predictor.fit(hyperparameters={
    "model.timm_image.train_transforms": [
        "resize_shorter_side", 
        "center_crop", 
        "trivial_augment"  # default
    ]
})

# Custom transforms using torchvision
predictor.fit(hyperparameters={
    "model.timm_image.train_transforms": [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip()
    ]
})

# Validation transforms
predictor.fit(hyperparameters={
    "model.timm_image.val_transforms": [
        "resize_shorter_side",
        "center_crop"  # default
    ]
})
```

# Object Detection Configurations

### MMDetection Models
```python
# Default YOLOv3
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_8xb24-320-300e_coco"
})

# YOLOX-L
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.checkpoint_name": "yolox_l"
})
```

### Bounding Box Format
```python
predictor = MultiModalPredictor(hyperparameters={
    "model.mmdet_image.output_bbox_format": "xyxy"  # default
    # or "xywh" for width/height format
})
```

# SAM (Segment Anything Model) Configurations

```python
# Model Selection
predictor.fit(hyperparameters={
    "model.sam.checkpoint_name": "facebook/sam-vit-huge"  # default
})

# Training Configuration
predictor.fit(hyperparameters={
    "model.sam.train_transforms": ["random_horizontal_flip"],
    "model.sam.img_transforms": ["resize_to_square"],
    "model.sam.gt_transforms": ["resize_gt_to_square"],
    "model.sam.num_mask_tokens": 1,
    "model.sam.ignore_label": 255
})
```

# Data Processing Configurations

### Missing Data Handling
```python
predictor.fit(hyperparameters={
    "data.image.missing_value_strategy": "zero"  # default
    # or "skip" to ignore missing images
})
```

### Data Type Conversions
```python
predictor.fit(hyperparameters={
    "data.categorical.convert_to_text": False,  # default
    "data.numerical.convert_to_text": False,    # default
    "data.text.normalize_text": False           # default
})
```


...(truncated)