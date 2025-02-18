# Condensed: Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning

Summary: This tutorial demonstrates parameter-efficient finetuning techniques for large language models using AutoGluon's MultiModalPredictor. It specifically covers implementation of IA3-bias finetuning combined with gradient checkpointing to enable training billion-parameter models (like FLAN-T5-XL) on limited hardware resources. The tutorial helps with tasks requiring efficient model adaptation while tuning only ~0.5% of parameters. Key features include memory optimization configurations, batch size adjustments, and cache management practices. It's particularly valuable for developers working on cross-lingual tasks or needing to finetune large models on single GPUs while maintaining comparable performance to full finetuning.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Parameter-Efficient Finetuning for Billion-scale Models

## Key Concepts
- Uses parameter-efficient finetuning to handle large foundation models
- Combines gradient checkpointing with efficient finetuning for single GPU training
- Enables finetuning of billion-parameter models on limited hardware

## Implementation Details

### Basic Parameter-Efficient Finetuning

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(train_df,
              presets="multilingual",
              hyperparameters={
                  "optimization.efficient_finetune": "ia3_bias",  # Enable efficient finetuning
                  "optimization.lr_decay": 0.9,
                  "optimization.learning_rate": 3e-03,
                  "optimization.end_lr": 3e-03,
                  "optimization.max_epochs": 2,
                  "env.batch_size": 32,
              })
```

**Key Points:**
- Uses `ia3_bias` algorithm for parameter-efficient finetuning
- Tunes only ~0.5% of model parameters
- Maintains comparable performance to full finetuning

### Large Model Training (FLAN-T5-XL)

```python
predictor.fit(train_df,
              presets="multilingual", 
              hyperparameters={
                  "model.hf_text.checkpoint_name": "google/flan-t5-xl",
                  "model.hf_text.gradient_checkpointing": True,  # Enable gradient checkpointing
                  "model.hf_text.low_cpu_mem_usage": True,
                  "optimization.efficient_finetune": "ia3_bias",
                  "optimization.learning_rate": 3e-03,
                  "optimization.max_epochs": 1,
                  "env.batch_size": 1,
                  "env.eval_batch_size_ratio": 1
              })
```

**Critical Configurations:**
- Enable gradient checkpointing with `gradient_checkpointing: True`
- Use small batch size (1) for large models
- Set `low_cpu_mem_usage: True` for memory efficiency
- Combine with `ia3_bias` for parameter-efficient training

## Best Practices
1. Clear cache between training runs:
```python
import os, shutil
if os.path.exists("cache"):
    shutil.rmtree("cache")
```

2. Use smaller training samples for initial testing
3. Monitor memory usage and adjust batch size accordingly
4. Combine gradient checkpointing with parameter-efficient finetuning for large models

## Performance Notes
- Achieves comparable results to full finetuning
- Works well for cross-lingual tasks
- Enables billion-parameter model training on single GPU (e.g., NVIDIA T4)