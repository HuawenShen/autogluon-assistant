# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial covers SpeechBrain's checkpointing system implementation, focusing on model state persistence and recovery. It demonstrates how to implement checkpoint management for PyTorch models, optimizers, and training states using the Checkpointer class. Key functionalities include basic checkpoint saving/loading, metadata management, selective checkpoint retention, and parameter transfer between models. The tutorial helps with tasks like implementing training recovery, managing model versions, and handling early stopping based on validation metrics. Essential features covered include checkpoint structure definition, metadata handling, storage management with predicates, and custom checkpoint hooks for specialized objects.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# Checkpointing in SpeechBrain

## Key Concepts
- **Purpose**: Save model state, optimizer parameters, epoch info for:
  1. Recovery from interruptions
  2. Early stopping based on validation performance
  3. Model parameter persistence

## Core Implementation

### Basic Usage
```python
import speechbrain as sb
import torch
from speechbrain.utils.checkpoints import Checkpointer

# Initialize components
model = torch.nn.Linear(1, 1, False)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
epoch_counter = sb.utils.epoch_loop.EpochCounter(10)

# Setup checkpointer
checkpointer = Checkpointer(
    "checkpoint_dir",
    recoverables={
        "mdl": model,
        "opt": optimizer,
        "epochs": epoch_counter
    }
)

# Training loop with checkpointing
checkpointer.recover_if_possible()  # Load latest checkpoint if available
for epoch in epoch_counter:
    # Training code...
    checkpointer.save_and_keep_only(meta={"loss": loss.item()})
```

### Important Features

1. **Checkpoint Structure**
```python
# Add components individually
checkpointer.add_recoverable("mdl", model)
checkpointer.add_recoverables({"opt": optimizer, "epoch": epoch_counter})
```

2. **Meta Information**
```python
# Save with metadata
checkpointer.save_checkpoint(meta={
    "loss": 15.5,
    "validation-type": "fast",
    "num-examples": 3
})

# Recover best checkpoint
checkpointer.recover_if_possible(min_key="loss")
```

3. **Checkpoint Management**
```python
# Keep limited checkpoints
checkpointer.save_and_keep_only(
    meta={"loss": 13.1},
    num_to_keep=2,
    ckpt_predicate=lambda ckpt: ckpt.meta.get("validation-type") == "full"
)
```

4. **Parameter Transfer**
```python
# Find best checkpoint
ckpt_finder = Checkpointer(checkpoint_dir)
best_ckpt = ckpt_finder.find_checkpoint(
    min_key="loss",
    ckpt_predicate=lambda ckpt: ckpt.meta.get("validation-type") == "full"
)

# Transfer parameters
sb.utils.checkpoints.torch_parameter_transfer(new_model, best_ckpt.paramfiles["mdl"])
```

## Best Practices
1. Always implement recovery at start of training
2. Save metadata for tracking performance metrics
3. Use `save_and_keep_only()` to manage storage efficiently
4. Implement custom save/load hooks for custom objects using `@sb.utils.checkpoints.register_checkpoint_hooks`

## Important Notes
- Checkpointer automatically handles saving/loading for torch Modules and Optimizers
- Custom objects can register their own save/load hooks
- Use predicates and filters for selective checkpoint management
- Parameter transfer requires careful handling of model architectures