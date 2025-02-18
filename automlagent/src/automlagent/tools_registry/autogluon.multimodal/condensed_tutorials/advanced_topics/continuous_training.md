# Condensed: Continuous Training with AutoMM

Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor implementation for continuous machine learning workflows, specifically covering three key techniques: extending training with new data, resuming interrupted training sessions, and transfer learning across tasks. It helps with coding tasks related to model persistence, checkpoint management, and transfer learning configuration for text, image, and fusion models. The tutorial details essential functionalities including loading/saving models, checkpoint handling, and hyperparameter configuration for transfer learning, while emphasizing best practices for production deployment and data consistency. Implementation specifics include working with model checkpoints, configuring paths for different model types (HuggingFace, TIMM, MMDetection), and managing training continuity across sessions.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and practices:

# Continuous Training with AutoMM

## Key Use Cases

### 1. Expanding Training with Additional Data/Time

```python
# Initial Training
from autogluon.multimodal import MultiModalPredictor
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)

# Continue Training with New Data
predictor_2 = MultiModalPredictor.load(model_path)
predictor_2.fit(train_data_2, time_limit=60)
```

**Important Notes:**
- Use same problem type and classes for additional data
- Longer time_limit recommended for production (e.g., 1 hour or None)
- Model checkpoint saved as `model.ckpt` under `model_path`

### 2. Resuming Interrupted Training

```python
# Resume from last checkpoint
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

**Key Point:** Uses `last.ckpt` instead of `model.ckpt`

### 3. Transfer Learning to New Tasks

```python
# Dump existing model
dump_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor.dump_model(save_path=dump_model_path)

# Use as foundation for new task
predictor_new = MultiModalPredictor(label="new_label", path="new_path")
predictor_new.fit(
    new_data, 
    hyperparameters={
        "model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"
    }, 
    time_limit=30
)
```

**Supported Model Types:**
- HuggingFace text models: `model.hf_text.checkpoint_name`
- TIMM image models: `model.timm_image.checkpoint_name`
- MMDetection models: `model.mmdet_image.checkpoint_name`
- Fusion models combining the above

**Warning:** Be aware of potential catastrophic forgetting when transferring to new tasks.

## Best Practices
1. Use longer training times for production (>1 hour)
2. Ensure consistent data format between original and new training data
3. Consider task similarity when using transfer learning
4. Monitor performance metrics when continuing training to avoid degradation