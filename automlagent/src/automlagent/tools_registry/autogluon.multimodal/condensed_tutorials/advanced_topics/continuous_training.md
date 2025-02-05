# Condensed: Continuous Training with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

# Continuous Training with AutoMM

This tutorial demonstrates three key use cases for continuous training with AutoMM, allowing models to build upon previously acquired knowledge.

## Use Case 1: Expanding Training with Additional Data or Training Time

AutoMM enables:
- Extending training time for underfitting models
- Incorporating new data with the same problem type and classes
- Continuing training from existing checkpoints

### Implementation Example
```python
# Load and prepare data
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal import MultiModalPredictor
import uuid

train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
train_data_1 = train_data.sample(n=1000, random_state=0)  # Subsample for demo

# Initial training
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)

# Continue training with new data
predictor_2 = MultiModalPredictor.load(model_path)
train_data_2 = train_data.drop(train_data_1.index).sample(n=1000, random_state=0)
predictor_2.fit(train_data_2, time_limit=60)
```

## Use Case 2: Resuming Training from Last Checkpoint

For interrupted training, resume from the last checkpoint using:

```python
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

## Use Case 3: Applying Pre-Trained Models to New Tasks

Transfer learning implementation for related but different tasks:

```python
# Dump existing model
dump_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor.dump_model(save_path=dump_model_path)

# Load for new task with custom model configuration
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label="score", path=sts_model_path)
predictor_sts.fit(
    sts_train_data, 
    hyperparameters={"model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"}, 
    time_limit=30
)
```

### Supported Model Types
- Text: HuggingFace models
- Image: timm and MMDetection models
- Fusion models combining the above

### Configuration Options
```python
# For timm image models
{"model.timm_image.checkpoint_name": timm_image_model_path}

# For MMDetection models
{"model.mmdet_image.checkpoint_name": mmdet_image_model_path}
```

**Important Note**: When applying models to new tasks, be aware of potential catastrophic forgetting issues where models may lose previously learned information.