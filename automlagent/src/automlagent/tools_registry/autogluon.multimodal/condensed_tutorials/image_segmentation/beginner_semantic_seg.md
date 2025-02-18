# Condensed: AutoMM for Semantic Segmentation - Quick Start

Summary: This tutorial demonstrates implementing semantic segmentation using AutoGluon's MultiModalPredictor with SAM (Segment Anything Model). It covers both zero-shot inference and fine-tuning workflows using the facebook/sam-vit-base model, with LoRA for efficient training. Key implementations include dataset preparation with image-mask pairs, model configuration for single-class segmentation, evaluation using IoU metrics, and visualization utilities. The tutorial helps with tasks like setting up semantic segmentation pipelines, performing zero-shot predictions, fine-tuning SAM models, and evaluating segmentation results. Notable features include path handling for datasets, model saving/loading functionality, and visualization tools for segmentation outputs.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Semantic Segmentation - Quick Start

## Key Implementation Details

### Setup and Data Preparation
```python
!pip install autogluon.multimodal

# Load and prepare dataset
import pandas as pd
import os

# Expand relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])
```

### Zero-Shot Evaluation
```python
from autogluon.multimodal import MultiModalPredictor

predictor_zero_shot = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label=label_col,
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    num_classes=1  # foreground-background segmentation
)

# Predict and evaluate
pred_zero_shot = predictor_zero_shot.predict({'image': [test_data.iloc[0]['image']]})
scores = predictor_zero_shot.evaluate(test_data, metrics=["iou"])
```

### Fine-tuning SAM
```python
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label="label",
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    path=save_path,
)

# Train the model
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180  # seconds
)
```

## Critical Configurations
- Uses LoRA for efficient fine-tuning
- Default model: SAM (Segment Anything Model)
- Problem type: "semantic_segmentation"
- Base model: "facebook/sam-vit-base"
- Single class segmentation (num_classes=1)

## Important Notes and Best Practices
1. Dataset format requires two columns:
   - Image paths
   - Corresponding groundtruth masks

2. Model saving and loading:
```python
# Load saved model
loaded_predictor = MultiModalPredictor.load(save_path)
```

⚠️ **Warning**: `MultiModalPredictor.load()` uses `pickle` module, which can be insecure. Only load trusted data.

3. Visualization utilities available:
```python
from autogluon.multimodal.utils import SemanticSegmentationVisualizer
visualizer = SemanticSegmentationVisualizer()
```

4. Evaluation metrics:
```python
scores = predictor.evaluate(test_data, metrics=["iou"])
```

## Performance Notes
- Zero-shot SAM performs basic segmentation but may lack domain-specific accuracy
- Fine-tuning significantly improves performance for specialized tasks
- LoRA enables efficient fine-tuning of the large SAM model

For customization options, refer to the "Customize AutoMM" documentation.