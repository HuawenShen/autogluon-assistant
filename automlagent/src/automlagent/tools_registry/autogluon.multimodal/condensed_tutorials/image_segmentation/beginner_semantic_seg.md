# Condensed: AutoMM for Semantic Segmentation - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information:

# AutoMM for Semantic Segmentation - Quick Start

## Overview
Semantic Segmentation creates pixel-wise segmentation maps of images. This tutorial demonstrates fine-tuning the Segment Anything Model (SAM) using AutoMM for domain-specific applications.

## Data Preparation

```python
# Install required package
!pip install autogluon.multimodal

# Download and extract dataset
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/leaf_disease_segmentation.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and prepare data
import pandas as pd
import os
dataset_path = os.path.join(download_dir, 'leaf_disease_segmentation')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = 'image'
label_col = 'label'

# Expand relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[per_col] = val_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[per_col] = test_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Zero-Shot Evaluation
Test pretrained SAM performance without fine-tuning:

```python
from autogluon.multimodal import MultiModalPredictor
predictor_zero_shot = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label=label_col,
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    num_classes=1
)

# Evaluate zero-shot performance
scores = predictor_zero_shot.evaluate(test_data, metrics=["iou"])
print(scores)
```

## Fine-tuning SAM
Fine-tune the model using LoRA for efficiency:

```python
import uuid
save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"
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

# Evaluate fine-tuned model
scores = predictor.evaluate(test_data, metrics=["iou"])
print(scores)
```

## Save and Load Model
```python
# Load saved model
loaded_predictor = MultiModalPredictor.load(save_path)
scores = loaded_predictor.evaluate(test_data, metrics=["iou"])
print(scores)
```

**Important Notes:**
- Use LoRA for efficient fine-tuning of large models
- `MultiModalPredictor.load()` uses pickle - only load trusted data
- The model is automatically saved after training
- For visualization, use `SemanticSegmentationVisualizer` class
- For customization options, refer to the Customize AutoMM documentation

This condensed version maintains all critical implementation details while removing redundant explanations and supplementary examples.