# Condensed: AutoMM Detection - Finetune on COCO Format Dataset with Customized Settings

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information:

# AutoMM Detection - Finetune on COCO Format Dataset

## Overview
This tutorial demonstrates how to finetune a pretrained model on the Pothole dataset (single object detection) using COCO format. The dataset contains 665 images with bounding box annotations for pothole detection.

## Setup and Dependencies

```python
!pip install autogluon.multimodal
!mim install "mmcv==2.1.0"
!pip install "mmdet==3.2.0"

from autogluon.multimodal import MultiModalPredictor
import os
from autogluon.core.utils.loaders import load_zip
```

## Data Preparation

```python
# Download and extract dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
download_dir = "./pothole"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Set paths for data splits
data_dir = os.path.join(download_dir, "pothole")
train_path = os.path.join(data_dir, "Annotations", "usersplit_train_cocoformat.json")
val_path = os.path.join(data_dir, "Annotations", "usersplit_val_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "usersplit_test_cocoformat.json")
```

## Model Configuration and Training

```python
# Model configuration
checkpoint_name = "yolox_s"  # YOLOX-small model for fast training
num_gpus = 1

# Initialize predictor
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_path,
)

# Training configuration
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optimization.learning_rate": 1e-4,  # Head layers use 100x this rate
        "env.per_gpu_batch_size": 16,
        "optimization.max_epochs": 30,
        "optimization.val_check_interval": 1.0,
        "optimization.check_val_every_n_epoch": 3,
        "optimization.patience": 3,
    },
)
```

## Evaluation and Prediction

```python
# Evaluate model
predictor.evaluate(test_path)

# Generate predictions
pred = predictor.predict(test_path)
```

## Visualization

```python
!pip install opencv-python

from autogluon.multimodal.utils import visualize_detection

# Visualize predictions
conf_threshold = 0.25
visualization_result_dir = "./"
visualized = visualize_detection(
    pred=pred[12:13],
    detection_classes=predictor.classes,
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)

from PIL import Image
from IPython.display import display
img = Image.fromarray(visualized[0][:, :, ::-1], 'RGB')
display(img)
```

## Quick Alternative Using Presets
For faster implementation, use predefined presets:

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",
)
predictor.fit(train_path, tuning_data=val_path)
predictor.evaluate(test_path)
```

## Key Implementation Notes
- Uses two-stage learning rate with 100x rate for head layers
- Default batch size of 16 (adjust based on GPU memory)
- Early stopping after 3 consecutive non-improving validations
- Validation checks every 3 epochs
- YOLOX-small model selected for speed (larger models available for better performance)
- Confidence threshold of 0.25 for visualization filtering

For higher performance, consider using larger models like VFNet with longer training times.