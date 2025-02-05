# Condensed: Handling Class Imbalance with AutoMM - Focal Loss

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Handling Class Imbalance with AutoMM - Focal Loss

This tutorial demonstrates how to use focal loss in AutoMM for handling class imbalance problems.

## Key Concepts
- Focal loss helps balance training between:
  - Hard vs. easy samples
  - Uneven class distributions
- Main parameters:
  - `alpha`: Per-class loss weights
  - `gamma`: Controls focus on hard samples
  - `reduction`: Loss aggregation method ("mean" or "sum")

## Implementation

### 1. Setup and Data Preparation
```python
!pip install autogluon.multimodal

from autogluon.multimodal.utils.misc import shopee_dataset

# Load dataset
download_dir = "./ag_automm_tutorial_imgcls_focalloss"
train_data, test_data = shopee_dataset(download_dir)

# Create imbalanced dataset
import numpy as np
import pandas as pd

ds = 1
imbalanced_train_data = []
for lb in range(4):
    class_data = train_data[train_data.label == lb]
    sample_index = np.random.choice(np.arange(len(class_data)), size=int(len(class_data) * ds), replace=False)
    ds /= 3  # downsample 1/3 each time for each class
    imbalanced_train_data.append(class_data.iloc[sample_index])
imbalanced_train_data = pd.concat(imbalanced_train_data)

# Calculate class weights
weights = []
for lb in range(4):
    class_data = imbalanced_train_data[imbalanced_train_data.label == lb]
    weights.append(1 / (class_data.shape[0] / imbalanced_train_data.shape[0]))
weights = list(np.array(weights) / np.sum(weights))
```

### 2. Training with Focal Loss
```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee_focal"
predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.loss_function": "focal_loss",
        "optimization.focal_loss.alpha": weights,  # Class weights
        "optimization.focal_loss.gamma": 1.0,      # Hard sample focus
        "optimization.focal_loss.reduction": "sum",
        "optimization.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)
```

### 3. Training without Focal Loss (Comparison)
```python
model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee_non_focal"
predictor2 = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor2.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)
```

## Best Practices
1. Use focal loss when dealing with significant class imbalance
2. Set `alpha` weights inversely proportional to class frequencies
3. Adjust `gamma` to control focus on hard samples (typical range: 0.5-2.0)
4. Compare performance with and without focal loss for your specific use case

## Evaluation
Use `predictor.evaluate(test_data, metrics=["acc"])` to assess model performance. Models trained with focal loss typically show better performance on imbalanced datasets.