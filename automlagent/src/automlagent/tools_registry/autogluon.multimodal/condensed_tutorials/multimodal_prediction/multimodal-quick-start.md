# Condensed: AutoGluon Multimodal - Quick Start

Summary: This tutorial demonstrates the implementation of AutoGluon's MultiModalPredictor for handling combined image, text, and tabular data tasks. It provides code examples for data preparation (including image path handling), model training with automatic feature detection and model selection, and prediction/evaluation workflows. Key functionalities covered include automatic problem type inference, late-fusion model addition, and support for both classification and regression tasks. The tutorial is particularly useful for tasks involving multimodal data processing, model training automation, and performance evaluation, with extensibility for embedding extraction, model distillation, fine-tuning, and semantic matching applications.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoGluon MultiModalPredictor Quick Start

## Key Setup
```python
# Install and import
!python -m pip install autogluon

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

## Implementation Details

### 1. Data Preparation
- Supports multiple input types: images, text, and tabular data
- Data must be in AutoGluon's multimodal dataframe format
- Image columns must contain paths to single image files

```python
# Load data
train_data = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)

# Image path handling
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) 
                    for path in path_l])

# Process image paths
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(
    lambda ele: path_expander(ele.split(';')[0], base_folder=dataset_path)
)
```

### 2. Model Training
```python
# Initialize and train
predictor = MultiModalPredictor(label='AdoptionSpeed').fit(
    train_data=train_data,
    time_limit=120  # in seconds
)
```

**Key Features:**
- Automatic problem type inference (classification/regression)
- Automatic feature modality detection
- Automatic model selection from multimodal pools
- Late-fusion model addition for multiple backbones

### 3. Prediction and Evaluation
```python
# Make predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Get probability scores (classification only)
probs = predictor.predict_proba(test_data.drop(columns=label_col))

# Evaluate performance
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

## Best Practices
1. Ensure image paths are properly formatted and accessible
2. Increase `time_limit` for better model performance
3. Use appropriate evaluation metrics for your problem type

## Important Notes
- Image columns must contain single file paths
- Multiple modalities (image, text, tabular) can be combined
- Automatic handling of model selection and training
- Supports both classification and regression tasks

This implementation can be extended for:
- Embedding extraction
- Model distillation
- Fine-tuning
- Text/image prediction
- Semantic matching