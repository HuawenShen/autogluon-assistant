# Condensed: AutoMM for Image Classification - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Image Classification - Quick Start

## Overview
This tutorial demonstrates using MultiModalPredictor for image classification. The predictor handles model training with a single `fit()` call once data is prepared in Pandas DataFrame format.

## Data Preparation
Using a subset of the Shopee-IET dataset containing clothing images with four categories: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = './ag_automm_tutorial_imgcls'
train_data_path, test_data_path = shopee_dataset(download_dir)
```

**Note**: MultiModalPredictor supports both image paths and bytearrays:
```python
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

## Model Training
```python
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data_path,
    time_limit=30, # seconds
)
```

Key Parameters:
- `label`: Column name containing target variable
- `path`: Directory for model and output storage
- `time_limit`: Training duration in seconds

## Model Evaluation
Evaluate on test data using either image paths or bytearrays:
```python
# Using image paths
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])

# Using bytearrays
scores = predictor.evaluate(test_data_byte, metrics=["accuracy"])
```

## Prediction
```python
# Single prediction
predictions = predictor.predict({'image': [image_path]})

# Probability predictions
proba = predictor.predict_proba({'image': [image_path]})

# Works with bytearrays too
image_byte = test_data_byte.iloc[0]['image']
predictions = predictor.predict({'image': [image_byte]})
```

## Feature Extraction
Extract image embeddings:
```python
feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)
```

## Model Persistence
```python
# Save happens automatically after fit()
# Load saved model
loaded_predictor = MultiModalPredictor.load(model_path)
```

⚠️ **Warning**: Only load models from trusted sources due to pickle security concerns.

## Key Features
- Supports both image paths and bytearrays
- Automatic model training with minimal configuration
- Built-in evaluation metrics
- Feature extraction capabilities
- Easy model persistence

For advanced customization, refer to the Customize AutoMM documentation.