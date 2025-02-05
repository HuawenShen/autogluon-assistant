# Condensed: AutoGluon Multimodal - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoGluon Multimodal - Quick Start

AutoGluon's `MultiModalPredictor` automatically builds deep learning models for images, text, and tabular data inputs. It predicts one column's values based on other features in a multimodal dataframe format.

## Setup

```python
!python -m pip install --upgrade pip
!python -m pip install autogluon

import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Data Preparation

Using a simplified PetFinder dataset to predict pet adoption rates (binary classification: 0=slow, 1=fast).

```python
from autogluon.core.utils.loaders import load_zip

# Download and extract dataset
download_dir = './ag_multimodal_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
dataset_path = f'{download_dir}/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

### Image Path Processing
Important: Image columns must contain single image file paths.

```python
image_col = 'Images'

# Take only first image if multiple exist
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Convert to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120  # Adjust time_limit for better performance
)
```

Key Features:
- Automatic problem type detection
- Feature modality detection
- Model selection from multimodal pools
- Late-fusion model addition for multiple backbones

## Prediction

```python
# Class predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Probability predictions
probs = predictor.predict_proba(test_data.drop(columns=label_col))
```

## Evaluation

```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

## Important Notes
- Longer training times generally improve performance
- The predictor automatically handles:
  - Problem type inference
  - Feature modality detection
  - Model selection
  - Multi-modal fusion

Advanced features available in other tutorials:
- Embedding extraction
- Model distillation
- Fine-tuning
- Text/image prediction
- Semantic matching