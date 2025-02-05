# Condensed: AutoMM for Image + Text + Tabular - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Image + Text + Tabular - Quick Start

AutoMM automatically builds deep learning models for multimodal datasets combining images, text, and tabular data. It requires only a multimodal dataframe as input to predict values based on features from multiple modalities.

## Setup

```python
!pip install autogluon.multimodal

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset Preparation

Using a simplified PetFinder dataset to predict animal adoption rates (binary classification: 0=slow, 1=fast).

```python
# Download and load dataset
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load CSV files
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'

# Process image paths
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

AutoMM automatically:
- Infers the problem type
- Detects data modalities
- Selects appropriate models
- Implements late-fusion when multiple backbones are available

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    time_limit=120, # seconds
)
```

## Evaluation and Prediction

```python
# Evaluate
scores = predictor.evaluate(test_data, metrics=["roc_auc"])

# Predict classes
predictions = predictor.predict(test_data.drop(columns=label_col))

# Get class probabilities (classification only)
probas = predictor.predict_proba(test_data.drop(columns=label_col))

# Extract embeddings
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
```

## Save and Load Models

```python
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-saved_model"
predictor.save(model_path)
loaded_predictor = MultiModalPredictor.load(model_path)
```

**Warning**: `MultiModalPredictor.load()` uses `pickle` module, which can be insecure. Only load trusted data.

## Additional Resources
- More examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- Customization: Refer to "Customize AutoMM" tutorial

This condensed version maintains all essential implementation details while removing redundant explanations and supplementary information not critical for basic usage.