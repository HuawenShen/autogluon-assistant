# Condensed: Image-to-Image Semantic Matching with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Image-to-Image Semantic Matching with AutoMM

## Overview
This tutorial demonstrates how to use AutoMM for computing similarity between image pairs using the Stanford Online Products (SOP) dataset. The model uses Swin Transformer to generate image embeddings and computes cosine similarity between them.

## Setup

```python
!pip install autogluon.multimodal

import os
import pandas as pd
import warnings
from IPython.display import Image, display
warnings.filterwarnings('ignore')
```

## Data Preparation

1. Download and extract the dataset:
```python
download_dir = './ag_automm_tutorial_img2img'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/Stanford_Online_Products.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

2. Load annotations:
```python
dataset_path = os.path.join(download_dir, 'Stanford_Online_Products')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col_1 = "Image1"
image_col_2 = "Image2"
label_col = "Label"
match_label = 1  # 1 indicates matching pairs
```

3. Expand image paths:
```python
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
    problem_type="image_similarity",
    query=image_col_1,          # first image column
    response=image_col_2,       # second image column
    label=label_col,           # label column
    match_label=match_label,   # label indicating matching pairs
    eval_metric='auc',         # evaluation metric
)

predictor.fit(
    train_data=train_data,
    time_limit=180,
)
```

## Model Usage

1. Evaluate model performance:
```python
score = predictor.evaluate(test_data)
print("evaluation score: ", score)
```

2. Predict matches:
```python
# Binary predictions (threshold = 0.5)
pred = predictor.predict(test_data.head(3))

# Probability predictions
proba = predictor.predict_proba(test_data.head(3))
```

3. Extract embeddings:
```python
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1][:5].tolist()})
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2][:5].tolist()})
```

## Key Points
- The model uses Swin Transformer for feature extraction
- Cosine similarity is used to compute image pair similarity
- Default prediction threshold is 0.5
- Dataset labels: 1 for matching pairs, 0 for non-matching pairs
- AUC is used as the evaluation metric

For customization options, refer to the AutoMM customization documentation.