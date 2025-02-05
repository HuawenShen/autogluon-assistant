# Condensed: Image-Text Semantic Matching with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Image-Text Semantic Matching with AutoMM

## Overview
This tutorial demonstrates how to use AutoMM for image-text semantic matching, which measures visual-semantic similarity between images and text using the Flickr30K dataset.

## Setup

```python
!pip install autogluon.multimodal

import os
import warnings
from IPython.display import Image, display
import numpy as np
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset Preparation

```python
# Download and load dataset
from autogluon.core.utils.loaders import load_pd, load_zip
download_dir = './ag_automm_tutorial_imgtxt'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load CSV files
dataset_path = os.path.join(download_dir, 'flickr30k_processed')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = "image"
text_col = "caption"

# Convert relative paths to absolute
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for data in [train_data, val_data, test_data]:
    data[image_col] = data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

# Prepare test data for evaluation
test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
test_data_with_label = test_data.copy()
test_label_col = "relevance"
test_data_with_label[test_label_col] = [1] * len(test_data)
```

## Model Setup and Training

```python
from autogluon.multimodal import MultiModalPredictor

# Initialize predictor
predictor = MultiModalPredictor(
    query=text_col,
    response=image_col,
    problem_type="image_text_similarity",
    eval_metric="recall",
)

# Zero-shot evaluation
txt_to_img_scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_text_data,
    response_data=test_image_data,
    label=test_label_col,
    cutoffs=[1, 5, 10],
)

# Finetune model
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180,
)
```

## Core Functionality

### Prediction

```python
# Binary matching prediction
pred = predictor.predict(test_data.head(5))

# Matching probabilities
proba = predictor.predict_proba(test_data.head(5))
```

### Embedding Extraction

```python
# Extract embeddings
image_embeddings = predictor.extract_embedding({image_col: test_image_data[image_col][:5].tolist()})
text_embeddings = predictor.extract_embedding({text_col: test_text_data[text_col][:5].tolist()})
```

### Semantic Search

```python
from autogluon.multimodal.utils import semantic_search

# Text-to-image search
text_to_image_hits = semantic_search(
    matcher=predictor,
    query_data=test_text_data.iloc[[3]],
    response_data=test_image_data,
    top_k=5,
)

# Image-to-text search
image_to_text_hits = semantic_search(
    matcher=predictor,
    query_data=test_image_data.iloc[[6]],
    response_data=test_text_data,
    top_k=5,
)
```

## Key Notes
- The predictor uses CLIP (openai/clip-vit-base-patch32) as the backbone
- Text-to-image recalls are typically higher than image-to-text recalls due to the 1:5 image-to-text ratio
- Finetuning can significantly improve performance over zero-shot prediction
- The model supports both binary matching prediction and probability scores

For customization options, refer to the AutoMM documentation.