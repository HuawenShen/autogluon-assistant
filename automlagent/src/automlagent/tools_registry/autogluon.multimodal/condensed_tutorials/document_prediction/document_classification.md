# Condensed: AutoMM for Scanned Document Classification

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Scanned Document Classification

## Overview
This tutorial demonstrates how to build a scanned document classifier using Autogluon Multimodal. The system can automatically classify digitized documents (like budgets, emails, and forms) using both text and visual features.

## Setup and Data Preparation

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

# Download and extract dataset
download_dir = './ag_automm_tutorial_doc_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and split data
dataset_path = os.path.join(download_dir, "rvl_cdip_sample")
rvl_cdip_data = pd.read_csv(f"{dataset_path}/rvl_cdip_train_data.csv")
train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
test_data = rvl_cdip_data.drop(train_data.index)

# Expand document paths
from autogluon.multimodal.utils.misc import path_expander
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

## Model Training
AutoMM automatically handles:
- Text recognition (handwritten and typed)
- Layout information extraction
- Visual feature processing

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
        "optimization.top_k_average_method": "best",
    },
    time_limit=120,
)
```

Key Parameters:
- `label`: Target variable column name
- `model.document_transformer.checkpoint_name`: Foundation model selection (supports layoutlmv3, layoutlmv2, layoutlm-base, layoutxlm, bert, deberta)

## Evaluation and Prediction

```python
# Evaluate model
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])

# Make predictions
doc_path = test_data.iloc[1][DOC_PATH_COL]
predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
proba = predictor.predict_proba({DOC_PATH_COL: [doc_path]})
```

## Feature Extraction
Extract document embeddings for downstream tasks:

```python
feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})
print(feature[0].shape)
```

## Important Notes
- The dataset uses three document categories: budget (0), email (1), and form (2)
- Model customization is available through hyperparameters
- Supports both document-specific and general text models
- Can extract embeddings for transfer learning or other applications

For advanced customization, refer to the AutoMM customization documentation.