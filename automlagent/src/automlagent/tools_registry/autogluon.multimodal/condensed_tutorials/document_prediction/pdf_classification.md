# Condensed: Classifying PDF Documents with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Classifying PDF Documents with AutoMM

## Prerequisites
- Requires poppler installation:
  - Windows: Install from [poppler-windows](https://github.com/oschwartz10612/poppler-windows) and add bin/ to PATH
  - Mac: `brew install poppler`
  - Linux: Install from [poppler source](https://poppler.freedesktop.org)

## Setup and Data Preparation

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

# Download and prepare dataset
download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and split data
dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)
```

## Configure Document Paths
```python
from autogluon.multimodal.utils.misc import path_expander

DOC_PATH_COL = "doc_path"
# Ensure correct document paths
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

## Create and Train PDF Classifier
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

## Model Evaluation and Prediction

```python
# Evaluate on test set
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])

# Single prediction
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(f"Ground-truth label: {test_data.iloc[0]['label']}, Prediction: {predictions}")

# Probability predictions
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(proba)
```

## Feature Extraction
```python
# Extract document embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(feature[0].shape)
```

## Key Implementation Notes
- Uses LayoutLM base model for document understanding
- Automatically handles PDF processing and text recognition
- Supports both classification and feature extraction
- Dataset contains binary classification (resume vs. historical documents)
- Training time limit can be adjusted based on needs

For customization options, refer to the AutoMM customization documentation.