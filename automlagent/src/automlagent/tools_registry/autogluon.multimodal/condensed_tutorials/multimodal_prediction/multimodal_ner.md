# Condensed: AutoMM for Entity Extraction with Text and Image - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Entity Extraction with Text and Image - Quick Start

## Overview
This tutorial demonstrates how to build a multimodal named entity recognition (NER) model using text and image data with AutoGluon's AutoMM. The example uses a Twitter dataset where each tweet contains both text and an image.

## Setup

```python
!pip install autogluon.multimodal

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

## Data Preparation

1. Download and extract the Twitter dataset:
```python
download_dir = './ag_automm_tutorial_ner'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/ner/multimodal_ner.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = download_dir + '/multimodal_ner'
train_data = pd.read_csv(f'{dataset_path}/twitter17_train.csv')
test_data = pd.read_csv(f'{dataset_path}/twitter17_test.csv')
label_col = 'entity_annotations'
```

2. Process image paths:
```python
image_col = 'image'
# Use first image for quick tutorial
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    p = ';'.join([os.path.abspath(base_folder+path) for path in path_l])
    return p

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

Key configuration:
- Set problem_type to "ner"
- Specify text_ner column type for entity extraction
- Set time limit for training duration

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    column_types={"text_snippet":"text_ner"},
    time_limit=300, #seconds
)
```

## Evaluation and Prediction

1. Evaluate model performance:
```python
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1"])
```

2. Make predictions:
```python
prediction_input = test_data.drop(columns=label_col).head(1)
predictions = predictor.predict(prediction_input)

# Display predictions
for entity in predictions[0]:
    print(f"Word '{prediction_input.text_snippet[0][entity['start']:entity['end']]}' belongs to group: {entity['entity_group']}")
```

## Model Persistence and Continued Training

Load and continue training a saved model:
```python
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1'])
```

## Important Notes
- AutoMM automatically detects data modalities and selects appropriate models
- For multiple backbones, AutoMM implements late-fusion
- The model is automatically saved during training
- Continued training can be used to improve model performance

For customization options, refer to the AutoMM customization documentation.