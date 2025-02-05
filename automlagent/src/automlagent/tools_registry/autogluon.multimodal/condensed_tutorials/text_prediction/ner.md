# Condensed: AutoMM for Named Entity Recognition - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the AutoMM NER tutorial:

# AutoMM for Named Entity Recognition - Quick Start

## Overview
Named Entity Recognition (NER) identifies and categorizes entities (e.g., person names, locations) in text. AutoMM simplifies NER model training and deployment.

## Data Preparation
Required format:
- DataFrame with text and annotation columns
- Annotations in JSON format with:
  - entity_group: category label
  - start: character position where entity begins
  - end: character position where entity ends

```python
!pip install autogluon.multimodal

import json
# Example annotation format
annotation = json.dumps([
    {"entity_group": "PERSON", "start": 0, "end": 15},
    {"entity_group": "LOCATION", "start": 28, "end": 35}
])
```

Visualization utility:
```python
from autogluon.multimodal.utils import visualize_ner

sentence = "Albert Einstein was born in Germany and is widely acknowledged to be one of the greatest physicists."
annotation = [{"entity_group": "PERSON", "start": 0, "end": 15},
              {"entity_group": "LOCATION", "start": 28, "end": 35}]

visualize_ner(sentence, annotation)
```

## Training
```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"

predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'google/electra-small-discriminator'},
    time_limit=300, # seconds
)
```

Key parameters:
- problem_type: Set to "ner"
- label: Column name containing annotations
- time_limit: Training duration (recommended: 30-60 minutes for production)
- hyperparameters: Model configuration (using 'google/electra-small-discriminator' as example)

## Evaluation
```python
# Supported metrics: overall_recall, overall_precision, overall_f1, overall_accuracy
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1", "actor"])
```

## Prediction and Visualization
```python
sentence = "Game of Thrones is an American fantasy drama television series created by David Benioff"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])

# Get prediction probabilities
predictions = predictor.predict_proba({'text_snippet': [sentence]})
```

## Model Management
Load and continue training:
```python
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
```

## Important Notes
- Use json.dumps() to convert Python objects to JSON strings before creating DataFrames
- BIO format (Beginning-Inside-Outside) is supported but optional
- Longer training times (>30 minutes) recommended for production models
- Model automatically saves during training
- Entity-specific metrics available by using entity group name in evaluation

This condensed version maintains all essential implementation details while removing redundant explanations and supplementary information not critical for implementation.