# Condensed: Text-to-Text Semantic Matching with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Text-to-Text Semantic Matching with AutoMM

## Overview
This tutorial demonstrates using AutoMM for text-to-text semantic matching using the Stanford Natural Language Inference (SNLI) corpus. The task involves computing similarity between text pairs and classifying them as semantically similar or different.

## Data Preparation
The SNLI dataset contains sentence pairs labeled as:
- Entailment (positive pairs, label=1)
- Contradiction (negative pairs, label=0)
- Neutral (discarded)

```python
!pip install autogluon.multimodal

from autogluon.core.utils.loaders import load_pd
import pandas as pd

snli_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_train.csv', delimiter="|")
snli_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_test.csv', delimiter="|")
```

## Model Training
AutoMM uses BERT to project sentences into vectors and treats matching as a classification problem:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="text_similarity",
    query="premise",          # first sentence column
    response="hypothesis",    # second sentence column
    label="label",           # label column
    match_label=1,           # label indicating semantic similarity
    eval_metric='auc',       
)

predictor.fit(
    train_data=snli_train,
    time_limit=180,
)
```

## Model Evaluation
Evaluate performance on test data:

```python
score = predictor.evaluate(snli_test)
print("evaluation score: ", score)
```

## Making Predictions

### Single Pair Prediction
```python
pred_data = pd.DataFrame.from_dict({
    "premise":["The teacher gave his speech to an empty room."], 
    "hypothesis":["There was almost nobody when the professor was talking."]
})

predictions = predictor.predict(pred_data)
print('Predicted entities:', predictions[0])
```

### Probability Prediction
```python
probabilities = predictor.predict_proba(pred_data)
print(probabilities)
```

### Embedding Extraction
Extract embeddings for individual sentences:

```python
embeddings_1 = predictor.extract_embedding({"premise":["The teacher gave his speech to an empty room."]})
embeddings_2 = predictor.extract_embedding({"hypothesis":["There was almost nobody when the professor was talking."]})
```

## Key Implementation Notes
- Labels must be binary
- `match_label` should be defined based on task context
- Model uses BERT for semantic understanding
- Supports both classification and embedding extraction

For customization options, refer to the AutoMM customization documentation.