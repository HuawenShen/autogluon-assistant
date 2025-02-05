# Condensed: AutoMM for Text - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# AutoMM for Text - Quick Start

## Overview
AutoGluon's `MultiModalPredictor` handles text data for tasks like sentiment analysis and sentence similarity. This tutorial demonstrates implementation using two NLP tasks.

## Setup

```python
!pip install autogluon.multimodal

import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Sentiment Analysis Implementation

### Data Loading
```python
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # Adjust for your needs
train_data = train_data.sample(n=subsample_size, random_state=0)
```

### Model Training
```python
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label='label', eval_metric='acc', path=model_path)
predictor.fit(train_data, time_limit=180)  # Increase time_limit for better performance
```

### Evaluation and Prediction
```python
# Evaluate with multiple metrics
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])

# Single prediction
sentence1 = "it's a charming and often affecting journey."
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({'sentence': [sentence1, sentence2]})

# Probability predictions
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})
```

### Model Persistence
```python
# Save and load model
loaded_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
loaded_predictor.save(new_model_path)
```

### Feature Extraction
```python
# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
```

## Sentence Similarity Implementation

### Data Preparation
```python
sts_train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]
sts_test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
```

### Model Training and Evaluation
```python
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label='score', path=sts_model_path)
predictor_sts.fit(sts_train_data, time_limit=60)

# Evaluate with multiple metrics
test_score = predictor_sts.evaluate(sts_test_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
```

## Key Points
- `MultiModalPredictor` automatically determines the prediction problem type
- Supports both classification and regression tasks
- Integrates with timm, huggingface/transformers, and openai/clip
- For production use, increase `time_limit` or set to None
- Supports multiple text columns in data tables

## Important Notes
- WARNING: Use `MultiModalPredictor.load()` only with trusted data sources due to pickle security concerns
- Save/load functionality is built-in and automatic after training
- Customize the predictor by referring to the Customization documentation

This condensed version maintains all essential implementation details while removing redundant explanations and supplementary examples.