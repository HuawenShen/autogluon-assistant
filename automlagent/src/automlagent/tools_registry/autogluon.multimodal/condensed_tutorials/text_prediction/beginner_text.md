# Condensed: AutoMM for Text - Quick Start

Summary: This tutorial demonstrates the implementation of AutoGluon's MultiModalPredictor for text-based machine learning tasks, specifically focusing on classification (sentiment analysis) and regression (sentence similarity). It provides code examples for model training, evaluation, prediction, and embedding extraction using the MultiModalPredictor API. Key functionalities covered include model saving/loading, multiple evaluation metrics (acc, f1, rmse, pearsonr, spearmanr), and handling tabular data with text columns. The tutorial helps with tasks requiring automated text classification, regression, and feature extraction, while highlighting important configurations, best practices for training, and system limitations regarding deep learning models and memory usage.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Text - Quick Start Guide

## Key Components
- Uses `MultiModalPredictor` for text-based tasks
- Supports classification and regression
- Integrates with timm, huggingface/transformers, and openai/clip

## Implementation Examples

### 1. Sentiment Analysis (Classification)

```python
from autogluon.multimodal import MultiModalPredictor

# Setup
predictor = MultiModalPredictor(
    label='label',
    eval_metric='acc',
    path='model_path'
)

# Train
predictor.fit(train_data, time_limit=180)

# Evaluate
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])

# Predict
predictions = predictor.predict({'sentence': [text1, text2]})
probabilities = predictor.predict_proba({'sentence': [text1, text2]})
```

### 2. Sentence Similarity (Regression)

```python
predictor_sts = MultiModalPredictor(
    label='score',
    path='sts_model_path'
)

# Train
predictor_sts.fit(train_data, time_limit=60)

# Evaluate
scores = predictor_sts.evaluate(test_data, 
                              metrics=['rmse', 'pearsonr', 'spearmanr'])
```

## Important Features

### Model Management
```python
# Save
predictor.save(model_path)

# Load
loaded_predictor = MultiModalPredictor.load(model_path)
```

### Feature Extraction
```python
# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
```

## Critical Configurations

1. Training Parameters:
   - `time_limit`: Set training duration (None for unrestricted)
   - `label`: Specify target column name
   - `eval_metric`: Define evaluation metric

2. Evaluation Options:
   - Multiple metrics supported: 'acc', 'f1', 'rmse', 'pearsonr', 'spearmanr'

## Best Practices

1. Training:
   - Use longer `time_limit` for production models (1+ hours recommended)
   - Provide sufficient training data for better performance

2. Security:
   - WARNING: Only load models from trusted sources due to pickle security risks

3. Data Format:
   - Input data should be in tabular format (DataFrame)
   - Supports multiple text columns
   - Labels should be in separate column

## Limitations
- Focuses on deep learning models
- Requires properly formatted tabular data
- Memory intensive for large datasets