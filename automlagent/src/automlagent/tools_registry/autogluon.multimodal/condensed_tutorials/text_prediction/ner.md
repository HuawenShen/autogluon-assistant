# Condensed: AutoMM for Named Entity Recognition - Quick Start

Summary: This tutorial demonstrates implementing Named Entity Recognition (NER) using AutoGluon's MultiModalPredictor. It covers essential techniques for training NER models with structured JSON annotations, including data preparation with specific entity formatting, model initialization, training configuration with customizable BERT-based backbones, and prediction workflows. Key functionalities include flexible model selection (with ELECTRA as a lightweight option), evaluation using seqeval metrics, handling BIO format, model persistence, and continuous training capabilities. The tutorial helps with tasks like setting up NER training pipelines, configuring hyperparameters, performing model evaluation, and making predictions with probability scores.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Named Entity Recognition - Quick Start

## Data Preparation
- Required format: DataFrame with text column and annotation column
- Annotation format must be JSON with specific structure:
```python
[{
    "entity_group": "CATEGORY",
    "start": char_start_position,
    "end": char_end_position
}]
```

Example:
```python
annotation = [
    {"entity_group": "PERSON", "start": 0, "end": 15},
    {"entity_group": "LOCATION", "start": 28, "end": 35}
]
```

## Implementation

1. Install and import:
```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
```

2. Training setup:
```python
predictor = MultiModalPredictor(
    problem_type="ner", 
    label="entity_annotations",
    path="model_path"
)

# Training
predictor.fit(
    train_data=train_data,
    hyperparameters={
        'model.ner_text.checkpoint_name':'google/electra-small-discriminator'
    },
    time_limit=300  # in seconds
)
```

3. Evaluation:
```python
metrics = predictor.evaluate(
    test_data,  
    metrics=['overall_recall', 'overall_precision', 'overall_f1']
)
```

4. Prediction:
```python
# Basic prediction
predictions = predictor.predict({'text_snippet': [text]})

# Prediction with probabilities
prob_predictions = predictor.predict_proba({'text_snippet': [text]})
```

## Key Configurations and Best Practices

1. Model Selection:
- Default: BERT-based models
- Lightweight option: 'google/electra-small-discriminator'

2. Training Parameters:
- Recommended time_limit: 30-60 minutes for production use
- Can specify custom backbone models via hyperparameters

3. Evaluation Metrics:
- Uses seqeval metrics
- Available metrics: overall_recall, overall_precision, overall_f1, overall_accuracy
- Entity-specific metrics available using entity group names

## Important Notes

- BIO format (Beginning-Inside-Outside) is supported but optional
- Model automatically saves during training
- Continuous training possible with loaded models
- Visualization available through `visualize_ner` utility
- JSON annotation format must use exact keys: entity_group, start, end

## Model Reloading
```python
loaded_predictor = MultiModalPredictor.load("model_path")
# Continue training if needed
loaded_predictor.fit(new_data, time_limit=60, save_path="new_model_path")
```