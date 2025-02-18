# Condensed: AutoMM for Entity Extraction with Text and Image - Quick Start

Summary: This tutorial demonstrates implementing multimodal entity extraction using AutoGluon's MultiModalPredictor, specifically focusing on processing text and image data from tweets. It covers essential techniques for dataset preparation (image path handling), model configuration for NER tasks, and training workflows. The tutorial helps with tasks like setting up multimodal NER models, handling image-text data preprocessing, and implementing transfer learning. Key features include automatic modality detection, model selection from multimodal pools, late-fusion model implementation, model persistence, and continuous training capabilities. It provides practical code examples for model training, evaluation, prediction, and best practices for handling NER-specific configurations.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Multimodal Entity Extraction - Quick Start

## Key Setup
```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
```

## Dataset Preparation
1. Dataset contains tweets with text and images
2. Critical preprocessing steps:

```python
# Expand image paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    p = ';'.join([os.path.abspath(base_folder+path) for path in path_l])
    return p

# Apply to image column
train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

### Essential Configuration
```python
predictor = MultiModalPredictor(
    problem_type="ner",  # Specify NER task
    label=label_col,
    path=model_path
)

# Training with critical parameters
predictor.fit(
    train_data=train_data,
    column_types={"text_snippet":"text_ner"},  # Important: Specify text_ner column type
    time_limit=300  # Training time in seconds
)
```

### Key Features
- Automatic modality detection
- Automatic model selection from multimodal pools
- Late-fusion model implementation for multiple backbones

## Evaluation and Prediction
```python
# Evaluate
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1"])

# Predict
predictions = predictor.predict(prediction_input)
```

## Model Persistence and Transfer Learning
```python
# Load saved model
new_predictor = MultiModalPredictor.load(model_path)

# Continue training
new_predictor.fit(
    train_data, 
    time_limit=60,
    save_path=new_model_path
)
```

## Important Notes
1. Set `problem_type="ner"` for entity extraction tasks
2. Use `column_types={"text_snippet":"text_ner"}` to specify NER text columns
3. Model automatically saves during training
4. Supports continuous training on loaded models

## Best Practices
1. Ensure correct image path expansion for training
2. Specify appropriate time limits based on dataset size
3. Use evaluation metrics suitable for NER tasks
4. Consider continuous training for model improvement