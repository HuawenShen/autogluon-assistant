# Condensed: AutoMM for Image + Text + Tabular - Quick Start

Summary: This tutorial demonstrates how to implement multimodal machine learning using AutoGluon's MultiModalPredictor, which can handle combined image, text, and tabular data. It covers essential techniques for data preparation (path handling), model training, prediction, evaluation, and embedding extraction. The tutorial helps with tasks like automated model selection, multimodal fusion, and model persistence. Key features include automatic problem type inference, modality detection, late-fusion implementation, and support for various prediction types (standard predictions, probability predictions for classification). It's particularly useful for developers looking to implement end-to-end multimodal ML pipelines with minimal manual configuration.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Image + Text + Tabular - Quick Start

## Key Setup
```python
!pip install autogluon.multimodal
import numpy as np
from autogluon.multimodal import MultiModalPredictor
```

## Implementation Details

### Data Preparation
1. Load multimodal data containing images, text, and tabular features
2. Process image paths:
```python
# Expand image paths to full path
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply to image column
train_data[image_col] = train_data[image_col].apply(
    lambda ele: path_expander(ele.split(';')[0], base_folder=dataset_path)
)
```

### Training
```python
predictor = MultiModalPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    time_limit=120  # seconds
)
```

### Key Operations
1. Prediction:
```python
# Standard prediction
predictions = predictor.predict(test_data)

# Probability prediction (classification only)
probas = predictor.predict_proba(test_data)
```

2. Evaluation:
```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

3. Embedding Extraction:
```python
embeddings = predictor.extract_embedding(test_data)
```

4. Model Persistence:
```python
# Save
predictor.save(model_path)

# Load
loaded_predictor = MultiModalPredictor.load(model_path)
```

## Important Notes & Best Practices

1. **Security Warning**: `MultiModalPredictor.load()` uses pickle - only load trusted data
2. AutoMM automatically:
   - Infers problem type (classification/regression)
   - Detects data modalities
   - Selects appropriate models
   - Implements late-fusion for multiple backbones

3. Supported Data Types:
   - Images
   - Text
   - Tabular features
   - Any combination of above

4. For classification tasks:
   - Use `predict_proba()` for class probabilities
   - Note: Will raise exception for regression tasks

5. Time limit can be adjusted based on dataset size and required performance

This condensed version maintains all critical implementation details while removing explanatory text and redundant examples.