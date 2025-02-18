# Condensed: Image-Text Semantic Matching with AutoMM

Summary: This tutorial demonstrates implementing image-text semantic matching using AutoGluon's MultiModalPredictor, covering essential techniques for both zero-shot and finetuned approaches. It helps with tasks like image-caption matching, semantic search, and embedding extraction. Key functionalities include dataset preparation with proper path handling, predictor configuration for image-text similarity problems, zero-shot evaluation, model finetuning, prediction methods (including probability scores and embedding extraction), and semantic search operations for both text-to-image and image-to-text retrieval. The implementation uses CLIP as the default backbone and supports various evaluation metrics with recall@k.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Image-Text Semantic Matching with AutoMM

## Key Implementation Details

### Setup
```python
!pip install autogluon.multimodal
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

### Dataset Preparation
1. Load Flickr30K dataset (image-caption pairs)
2. Structure requirements:
   - DataFrame with image and text columns
   - Image paths must be absolute
   - Each image can have multiple caption pairs

```python
# Convert relative to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply to dataframes
train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### Predictor Configuration
```python
predictor = MultiModalPredictor(
    query=text_col,
    response=image_col,
    problem_type="image_text_similarity",
    eval_metric="recall"
)
```

### Key Operations

1. **Zero-shot Evaluation**
```python
scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_text_data,
    response_data=test_image_data,
    label=test_label_col,
    cutoffs=[1, 5, 10]
)
```

2. **Finetuning**
```python
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180  # in seconds
)
```

3. **Prediction Methods**
```python
# Match prediction
predictions = predictor.predict(test_data)

# Matching probabilities
probabilities = predictor.predict_proba(test_data)

# Extract embeddings
image_embeddings = predictor.extract_embedding({image_col: image_paths})
text_embeddings = predictor.extract_embedding({text_col: text_list})
```

4. **Semantic Search**
```python
from autogluon.multimodal.utils import semantic_search

# Text-to-image search
results = semantic_search(
    matcher=predictor,
    query_data=text_data,
    response_data=image_data,
    top_k=5
)
```

## Important Notes

1. Default backbone: `openai/clip-vit-base-patch32`
2. Evaluation metrics use recall@k where k is specified in cutoffs
3. For image-to-text retrieval, recall@1 has 20% upper bound due to 1:5 image-text ratio
4. Finetuning can significantly improve performance over zero-shot
5. Both text-to-image and image-to-text search are supported

This implementation supports both zero-shot and finetuned approaches for image-text matching, with flexible evaluation and search capabilities.