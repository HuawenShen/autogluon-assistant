# Condensed: Image-Text Semantic Matching with AutoMM - Zero-Shot

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Zero-Shot Image-Text Semantic Matching with AutoMM

## Overview
AutoMM enables zero-shot image-text matching using CLIP, which features:
- Two-tower architecture (separate image and text encoders)
- Pre-trained embeddings for efficient similarity matching
- Offline embedding extraction for scalability

## Installation
```python
!pip install autogluon.multimodal
```

## Basic Usage

### 1. Prepare Data
```python
from autogluon.multimodal import download

# Sample texts and image URLs
texts = [
    "A cheetah chases prey on across a field.",
    "A man is eating a piece of bread.",
    # ... additional examples ...
]

urls = ['http://farm4.staticflickr.com/3179/2872917634_f41e6987a8_z.jpg',
        # ... additional URLs ...
]

image_paths = [download(url) for url in urls]
```

### 2. Extract Embeddings
```python
from autogluon.multimodal import MultiModalPredictor

# Initialize predictor
predictor = MultiModalPredictor(problem_type="image_text_similarity")

# Extract embeddings
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
```

### 3. Image-Text Matching Tasks

#### Image Retrieval with Text Query
```python
from autogluon.multimodal.utils import semantic_search

# Search images matching a text query
hits = semantic_search(
    matcher=predictor,
    query_embeddings=text_embeddings[6][None,],
    response_embeddings=image_embeddings,
    top_k=5
)
```

#### Text Retrieval with Image Query
```python
hits = semantic_search(
    matcher=predictor,
    query_embeddings=image_embeddings[4][None,],
    response_embeddings=text_embeddings,
    top_k=5
)
```

### 4. Direct Matching Predictions

```python
# Initialize predictor for matching
predictor = MultiModalPredictor(
    query="abc",
    response="xyz",
    problem_type="image_text_similarity"
)

# Predict matches
pred = predictor.predict({"abc": [image_paths[4]], "xyz": [texts[3]]})

# Get matching probabilities
proba = predictor.predict_proba({"abc": [image_paths[4]], "xyz": [texts[3]]})
```

## Key Implementation Notes
- Use `as_tensor=True` for efficient embedding operations
- Embeddings can be extracted offline for large-scale applications
- Semantic search supports customizable `top_k` results
- Predictor requires explicit `query` and `response` names for direct matching

## Best Practices
- Extract embeddings once and reuse for multiple queries
- Use probability predictions when threshold customization is needed
- Consider offline embedding extraction for large datasets

For customization options, refer to the AutoMM customization documentation.