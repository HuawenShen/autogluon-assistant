# Condensed: Image-Text Semantic Matching with AutoMM - Zero-Shot

Summary: This tutorial demonstrates implementing zero-shot image-text semantic matching using AutoMM's MultiModalPredictor with CLIP model. It covers three main functionalities: image retrieval, text retrieval, and pair matching. Key implementation techniques include efficient embedding extraction (with offline storage capability), semantic search operations, and pair matching predictions. The tutorial shows how to initialize the predictor, extract embeddings with tensors, perform search operations with customizable top-k results, and conduct pair matching with probability scores. It's particularly useful for building image-text search systems, content matching applications, and semantic similarity tasks without requiring model training.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Zero-Shot Image-Text Semantic Matching with AutoMM

## Key Concepts
- Uses CLIP model for image-text matching without training
- Two-tower architecture: separate encoders for images and text
- Enables offline embedding extraction for scalability
- Supports image retrieval, text retrieval, and pair matching

## Implementation

### 1. Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import semantic_search

# Initialize predictor
predictor = MultiModalPredictor(problem_type="image_text_similarity")
```

### 2. Embedding Extraction
```python
# Extract embeddings
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
```

### 3. Image-Text Search Operations

#### Image Retrieval with Text Query
```python
hits = semantic_search(
    matcher=predictor,
    query_embeddings=text_embeddings[query_idx][None,],
    response_embeddings=image_embeddings,
    top_k=5
)
```

#### Text Retrieval with Image Query
```python
hits = semantic_search(
    matcher=predictor,
    query_embeddings=image_embeddings[query_idx][None,],
    response_embeddings=text_embeddings,
    top_k=5
)
```

### 4. Pair Matching Predictions

```python
# Initialize predictor for pair matching
predictor = MultiModalPredictor(
    query="abc",
    response="xyz",
    problem_type="image_text_similarity"
)

# Predict matches
pred = predictor.predict({
    "abc": [image_paths[4]], 
    "xyz": [texts[3]]
})

# Get matching probabilities
proba = predictor.predict_proba({
    "abc": [image_paths[4]], 
    "xyz": [texts[3]]
})
```

## Important Notes
- Embedding extraction can be done offline for better scalability
- Use `as_tensor=True` for efficient embedding operations
- Specify `query` and `response` parameters when doing pair matching
- The model uses cosine similarity for matching scores

## Best Practices
1. Extract embeddings in batch for better performance
2. Store embeddings offline for large-scale applications
3. Use appropriate `top_k` values based on your use case
4. Consider probability thresholds for matching decisions