# Condensed: Text Semantic Search with AutoMM

Summary: This tutorial demonstrates implementing semantic search using AutoGluon's MultiModalPredictor, covering both pure semantic search and hybrid BM25 approaches. It provides implementation details for text-to-embedding conversion using the sentence-transformers/all-MiniLM-L6-v2 model, similarity matching, and hybrid scoring techniques. The tutorial helps with tasks like document retrieval, embedding extraction, and ranking evaluation using NDCG metrics. Key features include data preprocessing, embedding generation, hybrid search implementation with configurable weights (β=0.3), and performance optimization techniques for large-scale deployments using tools like Faiss. It's particularly useful for building production-ready semantic search systems that combine traditional BM25 with modern embedding-based approaches.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Text Semantic Search with AutoMM

## Key Concepts
- Semantic search converts text to vector embeddings ($\phi(x)$) for similarity matching
- Advantages over classical methods:
  - Matches meaning rather than word usage
  - Computationally efficient with pre-computed embeddings
  - Supports multi-modal search

## Implementation

### 1. Setup and Data Preparation
```python
!pip install autogluon.multimodal ir_datasets
import ir_datasets
import pandas as pd

# Load dataset
dataset = ir_datasets.load("beir/nfcorpus/test")
doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())

# Data preprocessing
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)
```

### 2. AutoMM Implementation
```python
from autogluon.multimodal import MultiModalPredictor

# Initialize predictor
predictor = MultiModalPredictor(
    query=query_id_col,
    response=doc_id_col,
    label=label_col,
    problem_type="text_similarity",
    hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"}
)

# Evaluate ranking
results = predictor.evaluate(
    labeled_data,
    query_data=query_data[[query_id_col]],
    response_data=doc_data[[doc_id_col]],
    id_mappings=id_mappings,
    cutoffs=cutoffs,
    metrics=["ndcg"]
)

# Extract embeddings
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
```

### 3. Hybrid BM25 Implementation
```python
def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    # Combine BM25 and semantic scores
    score = β * normalized_BM25 + (1 - β) * score_of_plm
    
    # Key parameters
    recall_num = 1000  # Number of documents to recall
    beta = 0.3        # Weight for BM25 score
```

## Critical Configurations
- Use `sentence-transformers/all-MiniLM-L6-v2` checkpoint for embeddings
- BM25 parameters: k1=1.2, b=0.75
- Hybrid search: β=0.3, recall_num=1000

## Best Practices
1. Pre-compute and store document embeddings for production systems
2. Use efficient similarity search methods (e.g., Faiss) for large-scale deployments
3. Consider hybrid approaches combining BM25 and semantic search for better results

## Performance Metrics
- Use NDCG (Normalized Discounted Cumulative Gain) with different cutoffs
- Evaluate at cutoffs=[5, 10, 20]
- Hybrid approach typically outperforms pure BM25 or semantic search