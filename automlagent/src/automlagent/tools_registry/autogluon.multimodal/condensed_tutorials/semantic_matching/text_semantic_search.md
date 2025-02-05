# Condensed: Text Semantic Search with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# Text Semantic Search with AutoMM

## 1. Introduction

Semantic search converts text into vector embeddings to match queries with documents based on semantic meaning rather than exact word matches. Key advantages:
- Matches related concepts even with different wording
- Computationally efficient through pre-computed embeddings
- Enables multi-modal search capabilities

## 2. Setup and Data Preparation

```python
!pip install autogluon.multimodal
!pip3 install ir_datasets

import ir_datasets
import pandas as pd

# Load dataset
dataset = ir_datasets.load("beir/nfcorpus/test")
doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())

# Configure columns
label_col = "relevance"
query_id_col = "query_id"
doc_id_col = "doc_id"
text_col = "text"

# Prepare data mappings
id_mappings = {
    query_id_col: query_data.set_index(query_id_col)[text_col], 
    doc_id_col: doc_data.set_index(doc_id_col)[text_col]
}

# Clean data
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)
query_data = query_data.drop("url", axis=1)
```

## 3. BM25 Baseline Implementation

```python
from rank_bm25 import BM25Okapi
from collections import defaultdict
import nltk
import numpy as np

def tokenize_corpus(corpus):
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    tokenized_docs = []
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        tokenized_doc = [w for w in tokens if w not in stop_words and len(w) > 2]
        tokenized_docs.append(tokenized_doc)
    return tokenized_docs

def evaluate_bm25(doc_data, query_data, qrel_dict, cutoffs):
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    results = rank_documents_bm25(
        query_data[text_col].tolist(), 
        query_data[query_id_col].tolist(), 
        doc_data[doc_id_col].tolist(), 
        max(cutoffs), 
        bm25_model
    )
    return compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
```

## 4. AutoMM Implementation

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
    metrics=["ndcg"],
)

# Extract embeddings for search
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
```

## 5. Hybrid BM25 Implementation

Combines BM25 and semantic search scores:
- Uses BM25 for initial document retrieval
- Applies semantic scoring on retrieved documents
- Final score: `β * normalized_BM25 + (1 - β) * semantic_score`

```python
def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    # Get BM25 scores
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    bm25_scores = rank_documents_bm25(query_data[text_col].tolist(), 
                                    query_data[query_id_col].tolist(), 
                                    doc_data[doc_id_col].tolist(), 
                                    recall_num, 
                                    bm25_model)
    
    # Normalize scores and combine with semantic similarity
    results = combine_scores(bm25_scores, query_embeds, doc_embeds, beta)
    return results

# Usage
recall_num = 1000
beta = 0.3
results = evaluate_hybridBM25(query_data, query_embeds, doc_data, doc_embeds, 
                            recall_num, beta, cutoffs)
```

Key Parameters:
- `recall_num`: Number of documents to retrieve in first stage (1000)
- `beta`: Weight between BM25 and semantic scores (0.3)
- `cutoffs`: Evaluation thresholds for NDCG calculation [5, 10, 20]

The hybrid approach typically outperforms both pure BM25 and pure semantic search.