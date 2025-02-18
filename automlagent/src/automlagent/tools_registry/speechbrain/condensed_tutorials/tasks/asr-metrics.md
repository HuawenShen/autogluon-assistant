# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides implementation details for various Automatic Speech Recognition (ASR) evaluation metrics using SpeechBrain. It covers code implementations for traditional metrics like Word Error Rate (WER) and Character Error Rate (CER), as well as advanced semantic metrics including Part-of-speech Error Rate (POSER), Lemma Error Rate (LER), Embedding Error Rate (EmbER), BERTScore, and Sentence Semantic Distance (SemDist). The tutorial helps with tasks involving ASR output evaluation, offering both surface-level and semantic comparison techniques. Key features include hyperpyyaml configurations, integration with various NLP libraries (spacy, flair), language model utilization, and error rate calculations with different granularities and semantic considerations.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# ASR Metrics Tutorial

## Key Metrics Overview

### Word Error Rate (WER)
```python
# Basic WER implementation
wer_hparams = load_hyperpyyaml("""
wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats
""")

# Usage
wer_hparams["wer_stats"].clear()
wer_hparams["wer_stats"].append(
    ids=list(range(len(refs))), 
    predict=hyps,
    target=refs,
)
```

**Formula**: $\dfrac{\#insertions + \#substitutions + \#deletions}{\#refwords}$

### Character Error Rate (CER)
```python
cer_hparams = load_hyperpyyaml("""
cer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats
    split_tokens: True
""")
```

### Part-of-speech Error Rate (POSER)
```python
poser_hparams = load_hyperpyyaml("""
# dPOSER
wer_stats_dposer: !new:speechbrain.utils.metric_stats.ErrorRateStats

# uPOSER with synonym dictionary
uposer_dict: !apply:speechbrain.utils.dictionaries.SynonymDictionary.from_json_path
    path: ./uposer.json
wer_stats_uposer: !new:speechbrain.utils.metric_stats.ErrorRateStats
    equality_comparator: !ref <uposer_dict>

pos_tagger: !apply:speechbrain.lobes.models.flair.FlairSequenceTagger.from_hf
    source: "qanastek/pos-french"
""")
```

### Lemma Error Rate (LER)
```python
ler_hparams = load_hyperpyyaml("""
ler_model: !apply:speechbrain.lobes.models.spacy.SpacyPipeline.from_name
    name: fr_core_news_md
    exclude: ["tagger", "parser", "ner", "textcat"]
wer_stats_ler: !new:speechbrain.utils.metric_stats.ErrorRateStats
""")
```

### Embedding Error Rate (EmbER)
```python
ember_hparams = load_hyperpyyaml("""
wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats
ember_embeddings: !apply:speechbrain.lobes.models.flair.embeddings.FlairEmbeddings.from_hf
    embeddings_class: !name:flair.embeddings.FastTextEmbeddings
    source: facebook/fasttext-fr-vectors

ember_metric: !new:speechbrain.utils.metric_stats.EmbeddingErrorRateSimilarity
    low_similarity_weight: 1.0
    high_similarity_weight: 0.1
    threshold: 0.4
""")
```

### BERTScore
```python
bertscore_hparams = load_hyperpyyaml("""
bertscore_stats: !new:speechbrain.utils.bertscore.BERTScoreStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: camembert/camembert-large
        num_layers: 8
""")
```

### Sentence Semantic Distance (SemDist)
```python
semdist_hparams = load_hyperpyyaml("""
semdist_stats: !new:speechbrain.utils.semdist.SemDistStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: camembert/camembert-large
    method: meanpool
""")
```

## Important Notes

1. WER limitations:
   - All errors weighted equally
   - Doesn't account for semantic similarity
   - Can exceed 100%

2. Implementation considerations:
   - CER is slower due to character-level computations
   - EmbER requires additional embedding setup
   - BERTScore compares cosine similarity of all targets/predictions
   - SemDist uses single embedding per sentence (normalized x1000)

3. Dependencies:
   - SpeechBrain
   - spacy
   - flair
   - Required models will be downloaded automatically

4. Best Practices:
   - Consider using multiple metrics for comprehensive evaluation
   - Choose metrics based on specific use case requirements
   - Account for language-specific characteristics