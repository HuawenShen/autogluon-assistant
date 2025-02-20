# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides implementation details for various Automatic Speech Recognition (ASR) evaluation metrics using SpeechBrain. It covers code implementations for traditional metrics like Word Error Rate (WER) and Character Error Rate (CER), as well as advanced semantic metrics including Part-of-speech Error Rate (POSER), Lemma Error Rate (LER), Embedding Error Rate (EmbER), BERTScore, and SemDist. The tutorial demonstrates how to configure and use these metrics with specific dependencies (SpeechBrain, Spacy, Flair), handle embeddings, and integrate language models. It's particularly useful for tasks involving ASR evaluation, semantic similarity assessment, and natural language processing metric implementations.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# ASR Metrics Tutorial

## Key Metrics Overview

### Word Error Rate (WER)
```python
# Basic WER implementation
wer_hparams = load_hyperpyyaml("""
wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats
""")

# Usage
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
# dPOSER configuration
wer_stats_dposer: !new:speechbrain.utils.metric_stats.ErrorRateStats

# uPOSER configuration
uposer_dict: !apply:speechbrain.utils.dictionaries.SynonymDictionary.from_json_path
    path: ./uposer.json
wer_stats_uposer: !new:speechbrain.utils.metric_stats.ErrorRateStats
    equality_comparator: !ref <uposer_dict>

pos_tagger: !apply:speechbrain.lobes.models.flair.FlairSequenceTagger.from_hf
    source: "qanastek/pos-french"
    save_path: ./pretrained_models/
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
    save_path: ./pretrained_models/

ember_metric: !new:speechbrain.utils.metric_stats.EmbeddingErrorRateSimilarity
    embedding_function: !name:speechbrain.lobes.models.flair.embeddings.FlairEmbeddings.embed_word
        - !ref <ember_embeddings>
    low_similarity_weight: 1.0
    high_similarity_weight: 0.1
    threshold: 0.4
""")
```

### BERTScore
```python
bertscore_hparams = load_hyperpyyaml("""
bertscore_model_name: camembert/camembert-large
bertscore_model_device: cuda

bertscore_stats: !new:speechbrain.utils.bertscore.BERTScoreStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: !ref <bertscore_model_name>
        save_path: pretrained_models/
        device: !ref <bertscore_model_device>
        num_layers: 8
""")
```

### SemDist (Sentence Semantic Distance)
```python
semdist_hparams = load_hyperpyyaml("""
semdist_model_name: camembert/camembert-large
semdist_model_device: cuda

semdist_stats: !new:speechbrain.utils.semdist.SemDistStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: !ref <semdist_model_name>
        save_path: pretrained_models/
        device: !ref <semdist_model_device>
    method: meanpool
""")
```

## Important Notes

1. WER limitations: All errors are weighted equally regardless of semantic impact
2. CER operates at character level, providing finer granularity but slower computation
3. EmbER requires additional setup for embeddings and configuration
4. BERTScore compares cosine similarity of all targets and predicted embeddings
5. SemDist score is normalized by x1000 for readability (lower is better)

## Dependencies
```python
pip install speechbrain spacy flair
```

For detailed implementation examples and metric comparisons, refer to the original tutorial.