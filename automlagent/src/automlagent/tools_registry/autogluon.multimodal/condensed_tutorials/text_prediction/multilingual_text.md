# Condensed: AutoMM for Text - Multilingual Problems

Summary: This tutorial demonstrates implementing multilingual text classification using AutoGluon's MultiModalPredictor, covering both monolingual and cross-lingual approaches. It provides code for dataset handling, model configuration, and fine-tuning using language-specific BERT models (e.g., German BERT) and multilingual models (XLM-R/DeBERTa-V3). Key functionalities include zero-shot cross-lingual transfer without translation, support for 7,100+ languages, and flexible model selection between language-specific and multilingual presets. The tutorial helps with tasks like setting up multilingual datasets, configuring appropriate models, and implementing cross-lingual transfer learning, with specific focus on optimization parameters and best practices for model selection.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# AutoMM for Multilingual Text Classification

## Key Implementation Details

### 1. Dataset Setup
```python
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# Load multilingual data (German/English/Japanese Amazon reviews)
train_de_df = pd.read_csv('de_train.tsv', sep='\t', 
                         header=None, names=['label', 'text'])
test_de_df = pd.read_csv('de_test.tsv', sep='\t', 
                        header=None, names=['label', 'text'])
```

### 2. German BERT Finetuning
```python
# Finetune German BERT
predictor = MultiModalPredictor(label='label')
predictor.fit(train_de_df,
             hyperparameters={
                 'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                 'optimization.max_epochs': 2
             })
```

### 3. Cross-lingual Transfer
```python
# Enable multilingual support using XLM-R/DeBERTa-V3
predictor = MultiModalPredictor(label='label')
predictor.fit(train_en_df,
             presets='multilingual',
             hyperparameters={
                 'optimization.max_epochs': 2
             })
```

## Critical Configurations

1. **Monolingual Setup**:
   - Use specific language model (e.g., 'bert-base-german-cased')
   - Suitable for single language tasks

2. **Multilingual Setup**:
   - Enable with `presets='multilingual'`
   - Automatically uses advanced models like DeBERTa-V3
   - Supports zero-shot cross-lingual transfer

## Best Practices

1. **Model Selection**:
   - Use language-specific models for single language tasks
   - Use multilingual preset for cross-lingual transfer

2. **Cross-lingual Transfer**:
   - Train on source language (e.g., English)
   - Model can be applied directly to other languages
   - No translation needed for inference

## Important Notes

- Multilingual models support zero-shot transfer across languages
- Performance may vary across different languages
- Check [Single GPU Billion-scale Model Training](../advanced_topics/efficient_finetuning_basic.ipynb) for better performance
- For customization options, refer to [Customize AutoMM](../advanced_topics/customization.ipynb)

This implementation supports over 7,100 languages through multilingual models and demonstrates effective cross-lingual transfer without the need for translation services.