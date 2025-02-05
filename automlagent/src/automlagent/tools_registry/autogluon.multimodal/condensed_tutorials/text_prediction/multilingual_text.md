# Condensed: AutoMM for Text - Multilingual Problems

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Text - Multilingual Problems

## Overview
This tutorial demonstrates how to use `MultiModalPredictor` for multilingual text classification using the Cross-Lingual Amazon Product Review Sentiment dataset (English, German, French, and Japanese reviews). We'll cover:
1. Finetuning German BERT
2. Cross-lingual transfer from English to other languages

## Dataset Preparation
```python
!pip install autogluon.multimodal
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load German data
train_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)
test_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)

# Load English data
train_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)
test_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
```

## Approach 1: German BERT Finetuning
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label')
predictor.fit(train_de_df,
              hyperparameters={
                  'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                  'optimization.max_epochs': 2
              })

# Evaluate
score = predictor.evaluate(test_de_df)
print('Score on the German Testset:', score)
```

## Approach 2: Cross-lingual Transfer
Using multilingual models for zero-shot transfer from English to other languages:

```python
predictor = MultiModalPredictor(label='label')
predictor.fit(train_en_df,
              presets='multilingual',
              hyperparameters={
                  'optimization.max_epochs': 2
              })

# Evaluate on multiple languages
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)

# Test Japanese performance
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
score_in_jp = predictor.evaluate(test_jp_df)
```

## Key Points
- Use `presets="multilingual"` for zero-shot cross-lingual transfer
- The multilingual model works across languages without additional training
- AutoGluon automatically uses state-of-the-art models (e.g., DeBERTa-V3)
- For better performance, consider parameter-efficient finetuning techniques

## Additional Resources
- For customization options: See "Customize AutoMM" tutorial
- More examples: Check AutoMM Examples in the GitHub repository
- For improved performance: Review "Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning"