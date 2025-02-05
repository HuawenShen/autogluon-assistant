# Condensed: Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information:

# Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning

## Key Concepts
- Parameter-efficient finetuning enables training large foundation models on limited hardware
- Key techniques: BitFit, Prompt Tuning, LoRA, Adapter, MAM Adapter, and IA^3
- Combines efficient finetuning with gradient checkpointing for large model training

## Implementation

### 1. Setup and Data Preparation
```python
!pip install autogluon.multimodal

# Download and prepare dataset
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .

# Cache management
import os, shutil
os.environ["TRANSFORMERS_CACHE"] = "cache"
```

```python
# Load multilingual sentiment data
import pandas as pd
train_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_train.tsv",
                          sep="\t",
                          header=None,
                          names=["label", "text"]) \
                .sample(1000, random_state=123).reset_index(drop=True)

# Load test sets similarly for en, de, jp
```

### 2. IA3 + BitFit Finetuning
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(train_en_df,
              presets="multilingual",
              hyperparameters={
                  "optimization.efficient_finetune": "ia3_bias",  # Enable efficient finetuning
                  "optimization.lr_decay": 0.9,
                  "optimization.learning_rate": 3e-03,
                  "optimization.max_epochs": 2,
                  "env.batch_size": 32,
              })
```

### 3. FLAN-T5-XL Training on Single GPU
```python
predictor = MultiModalPredictor(label="label", path=new_model_path)
predictor.fit(train_en_df_downsample,
              presets="multilingual",
              hyperparameters={
                  "model.hf_text.checkpoint_name": "google/flan-t5-xl",
                  "model.hf_text.gradient_checkpointing": True,  # Enable gradient checkpointing
                  "model.hf_text.low_cpu_mem_usage": True,
                  "optimization.efficient_finetune": "ia3_bias",
                  "env.batch_size": 1,
                  "env.eval_batch_size_ratio": 1
              })
```

## Key Implementation Notes
1. Efficient finetuning reduces trainable parameters to ~0.5% of total parameters
2. Gradient checkpointing enables training billion-parameter models on limited GPU memory
3. Combined techniques allow training FLAN-T5-XL (2B parameters) on a single T4 GPU
4. Model maintains performance across languages despite training only on English data

## Important Parameters
- `optimization.efficient_finetune`: Set to "ia3_bias" for parameter-efficient training
- `model.hf_text.gradient_checkpointing`: Enable for memory efficiency
- `env.batch_size`: Adjust based on available GPU memory
- `optimization.learning_rate`: Critical for efficient finetuning performance

## Evaluation
```python
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)
score_in_jp = predictor.evaluate(test_jp_df)
```

For customization options, refer to the customization documentation.