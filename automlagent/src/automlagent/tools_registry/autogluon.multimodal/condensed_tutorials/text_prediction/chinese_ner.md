# Condensed: AutoMM for Named Entity Recognition in Chinese - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Chinese Named Entity Recognition - Quick Start

## Overview
This tutorial demonstrates Chinese Named Entity Recognition using AutoMM on a TaoBao.com e-commerce dataset. The model identifies entities like brands (HPPX), products (HCCX), patterns (XH), and miscellaneous information (MISC) in product descriptions.

## Setup and Data Loading

```python
!pip install autogluon.multimodal

import autogluon.multimodal
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal.utils import visualize_ner

# Load preprocessed datasets
train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_train.csv')
dev_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_dev.csv')
```

## Model Training
Key points:
- Uses `'hfl/chinese-lert-small'` as the backbone model for Chinese text
- Sets problem type to "ner" for named entity recognition

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"

predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'hfl/chinese-lert-small'},
    time_limit=300, # 5 minutes training limit
)
```

## Evaluation and Prediction

```python
# Evaluate model performance
predictor.evaluate(dev_data)

# Make predictions
output = predictor.predict(dev_data)
visualize_ner(dev_data["text_snippet"].iloc[0], output[0])

# Example with custom text
sentence = "2023年兔年挂件新年装饰品小挂饰乔迁之喜门挂小兔子"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

## Important Notes
- The model automatically handles Chinese text processing
- Visualization functions help inspect entity recognition results
- Training time can be adjusted via the `time_limit` parameter
- Model path should be specified for saving and loading the model

For customization options, refer to the AutoMM customization documentation. Additional examples are available in the [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) repository.