# Condensed: AutoMM for Text + Tabular - Quick Start

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM for Text + Tabular - Quick Start

This tutorial demonstrates how to use AutoGluon's `MultiModalPredictor` to train a neural network that jointly processes multiple feature types (text, categorical, and numerical data).

## Setup

```python
!pip install autogluon.multimodal
!python3 -m pip install openpyxl

import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Data Preparation

We'll use the book price prediction dataset to predict book prices based on features like author, abstract, and ratings.

```python
# Download and load data
!mkdir -p price_of_books
!wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
!cd price_of_books && unzip -o Data.zip

train_df = pd.read_excel(os.path.join('price_of_books', 'Participants_Data', 'Data_Train.xlsx'), engine='openpyxl')
```

### Data Preprocessing
Convert Reviews and Ratings to numeric values and transform prices to log-scale:

```python
def preprocess(df):
    df = df.copy(deep=True)
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)
    return df

# Create train/test splits
train_subsample_size = 1500  # Adjust for larger datasets
test_subsample_size = 5
train_df = preprocess(train_df)
train_data = train_df.iloc[100:].sample(train_subsample_size, random_state=123)
test_data = train_df.iloc[:100].sample(test_subsample_size, random_state=245)
```

## Model Training

The `MultiModalPredictor` automatically handles different feature types:

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

time_limit = 3 * 60  # Increase for better performance
model_path = f"./tmp/{uuid.uuid4().hex}-automm_text_book_price_prediction"
predictor = MultiModalPredictor(label='Price', path=model_path)
predictor.fit(train_data, time_limit=time_limit)
```

## Prediction and Evaluation

```python
# Make predictions
predictions = predictor.predict(test_data)
print('Predictions:', np.exp(predictions) - 1)
print('True Value:', np.exp(test_data['Price']) - 1)

# Evaluate model
performance = predictor.evaluate(test_data)
print(performance)

# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
```

## Key Notes
- The model automatically handles multiple data types (text, categorical, numerical)
- Increase `time_limit` for better model performance
- Use `extract_embedding()` to get feature representations
- For customization options, refer to the AutoMM documentation

For more examples, visit [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm).