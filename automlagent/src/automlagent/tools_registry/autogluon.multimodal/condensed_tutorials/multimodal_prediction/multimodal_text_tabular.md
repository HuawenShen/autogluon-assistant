# Condensed: AutoMM for Text + Tabular - Quick Start

Summary: This tutorial demonstrates implementing AutoGluon's MultiModalPredictor for combined text and tabular data processing. It covers essential techniques for data preprocessing (including numeric conversions and log transformations), model configuration, training, and prediction workflows. The tutorial helps with tasks like mixed-data type handling, price prediction, and feature embedding extraction. Key functionalities include automatic data type inference, neural network architecture generation, multi-modal feature fusion, and evaluation methods. It's particularly useful for implementing machine learning systems that need to process both textual and structured data simultaneously while maintaining a simple API interface.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Text + Tabular - Quick Start

## Key Implementation Details

### Setup
```python
!pip install autogluon.multimodal
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

### Data Preprocessing
```python
def preprocess(df):
    df = df.copy(deep=True)
    # Convert Reviews and Ratings to numeric
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    # Log transform prices
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)
    return df
```

### Training Configuration
```python
from autogluon.multimodal import MultiModalPredictor
import uuid

# Critical parameters
time_limit = 3 * 60  # training time in seconds
model_path = f"./tmp/{uuid.uuid4().hex}-automm_text_book_price_prediction"

# Initialize and train predictor
predictor = MultiModalPredictor(label='Price', path=model_path)
predictor.fit(train_data, time_limit=time_limit)
```

### Prediction and Evaluation
```python
# Get predictions
predictions = predictor.predict(test_data)

# Evaluate model performance
performance = predictor.evaluate(test_data)

# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
```

## Important Notes

1. **Data Handling**:
   - Supports mixed data types (text, categorical, numerical)
   - Automatically infers data types and generates appropriate neural network architecture

2. **Model Architecture**:
   - Single neural network that jointly processes multiple feature types
   - Embeds text, categorical, and numeric fields separately
   - Fuses features across modalities

3. **Best Practices**:
   - Increase `time_limit` for better model performance in production
   - Use larger training samples when possible (demo uses subsample for speed)
   - Log-transform price values for better prediction accuracy

4. **Customization**:
   - Refer to "Customize AutoMM" documentation for advanced configurations
   - Additional examples available in AutoMM Examples repository

This implementation allows for seamless handling of mixed data types while maintaining simplicity in the API interface.