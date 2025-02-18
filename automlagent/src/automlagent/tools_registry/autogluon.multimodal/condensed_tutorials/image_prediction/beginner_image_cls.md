# Condensed: AutoMM for Image Classification - Quick Start

Summary: This tutorial demonstrates the implementation of AutoGluon's MultiModalPredictor for image classification tasks, covering essential techniques for model training, evaluation, and deployment. It shows how to load image data using both file paths and bytearrays, train models with customizable time limits, perform predictions, extract embeddings, and handle model persistence. Key functionalities include flexible data input formats, automatic model training, evaluation metrics calculation, probability predictions, and feature extraction. The tutorial is particularly useful for tasks involving quick setup of image classification pipelines, model evaluation, and production deployment with proper model saving/loading mechanisms.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Image Classification - Quick Start

## Key Implementation Details

### Setup and Data Loading
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset

# Load dataset (supports two formats)
# 1. With image paths
train_data_path, test_data_path = shopee_dataset(download_dir)

# 2. With bytearrays
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

### Model Training
```python
predictor = MultiModalPredictor(
    label="label",  # Column name containing target variable
    path="./model_path"  # Save directory for models
)

predictor.fit(
    train_data=train_data_path,
    time_limit=30  # Training time in seconds
)
```

### Evaluation and Prediction
```python
# Evaluate model
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])

# Make predictions
predictions = predictor.predict({'image': [image_path]})
probabilities = predictor.predict_proba({'image': [image_path]})

# Extract embeddings
features = predictor.extract_embedding({'image': [image_path]})
```

### Model Persistence
```python
# Save happens automatically after fit()
# Load saved model
loaded_predictor = MultiModalPredictor.load(model_path)
```

## Important Notes

1. Data Format:
   - Requires Pandas DataFrame with image paths or bytearrays
   - Must have columns for image data and labels

2. Input Flexibility:
   - Supports both image paths and bytearrays
   - Can mix formats between training and inference

3. Security Warning:
   - `MultiModalPredictor.load()` uses pickle
   - Only load models from trusted sources

4. Model Output Options:
   - Class predictions
   - Class probabilities
   - Feature embeddings (N-dimensional vectors)

## Best Practices

1. Adjust `time_limit` parameter based on dataset size and requirements
2. Use consistent data formats throughout the pipeline
3. Save model checkpoints in a dedicated directory
4. Validate model performance on test data before deployment

For advanced customization, refer to the AutoMM customization documentation.