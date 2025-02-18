# Condensed: AutoMM for Scanned Document Classification

Summary: This tutorial demonstrates implementing document classification using AutoGluon's MultiModalPredictor, specifically focusing on scanned documents. It covers essential techniques for loading and processing document datasets, training models using pre-trained document transformers (like LayoutLM), and performing predictions and embedding extraction. The tutorial helps with tasks including document classification, feature extraction, and probability prediction. Key functionalities include automatic text recognition, utilization of layout and visual features, support for various document/text models (LayoutLMv2/v3, BERT, DeBERTa), and customizable model training through hyperparameters, all implemented with minimal code requirements.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Scanned Document Classification

## Key Implementation Details

### Setup and Data Preparation
```python
# Install required package
!pip install autogluon.multimodal

# Load and prepare data
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal.utils.misc import path_expander

# Download dataset
download_dir = './ag_automm_tutorial_doc_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Split data
train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
test_data = rvl_cdip_data.drop(train_data.index)

# Expand document paths
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(
    lambda ele: path_expander(ele, base_folder=download_dir)
)
```

### Model Training and Prediction
```python
from autogluon.multimodal import MultiModalPredictor

# Initialize and train predictor
predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
        "optimization.top_k_average_method": "best",
    },
    time_limit=120,
)

# Evaluate
scores = predictor.evaluate(test_data, metrics=["accuracy"])

# Make predictions
predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
probabilities = predictor.predict_proba({DOC_PATH_COL: [doc_path]})

# Extract embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})
```

## Important Technical Notes

1. **Supported Models**:
   - Document models: layoutlmv3, layoutlmv2, layoutlm-base, layoutxlm
   - Text models: bert, deberta

2. **Key Features**:
   - Automatic recognition of handwritten/typed text
   - Utilizes text, layout information, and visual features
   - Supports embedding extraction for document representation

3. **Best Practices**:
   - Use `path_expander` for correct document path handling
   - Specify model checkpoint name in hyperparameters for customization
   - Set appropriate time_limit based on dataset size and requirements

4. **Model Customization**:
   - Configurable through hyperparameters
   - Supports various document and text foundation models
   - Allows for metric selection during evaluation

This implementation enables automated document classification with minimal code while maintaining flexibility for customization.