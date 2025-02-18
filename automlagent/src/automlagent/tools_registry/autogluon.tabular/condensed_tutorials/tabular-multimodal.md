# Condensed: Multimodal Data Tables: Tabular, Text, and Image

Summary: This tutorial demonstrates how to implement multimodal machine learning using AutoGluon, specifically combining tabular, text, and image data for prediction tasks. It provides code for handling multiple image paths, configuring feature metadata for different data types, and setting up appropriate hyperparameters for multimodal models. The tutorial covers essential implementation techniques including dataset preparation, image preprocessing, feature metadata configuration, and model training using TabularPredictor. Key functionalities include handling mixed data types, automatic text column identification, image path processing, and integration of various models (tabular models, Electra BERT for text, and ResNet for images) within a unified training pipeline.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Multimodal Data Tables: Tabular, Text, and Image Training

## Key Requirements
- GPU required for image and text models
- Appropriate CUDA versions for Torch installations

## Implementation Steps

### 1. Dataset Preparation
```python
# Download and unzip dataset
download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'

from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
```

### 2. Image Column Preprocessing
```python
# Handle multiple images (keep only first image)
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Update image paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### 3. Feature Metadata Configuration
```python
from autogluon.tabular import FeatureMetadata
# Create and customize feature metadata
feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
```

### 4. Hyperparameter Configuration
```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

### 5. Model Training
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,
)
```

## Important Best Practices

1. **Data Sampling**: For prototyping:
   - Start with smaller samples (e.g., 500 rows)
   - Gradually increase data size and time limits
   - Identify effective models before full-scale training

2. **Image Processing**:
   - AutoGluon supports only one image per row
   - Ensure correct image paths are provided
   - Images must be accessible from the specified paths

3. **Feature Metadata**:
   - Text columns are automatically identified
   - Image columns must be explicitly specified using 'image_path' special type
   - Use FeatureMetadata for custom column type specifications

4. **Model Configuration**:
   - Use 'multimodal' preset for combined tabular, text, and image processing
   - Includes:
     - Various tabular models
     - Electra BERT text model
     - ResNet image model

## Performance Evaluation
```python
leaderboard = predictor.leaderboard(test_data)
```

This condensed version maintains all critical implementation details while removing unnecessary explanations and verbose examples.