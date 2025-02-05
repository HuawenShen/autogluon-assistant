# Condensed: Multimodal Data Tables: Tabular, Text, and Image

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Multimodal Data Tables: Tabular, Text, and Image

**Prerequisites:**
- GPU required for image/text models
- Basic understanding of TabularPredictor API
- Appropriate CUDA-enabled Torch installation

## Dataset Overview
Using PetFinder dataset to predict animal adoption rates using:
- Images of animals
- Text descriptions
- Tabular features (age, breed, color, etc.)

## Implementation

### 1. Setup and Data Loading
```python
!pip install autogluon

# Download and extract dataset
download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'

from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
dataset_path = download_dir + '/petfinder_processed'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)

label = 'AdoptionSpeed'
image_col = 'Images'
```

### 2. Image Column Preprocessing
```python
# Select first image only (AutoGluon limitation)
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

### 4. Model Configuration and Training
```python
# Get multimodal configuration
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')

# Train predictor
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,
)

# Evaluate models
leaderboard = predictor.leaderboard(test_data)
```

## Key Points
1. **Data Requirements**:
   - One image per row supported
   - Images must be accessible via file paths
   - Text columns automatically detected

2. **Best Practices**:
   - Start with small data samples for prototyping
   - Gradually increase data size and time limits
   - Use 'multimodal' preset for balanced performance

3. **Model Components**:
   - Tabular models
   - Electra BERT for text
   - ResNet for images

This implementation creates a multi-modal ensemble capable of processing tabular, text, and image data simultaneously using AutoGluon.