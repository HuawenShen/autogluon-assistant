# Condensed: Image-to-Image Semantic Matching with AutoMM

Summary: This tutorial demonstrates implementing image-to-image semantic matching using AutoGluon's MultiModalPredictor. It covers techniques for setting up paired image comparison tasks, including data preparation with path handling, model configuration for similarity learning, and training implementation. The tutorial helps with tasks like binary classification of image pairs, generating similarity scores, and extracting image embeddings. Key features include Swin Transformer-based feature extraction, cosine similarity computation, and multiple prediction methods (binary predictions, probability scores, and embedding extraction), making it valuable for applications requiring image pair matching or similarity assessment.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Image-to-Image Semantic Matching with AutoMM

## Key Implementation Details

### Setup
```python
!pip install autogluon.multimodal
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

### Data Preparation
1. Dataset structure:
   - Two image columns (Image1, Image2)
   - Label column (1 = matching pair, 0 = non-matching pair)
   - Uses Stanford Online Products dataset with 12 product categories

2. Path handling:
```python
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply to image columns
for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### Model Training
```python
predictor = MultiModalPredictor(
    problem_type="image_similarity",
    query=image_col_1,          # first image column
    response=image_col_2,       # second image column
    label=label_col,           # label column
    match_label=match_label,   # label value indicating match (e.g., 1)
    eval_metric='auc'          # evaluation metric
)

predictor.fit(
    train_data=train_data,
    time_limit=180
)
```

### Key Features

1. Prediction Methods:
```python
# Binary predictions
predictions = predictor.predict(test_data)

# Probability scores
probabilities = predictor.predict_proba(test_data)

# Extract embeddings
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1]})
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2]})
```

2. Model Architecture:
- Uses Swin Transformer for image feature extraction
- Computes cosine similarity between image feature vectors

## Important Notes

1. Technical Details:
- Default probability threshold: 0.5
- Outputs embeddings as high-dimensional vectors
- Uses AUC as default evaluation metric

2. Best Practices:
- Ensure image paths are properly expanded
- Correctly specify match_label based on your dataset
- Consider adjusting time_limit based on dataset size

3. Data Requirements:
- Two image columns with corresponding paths
- Binary labels indicating match/non-match
- Consistent image format and accessibility