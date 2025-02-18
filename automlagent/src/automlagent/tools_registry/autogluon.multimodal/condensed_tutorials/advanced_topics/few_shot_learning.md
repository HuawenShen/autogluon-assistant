# Condensed: Few Shot Learning with AutoMM

Summary: This tutorial demonstrates implementing few-shot learning using AutoGluon's MultiModalPredictor for both text and image classification tasks. It provides code examples for setting up few-shot classifiers alongside standard classifiers for comparison, specifically designed for scenarios with limited training data per class. The tutorial covers essential configurations like setting problem_type="few_shot_classification", data formatting requirements for DataFrame inputs, and implementation best practices. Key features include foundation model + SVM architecture, support for both text and image modalities, and performance optimization for small datasets. This knowledge is particularly valuable for developing classification models with limited training samples.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Few Shot Learning with AutoMM

## Key Concepts
- Uses foundation model features + SVM for few shot classification
- Works for both text and image classification tasks
- Particularly effective for small datasets (few samples per class)

## Implementation Details

### Text Classification

```python
from autogluon.multimodal import MultiModalPredictor

# Few-shot classifier
predictor_fs = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="label",
    eval_metric="acc"
)
predictor_fs.fit(train_df)

# Standard classifier (for comparison)
predictor_default = MultiModalPredictor(
    problem_type="classification",
    label="label",
    eval_metric="acc"
)
predictor_default.fit(train_df)
```

### Image Classification

```python
# Few-shot classifier
predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",
    eval_metric="acc"
)
predictor_fs_image.fit(train_df)

# Standard classifier (for comparison)
predictor_default_image = MultiModalPredictor(
    problem_type="classification",
    label="LabelName",
    eval_metric="acc"
)
predictor_default_image.fit(train_df)
```

## Critical Configurations
- Must specify `problem_type="few_shot_classification"` for few-shot learning
- Input data should be in `pd.DataFrame` format
- For images: DataFrame should contain image paths and labels
- For text: DataFrame should contain text content and labels

## Best Practices
1. Use few-shot learning when you have very limited samples per class
2. Ensure data is properly formatted before training
3. Compare performance with standard classification approach
4. Use appropriate evaluation metrics (acc, f1_macro recommended)

## Important Notes
- Few-shot classification typically performs better than standard classification with limited data
- Works with both text and image modalities
- Built on foundation models + SVM architecture
- For customization options, refer to the AutoMM customization documentation