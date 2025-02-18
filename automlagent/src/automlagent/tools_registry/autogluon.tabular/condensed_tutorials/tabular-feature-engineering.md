# Condensed: AutoGluon Tabular - Feature Engineering

Summary: This tutorial covers AutoGluon's tabular feature engineering implementation, focusing on automatic column type detection (boolean, numerical, categorical, datetime, text, image) and their processing rules. It demonstrates how to handle feature processing for different data types, particularly datetime (timestamp conversion, temporal features) and text (transformer networks, n-gram features) columns. The tutorial helps with tasks like custom feature pipeline creation, problem type specification, and missing value handling. Key functionalities include automatic type detection, custom feature generation using PipelineFeatureGenerator, MultiModal processing for text, and configuration of the AutoML pipeline through explicit parameters and feature generators.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and practices:

# AutoGluon Tabular - Feature Engineering

## Core Column Types
```python
- boolean: [A, B]
- numerical: [1.3, 2.0, -1.6]
- categorical: [Red, Blue, Yellow]
- datetime: [1/31/2021, Mar-31]
- text: [longer text strings]
- image: [path/image123.png] (with MultiModal option)
```

## Key Detection Rules
- **Boolean**: Columns with exactly 2 unique values
- **Categorical**: String columns not classified as text
- **Numerical**: Passed through unchanged (float/int)
- **Text**: Most rows unique + multiple words per row
- **Datetime**: Auto-detected via Pandas datetime conversion

## Important Implementation Details

### Problem Type Detection
```python
predictor = TabularPredictor(
    label='class', 
    problem_type='multiclass'  # Override auto-detection
).fit(train_data)
```

### Feature Processing

**Datetime Columns**:
- Converted to numerical timestamp
- Generated features: [year, month, day, dayofweek]
- Missing values → column mean

**Text Columns**:
1. MultiModal enabled: Uses Transformer neural network
2. Standard processing:
   - N-gram feature generation
   - Special features (word counts, char counts, etc.)

### Custom Feature Engineering
```python
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator

# Example custom pipeline
mypipeline = PipelineFeatureGenerator(
    generators = [[        
        CategoryFeatureGenerator(maximum_num_cat=10),
        IdentityFeatureGenerator(
            infer_features_in_args=dict(
                valid_raw_types=[R_INT, R_FLOAT]
            )
        ),
    ]]
)
```

## Best Practices

1. Explicitly mark categorical columns:
```python
df["column"] = df["column"].astype("category")
```

2. Handle missing values appropriately:
- Numeric/Categorical/Text: NaN preserved
- Datetime: Replaced with mean

3. Use problem_type parameter to override automatic inference when needed

4. Consider using MultiModal option for complex text processing

5. Customize feature generation pipeline for specific needs using PipelineFeatureGenerator

## Critical Configurations
```python
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
predictor.fit(
    df,
    hyperparameters={'GBM': {}},
    feature_generator=auto_ml_pipeline_feature_generator
)
```