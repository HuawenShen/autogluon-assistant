# Condensed: AutoGluon Tabular - Feature Engineering

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# AutoGluon Tabular - Feature Engineering

## Core Concepts

### Column Types
AutoGluon supports these primary feature types:
- boolean: Binary values
- numerical: Continuous or discrete numbers
- categorical: Discrete categories
- datetime: Date and time values
- text: Free-form text
- image (with MultiModal option): Image file paths

### Automatic Type Detection
- **Boolean**: Columns with exactly 2 unique values
- **Categorical**: String columns that don't qualify as text
- **Numerical**: Passed through as float or int
- **Text**: Mostly unique rows with multiple words per row
- **Datetime**: Detected via Pandas datetime conversion

### Problem Type Detection
AutoGluon automatically determines between:
- Regression: Float values, many unique values
- Classification: Binary or multiclass

Override with:
```python
predictor = TabularPredictor(label='class', problem_type='multiclass').fit(train_data)
```

## Feature Engineering Implementation

### Automatic Processing by Type

**Numerical**
- No automatic transformations

**Categorical**
- Mapped to monotonically increasing integers

**Datetime**
- Converted to numerical timestamp
- Generated features: year, month, day, dayofweek
- Missing values replaced with mean

**Text** (Standard Processing)
- N-gram feature generation
- Special features: word counts, character counts, etc.

### Core Example

```python
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
from datetime import datetime

# Create sample data
x, y = make_regression(n_samples=100, n_features=5, n_targets=1)
dfx = pd.DataFrame(x, columns=['A','B','C','D','E'])

# Add different column types
dfx['B'] = dfx['B'].astype(int)
dfx['C'] = datetime(2000,1,1) + pd.to_timedelta(dfx['C'].astype(int), unit='D')
dfx['D'] = pd.cut(dfx['D'] * 10, [-np.inf,-5,0,5,np.inf], labels=['v','w','x','y'])

# Process with default feature generator
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
generator = AutoMLPipelineFeatureGenerator()
transformed_data = generator.fit_transform(X=dfx)
```

### Missing Value Handling
- Numeric/Categorical/Text: NaN values preserved
- Datetime: NaN replaced with mean

### Custom Feature Engineering
```python
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
from autogluon.common.features.types import R_INT, R_FLOAT

# Custom pipeline example
custom_pipeline = PipelineFeatureGenerator(
    generators = [[        
        CategoryFeatureGenerator(maximum_num_cat=10),  # Limit categorical values
        IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    ]]
)
transformed = custom_pipeline.fit_transform(X=dfx)
```

## Important Notes
- Columns with single value are dropped
- Duplicate columns are removed
- For categorical features, use `.astype("category")` to force categorical treatment
- DateTime features have min/max limits from Pandas
- MultiModal option enables advanced text processing with Transformers

This condensed version maintains all critical implementation details while removing redundant explanations and focusing on practical usage.