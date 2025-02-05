# Condensed: Adding a custom model to AutoGluon (Advanced)

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial maintaining essential information and implementation details:

# Adding a Custom Model to AutoGluon (Advanced)

## Prerequisites
- Familiarity with basic AutoGluon usage
- Understanding of the basic custom model tutorial

## Key Concepts
This tutorial covers advanced techniques for customizing model behavior in AutoGluon, specifically:
- Preventing feature dropping during preprocessing
- Implementing custom feature generators
- Handling special feature types

## Implementation

### 1. Basic Setup
```python
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)
```

### 2. Custom Model Implementation
```python
from autogluon.core.models import AbstractModel

class DummyModelKeepUnique(AbstractModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # Prevents dropping features with only 1 unique value
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

### 3. Custom Feature Generator
```python
from autogluon.features import BulkFeatureGenerator, AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator

class CustomFeatureGeneratorWithUserOverride(BulkFeatureGenerator):
    def __init__(self, automl_generator_kwargs: dict = None, **kwargs):
        generators = self._get_default_generators(automl_generator_kwargs=automl_generator_kwargs)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, automl_generator_kwargs: dict = None):
        if automl_generator_kwargs is None:
            automl_generator_kwargs = dict()
        
        generators = [[
            AutoMLPipelineFeatureGenerator(banned_feature_special_types=['user_override'], 
                                         **automl_generator_kwargs),
            IdentityFeatureGenerator(infer_features_in_args=dict(
                required_special_types=['user_override'])),
        ]]
        return generators
```

### 4. Feature Metadata Configuration
```python
from autogluon.tabular import FeatureMetadata

feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({
    'age': ['user_override'],
    'native-country': ['user_override'],
    'dummy_feature': ['user_override'],
})
```

### 5. Usage with TabularPredictor
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,
    feature_generator=CustomFeatureGeneratorWithUserOverride(),
    hyperparameters={
        'GBM': {},
        DummyModelKeepUnique: {},
    }
)
```

## Important Notes
- Custom models and feature generators must be defined in separate Python files for serialization
- The `user_override` special type prevents features from being preprocessed
- Use `drop_unique=False` to prevent dropping features with single unique values
- Custom feature generators can implement arbitrary preprocessing logic for different feature types

## Best Practices
1. Always test feature preservation by checking preprocessed data columns
2. Use feature metadata to control preprocessing behavior
3. Implement custom feature generators when needing different preprocessing for different features
4. Consider serialization requirements when implementing custom components