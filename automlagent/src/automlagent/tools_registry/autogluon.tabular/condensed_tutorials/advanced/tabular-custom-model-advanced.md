# Condensed: Adding a custom model to AutoGluon (Advanced)

Summary: This tutorial demonstrates advanced integration of custom models in AutoGluon, focusing on three key implementation techniques: preventing feature dropping during preprocessing through custom model class implementation, creating custom feature generators with selective preprocessing control, and managing feature metadata for preprocessing behavior. It helps with tasks like preserving specific features during model training, implementing custom preprocessing pipelines, and controlling feature transformation at both model-specific and global levels. Key functionalities covered include custom model class extension, feature generator implementation with user overrides, feature metadata configuration, and integration with TabularPredictor, along with proper serialization practices.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial on adding custom models to AutoGluon, focusing on key implementation details:

# Advanced Custom Model Integration in AutoGluon

## Key Concepts
- Demonstrates how to prevent feature dropping during preprocessing
- Shows implementation of custom feature generators
- Explains model-specific and global preprocessing controls

## Implementation Details

### 1. Preventing Feature Dropping in Model-Specific Preprocessing

```python
from autogluon.core.models import AbstractModel

class CustomModel(AbstractModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # Prevents dropping features with only 1 unique value
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

### 2. Custom Feature Generator Implementation

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
            AutoMLPipelineFeatureGenerator(
                banned_feature_special_types=['user_override'], 
                **automl_generator_kwargs
            ),
            IdentityFeatureGenerator(
                infer_features_in_args=dict(
                    required_special_types=['user_override']
                )
            ),
        ]]
        return generators
```

### 3. Feature Metadata Configuration

```python
from autogluon.tabular import FeatureMetadata

feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({
    'feature1': ['user_override'],
    'feature2': ['user_override']
})
```

### 4. Usage with TabularPredictor

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,
    feature_generator=CustomFeatureGeneratorWithUserOverride(),
    hyperparameters={
        'CustomModel': {'ag_args_fit': {'drop_unique': False}}
    }
)
```

## Important Notes
- Custom model and feature generator code must be in separate Python files for serialization
- Use `user_override` special type to prevent feature preprocessing
- The feature generator can be customized for different preprocessing logic
- Features can be preserved either through model-specific parameters or global feature generator settings

## Best Practices
1. Always test feature preservation with a small dataset first
2. Verify feature columns after preprocessing
3. Keep custom preprocessing logic in separate files
4. Use feature metadata to control preprocessing behavior