# Condensed: AutoMM Problem Types And Metrics

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM Problem Types And Metrics

This tutorial covers AutoGluon Multimodal's supported problem types, their modalities, and evaluation metrics.

```python
!pip install autogluon.multimodal
import warnings
warnings.filterwarnings('ignore')

# Helper function for displaying problem type information
from autogluon.multimodal.constants import *
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG

def print_problem_type_info(name: str, props):
    print(f"\n=== {name} ===")
    print("\nSupported Input Modalities:")
    for modality in sorted(list(props.supported_modality_type)):
        print(f"- {modality}")
    if hasattr(props, 'supported_evaluation_metrics') and props.supported_evaluation_metrics:
        print("\nEvaluation Metrics:")
        for metric in sorted(list(props.supported_evaluation_metrics)):
            if metric == props.fallback_evaluation_metric:
                print(f"- {metric} (default)")
            else:
                print(f"- {metric}")
    if hasattr(props, 'support_zero_shot'):
        print("\nSpecial Capabilities:")
        print(f"- Zero-shot prediction: {'Supported' if props.support_zero_shot else 'Not supported'}")
        print(f"- Training support: {'Supported' if props.support_fit else 'Not supported'}")
```

## Core Problem Types

### 1. Classification
Supports binary (2 classes) and multiclass (3+ classes) classification:

```python
binary_props = PROBLEM_TYPES_REG.get(BINARY)
multiclass_props = PROBLEM_TYPES_REG.get(MULTICLASS)
print_problem_type_info("Binary Classification", binary_props)
print_problem_type_info("Multiclass Classification", multiclass_props)
```

### 2. Regression
Supports numerical value prediction:

```python
regression_props = PROBLEM_TYPES_REG.get(REGRESSION)
print_problem_type_info("Regression", regression_props)
```

### 3. Computer Vision Tasks
- Object Detection: Identifies and localizes objects with bounding boxes
- Semantic Segmentation: Performs pixel-level classification

```python
object_detection_props = PROBLEM_TYPES_REG.get(OBJECT_DETECTION)
print_problem_type_info("Object Detection", object_detection_props)

segmentation_props = PROBLEM_TYPES_REG.get(SEMANTIC_SEGMENTATION)
print_problem_type_info("Semantic Segmentation", segmentation_props)
```

### 4. Similarity Matching
Supports three types of similarity matching:
- Text-to-Text
- Image-to-Image
- Image-to-Text

```python
similarity_types = [
    (TEXT_SIMILARITY, "Text Similarity"),
    (IMAGE_SIMILARITY, "Image Similarity"),
    (IMAGE_TEXT_SIMILARITY, "Image-Text Similarity")
]

print("\n=== Similarity Matching ===")
for type_key, type_name in similarity_types:
    props = PROBLEM_TYPES_REG.get(type_key)
    print(f"\n{type_name}:")
    print("Input Requirements:")
    for modality in props.supported_modality_type:
        print(f"- {modality}")
    print(f"Zero-shot prediction: {'Supported' if props.support_zero_shot else 'Not supported'}")
```

### 5. Specialized Tasks

#### Named Entity Recognition (NER)
Identifies and classifies named entities in text:

```python
ner_props = PROBLEM_TYPES_REG.get(NER)
print_problem_type_info("Named Entity Recognition", ner_props)
```

#### Feature Extraction
Converts raw data into feature vectors:

```python
feature_extraction_props = PROBLEM_TYPES_REG.get(FEATURE_EXTRACTION)
print_problem_type_info("Feature Extraction", feature_extraction_props)
```

#### Few-shot Classification
Learns to classify from limited examples per class:

```python
few_shot_props = PROBLEM_TYPES_REG.get(FEW_SHOT_CLASSIFICATION)
print_problem_type_info("Few-shot Classification", few_shot_props)
```

## Additional Resources
- More examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- Customization guide: [Customize AutoMM](../advanced_topics/customization.ipynb)