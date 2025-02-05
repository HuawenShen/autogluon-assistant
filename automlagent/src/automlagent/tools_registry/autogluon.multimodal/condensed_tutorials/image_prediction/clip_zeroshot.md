# Condensed: Zero-Shot Image Classification with CLIP

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the CLIP zero-shot classification tutorial:

# Zero-Shot Image Classification with CLIP

CLIP enables visual classification without training data by matching images with text descriptions. It achieves 76.2% top-1 accuracy on ImageNet without using training samples, comparable to supervised ResNet50.

## Implementation

First, install the required package:
```python
!pip install autogluon.multimodal
```

### Basic Usage
Import core dependencies:
```python
from autogluon.multimodal import MultiModalPredictor, download
from IPython.display import Image, display
```

#### Example 1: Dog Breed Classification
```python
# Load image
url = "https://farm4.staticflickr.com/3445/3262471985_ed886bf61a_z.jpg"
dog_image = download(url)

# Initialize predictor
predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")

# Predict with candidate labels
prob = predictor.predict_proba(
    {"image": [dog_image]}, 
    {"text": ['This is a Husky', 'This is a Golden Retriever', 
              'This is a German Sheperd', 'This is a Samoyed.']}
)
print("Label probs:", prob)
```

#### Example 2: Uncommon Object Classification
```python
# Classify Segway image
url = "https://live.staticflickr.com/7236/7114602897_9cf00b2820_b.jpg"
segway_image = download(url)

prob = predictor.predict_proba(
    {"image": [segway_image]}, 
    {"text": ['segway', 'bicycle', 'wheel', 'car']}
)
print("Label probs:", prob)
```

## Key Concepts
- CLIP uses contrastive learning on 400M image-text pairs
- Matches images with text descriptions without traditional training
- Suitable for arbitrary visual classification tasks
- Input format: Dictionary with image path and list of text candidates

## Important Limitations

### Typographic Attack Example
```python
# Original apple image classification
url = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
image_path = download(url)

prob = predictor.predict_proba(
    {"image": [image_path]}, 
    {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']}
)

# Same image with text overlay
url = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
image_path = download(url)

prob = predictor.predict_proba(
    {"image": [image_path]}, 
    {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']}
)
```

## Best Practices
1. Provide clear, descriptive text candidates
2. Consider potential text interference in images
3. Test with multiple candidate labels for better accuracy

For customization options, refer to the AutoMM documentation. Additional examples available in the [AutoMM Examples repository](https://github.com/autogluon/autogluon/tree/master/examples/automm).