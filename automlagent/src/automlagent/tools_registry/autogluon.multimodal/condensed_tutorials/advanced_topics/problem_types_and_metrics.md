# Condensed: AutoMM Problem Types And Metrics

Summary: This tutorial covers AutoMM's implementation of seven core machine learning problem types: classification (binary/multiclass), regression, computer vision (object detection/segmentation), similarity matching, NER, feature extraction, and few-shot classification. It helps with tasks requiring multi-modal input handling and metric selection, detailing supported input types (text, image, numerical, categorical) and relevant evaluation metrics for each problem type. Key features include zero-shot capabilities for specific tasks (similarity matching, feature extraction), default metric configurations, and comprehensive modality support across problem types, with special focus on vision-specific tasks and multi-modal applications.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# AutoMM Problem Types and Metrics Guide

## Core Problem Types

### 1. Classification
- **Binary Classification**
  - Input: categorical, numerical, text, image (including base64/bytearray)
  - Default metric: roc_auc
  - Key metrics: accuracy, f1, precision, recall, log_loss

- **Multiclass Classification**
  - Same input types as binary
  - Default metric: accuracy
  - Additional metrics: roc_auc_ovo, roc_auc_ovr variants

### 2. Regression
- Supports all standard modalities
- Default metric: rmse
- Key metrics: mae, mse, r2, pearsonr, spearmanr

### 3. Computer Vision Tasks
**Object Detection**
```python
# Supported modalities
- image only

# Key metrics
- map (default)
- map_50, map_75
- map_large/medium/small
```

**Semantic Segmentation**
```python
# Supported modalities
- image only

# Key metrics
- iou (default)
- ber
- sm
```

### 4. Similarity Matching
```python
# Three types:
1. Text-to-Text
2. Image-to-Image 
3. Image-to-Text

# Features
- All support zero-shot prediction
- No training required
```

### 5. NER (Named Entity Recognition)
```python
# Supported modalities
- text_ner
- text
- categorical
- numerical
- image

# Key metrics
- overall_f1 (default)
- ner_token_f1
```

### 6. Feature Extraction
- Supports image and text
- Zero-shot capable
- No training required

### 7. Few-shot Classification
- Supports image and text
- Same metrics as multiclass classification
- Requires training

## Important Notes

1. **Zero-shot Capabilities**:
   - Supported: Object Detection, Segmentation, Similarity Matching, Feature Extraction
   - Not supported: Classification, Regression, NER, Few-shot Classification

2. **Default Configurations**:
   - Each problem type has predefined default metrics
   - Multiple evaluation metrics available for most tasks

3. **Modality Support**:
   - Most tasks support multiple input types
   - Vision tasks are image-only
   - Similarity matching requires specific modality pairs

For implementation examples, refer to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm).