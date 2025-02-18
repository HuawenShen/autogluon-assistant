# Condensed: AutoMM Detection - Object detection data formats

Summary: This tutorial covers object detection data handling and model training in AutoGluon's MultiModal framework, specifically focusing on COCO and DataFrame format implementations. It provides code examples for format conversions between COCO JSON and DataFrame structures, along with essential data fields and their requirements. The tutorial demonstrates how to configure and train object detection models using MultiModalPredictor, including hyperparameter settings, GPU utilization, and learning rate optimization. Key functionalities include format conversion utilities, model training setup, and best practices for handling detection datasets, making it valuable for implementing object detection pipelines with proper data formatting and model configuration.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection Data Formats

## Supported Data Formats
1. COCO Format (.json)
2. DataFrame Format (pandas)

## COCO Format Implementation
```python
data = {
    "categories": [
        {"supercategory": "none", "id": 1, "name": "person"},
        # ... additional categories
    ],
    "images": [
        {
            "file_name": "<imagename>.<ext>",
            "height": 427,
            "width": 640,
            "id": 1
        },
        # ... additional images
    ],
    "annotations": [
        {
            'area': 33453,
            'iscrowd': 0,
            'bbox': [x, y, width, height],
            'category_id': 8,
            'ignore': 0,
            'segmentation': [],
            'image_id': 1617,
            'id': 1
        },
        # ... additional annotations
    ],
    "type": "instances"
}
```

## DataFrame Format Requirements
Required columns:
- `image`: Path to image file
- `rois`: List of arrays with format `[x1, y1, x2, y2, class_label]`
- `label`: Copy of `rois` column

## Key Implementation Steps

### 1. Format Conversion Utilities
```python
# COCO to DataFrame
from autogluon.multimodal.utils.object_detection import from_coco
train_df = from_coco(train_path)

# DataFrame to COCO
from autogluon.multimodal.utils.object_detection import object_detection_df_to_coco
train_coco = object_detection_df_to_coco(train_df, save_path="output.json")
```

### 2. Model Training Setup
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_320_300e_coco",
        "env.num_gpus": -1,  # use all GPUs
    },
    problem_type="object_detection",
    sample_data_path=train_df,
    path=model_path,
)
```

### 3. Training Configuration
```python
predictor.fit(
    train_df,
    hyperparameters={
        "optimization.learning_rate": 2e-4,
        "optimization.max_epochs": 30,
        "env.per_gpu_batch_size": 32,
    },
)
```

## Important Notes
- When loading from JSON, ensure correct root path for images
- Dependencies required: `mmcv` and `mmdet==3.1.0`
- Adjust batch size based on model size and available memory
- Two-stage detection head uses 100x learning rate

## Best Practices
1. Verify data format consistency before training
2. Use appropriate batch size for available GPU memory
3. Ensure correct image paths when converting between formats
4. Test with small dataset before full training