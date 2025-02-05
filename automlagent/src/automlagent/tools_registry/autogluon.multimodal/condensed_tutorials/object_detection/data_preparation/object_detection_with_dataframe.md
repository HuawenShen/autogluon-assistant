# Condensed: AutoMM Detection - Object detection data formats

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Object Detection Data Formats

## Supported Data Formats

AutoMM Detection supports two data formats:
1. COCO Format (JSON)
2. DataFrame Format

## COCO Format

Essential structure:
```python
data = {
    "categories": [
        {"supercategory": "none", "id": 1, "name": "person"},
        # ... other categories
    ],
    "images": [
        {
            "file_name": "<imagename0>.<ext>",
            "height": 427,
            "width": 640,
            "id": 1
        },
        # ... other images
    ],
    "annotations": [
        {
            'area': 33453,
            'iscrowd': 0,
            'bbox': [181, 133, 177, 189],  # [x, y, width, height]
            'category_id': 8,
            'image_id': 1617,
            'id': 1
        },
        # ... other annotations
    ],
    "type": "instances"
}
```

## DataFrame Format

Required columns:
- `image`: Path to image file
- `rois`: List of arrays with format `[x1, y1, x2, y2, class_label]`
- `label`: Copy of `rois` column

## Implementation

### Setup and Data Preparation
```python
import os
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.object_detection import from_coco, object_detection_df_to_coco

# Download sample dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"
load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

### Format Conversion
```python
# COCO to DataFrame
train_df = from_coco(train_path)

# DataFrame to COCO
train_coco = object_detection_df_to_coco(train_df, save_path="./df_converted_to_coco.json")

# Load saved COCO format
train_df_from_saved_coco = from_coco("./df_converted_to_coco.json", root="./")
```

### Model Training
```python
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs

predictor_df = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_df,
    path=f"./tmp/{uuid.uuid4().hex}-df_train_temp_save"
)

predictor_df.fit(
    train_df,
    hyperparameters={
        "optimization.learning_rate": 2e-4,
        "optimization.max_epochs": 30,
        "env.per_gpu_batch_size": 32,
    },
)
```

### Evaluation
```python
test_df = from_coco(test_path)
predictor_df.evaluate(test_df)
```

## Key Points
- Install required dependencies: `mmcv` and `mmdet==3.1.0`
- Ensure correct image paths when converting between formats
- Adjust batch size based on model size and available memory
- Learning rate uses two-stage training with detection head having 100x lr

For customization options, refer to the AutoMM customization documentation.