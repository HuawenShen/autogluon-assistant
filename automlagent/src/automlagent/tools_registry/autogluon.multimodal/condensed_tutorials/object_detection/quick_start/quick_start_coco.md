# Condensed: AutoMM Detection - Quick Start on a Tiny COCO Format Dataset

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM Detection - Quick Start on COCO Format Dataset

## Setup and Installation

```python
!pip install autogluon.multimodal
!mim install "mmcv==2.1.0"
!pip install "mmdet==3.2.0"

from autogluon.multimodal import MultiModalPredictor
import os
import time
from autogluon.core.utils.loaders import load_zip
```

## Data Preparation

Download and extract sample dataset:
```python
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"
load_zip.unzip(zip_file, unzip_dir=download_dir)

data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

## Model Configuration and Training

Initialize predictor with medium quality preset (uses YOLOX-large model):
```python
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-quick_start_tutorial_temp_save"

predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",
    path=model_path,
)
```

Train the model:
```python
start = time.time()
predictor.fit(train_path)
train_end = time.time()
print("Training time: %.2f seconds" % (train_end - start))
```

Key Features:
- Uses two-stage learning rate during finetuning
- Head layers use 100x higher learning rate
- Optimized for small datasets (hundreds/thousands of images)

## Evaluation

```python
predictor.evaluate(test_path)
```

Load saved model:
```python
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)  # Optionally adjust GPU usage
```

## Inference

Make predictions:
```python
# Predict and return results
pred = predictor.predict(test_path)

# Predict and save results to file
pred = predictor.predict(test_path, save_results=True)
```

## Visualization

```python
!pip install opencv-python

from autogluon.multimodal.utils import ObjectDetectionVisualizer

conf_threshold = 0.4
image_result = pred.iloc[30]
img_path = image_result.image

visualizer = ObjectDetectionVisualizer(img_path)
out = visualizer.draw_instance_predictions(image_result, conf_threshold=conf_threshold)
visualized = out.get_image()
```

## Custom Data Inference

```python
# Download test image
from autogluon.multimodal import download
image_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
test_image = download(image_url)

# Predict using COCO format
import json
data = {"images": [{"id": 0, "width": -1, "height": -1, "file_name": test_image}], "categories": []}
os.mkdir("input_data_for_demo")
input_file = "input_data_for_demo/demo_annotation.json"
with open(input_file, "w+") as f:
    json.dump(data, f)

pred_test_image = predictor.predict(input_file)

# Predict using image file list
pred_test_image = predictor.predict([test_image])
```

Important Notes:
- For better performance, use "high_quality" or "best_quality" presets
- See customization documentation for advanced configurations
- Output format: DataFrame with 'image' and 'bboxes' columns
- Bboxes format: `{"class": class_name, "bbox": [x1, y1, x2, y2], "score": confidence}`