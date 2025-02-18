# Condensed: AutoMM Detection - Quick Start on a Tiny COCO Format Dataset

Summary: This tutorial provides implementation guidance for object detection using AutoGluon's MultiModalPredictor, specifically covering YOLOX and DINO model configurations. It demonstrates how to set up, train, and perform inference with object detection models using different quality presets (medium, high, best). Key functionalities include model initialization, training with COCO-format data, inference with confidence thresholds, and result visualization. The tutorial helps with tasks like configuring GPU usage, handling model I/O, and managing different prediction output formats. It emphasizes critical dependencies (MMCV, MMDet) and version compatibility requirements, making it valuable for both prototyping and production deployment of object detection systems.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and practices:

# AutoMM Detection Quick Start Guide

## Key Setup Requirements

```bash
# Critical installations
pip install autogluon.multimodal
pip install -U pip setuptools wheel
# Install MMCV and dependencies
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"
```

**Important:** MMDet requires MMCV 2.1.0 and is CUDA-version sensitive. Best results with CUDA 12.4 + PyTorch 2.5.

## Implementation Details

### 1. Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
import os

# Initialize predictor
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",
    path=model_path
)
```

### 2. Model Configuration Options

- **medium_quality**: YOLOX-large (default, balanced speed/accuracy)
- **high_quality**: DINO-Resnet50 (better accuracy)
- **best_quality**: DINO-SwinL (highest accuracy, slower)

### 3. Training and Evaluation
```python
# Train model
predictor.fit(train_path)

# Evaluate
predictor.evaluate(test_path)
```

### 4. Inference
```python
# Predict with confidence threshold
pred = predictor.predict(test_path, save_results=True, as_coco=False)

# Visualization
from autogluon.multimodal.utils import ObjectDetectionVisualizer
visualizer = ObjectDetectionVisualizer(img_path)
out = visualizer.draw_instance_predictions(image_result, conf_threshold=0.4)
```

## Critical Configurations

1. **Data Format Requirements**:
   - COCO format JSON files
   - Required files: `trainval_cocoformat.json`, `test_cocoformat.json`

2. **Model Output Format**:
   ```python
   {
       "class": "class_name",
       "bbox": [x1, y1, x2, y2],  # Corner coordinates
       "score": confidence_score
   }
   ```

## Best Practices

1. **Model Selection**:
   - Use `medium_quality` for quick prototyping
   - Use `high_quality` or `best_quality` for production

2. **GPU Usage**:
   ```python
   predictor.set_num_gpus(1)  # Adjust based on availability
   ```

3. **Save/Load Models**:
   ```python
   # Load saved model
   new_predictor = MultiModalPredictor.load(model_path)
   ```

4. **Inference Options**:
   - Support for single images, image lists, or COCO format files
   - Can save predictions in CSV or COCO JSON formats

## Important Warnings

- MMDet compatibility issues with newer CUDA versions
- Restart kernel after MMCV installation
- Ensure proper CUDA/PyTorch version matching
- Monitor GPU memory usage with higher quality models