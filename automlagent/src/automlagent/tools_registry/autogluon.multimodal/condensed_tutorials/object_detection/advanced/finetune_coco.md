# Condensed: AutoMM Detection - Finetune on COCO Format Dataset with Customized Settings

Summary: This tutorial demonstrates implementing object detection using AutoGluon's MultiModalPredictor with COCO-format datasets. It covers essential techniques for model setup, training, and evaluation using YOLOX architectures, specifically focusing on finetuning pretrained models. Key functionalities include configuring two-stage learning rates, batch size optimization, and early stopping strategies. The tutorial helps with tasks like setting up the detection pipeline, customizing training parameters, and visualizing predictions. Notable features include preset configurations for different quality requirements, GPU utilization settings, and performance optimization techniques. It's particularly useful for implementing efficient object detection systems with minimal code while maintaining flexibility for advanced customization.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - COCO Format Dataset Finetuning

## Key Setup Requirements

```bash
# Required installations
pip install autogluon.multimodal
pip install -U pip setuptools wheel
sudo apt-get install -y ninja-build gcc g++
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"
```

⚠️ **Important**: MMDet requires MMCV 2.1.0 and works best with CUDA 12.4 + PyTorch 2.5

## Implementation Steps

1. **Data Preparation**
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_zip

# Download and setup dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
download_dir = "./pothole"
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

2. **Model Configuration**
```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "yolox_s",  # Using YOLOX-small
        "env.num_gpus": 1,
    },
    problem_type="object_detection",
    sample_data_path=train_path,
)
```

3. **Training Configuration**
```python
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optimization.learning_rate": 1e-4,  # Head layers get 100x this rate
        "env.per_gpu_batch_size": 16,
        "optimization.max_epochs": 30,
        "optimization.val_check_interval": 1.0,
        "optimization.check_val_every_n_epoch": 3,
        "optimization.patience": 3,
    },
)
```

## Quick Implementation Using Presets

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",  # Options: medium_quality, high_quality, best_quality
)
predictor.fit(train_path, tuning_data=val_path)
```

## Evaluation and Visualization

```python
# Evaluate model
results = predictor.evaluate(test_path)

# Visualize predictions
from autogluon.multimodal.utils import visualize_detection
pred = predictor.predict(test_path)
visualized = visualize_detection(
    pred=pred[12:13],
    detection_classes=predictor.classes,
    conf_threshold=0.25,
    visualization_result_dir="./"
)
```

## Best Practices

1. Use two-stage learning rate (default) for faster convergence
2. Adjust batch size based on GPU memory
3. Consider using larger models (via `checkpoint_name`) for better performance
4. Use predefined presets for optimal configurations
5. Monitor validation metrics for early stopping

## Performance Notes
- Fast finetuning achieves decent results in hundreds of seconds
- For higher performance, consider longer training times with larger models
- Default configuration uses YOLOX-small for balance of speed and accuracy