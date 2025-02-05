# Condensed: AutoMM Detection - Prepare Pothole Dataset

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM Detection - Prepare Pothole Dataset

## Overview
The Pothole dataset is a small object detection dataset (665 images) for detecting potholes on roads. This guide shows how to prepare it for use with AutoMM Detection.

**Requirements:**
- Disk space: 1 GB
- Preferred storage: SSD
- Typical preparation time: ~3 min on AWS EC2 with EBS

## Dataset Preparation Options

### Option 1: Using AutoGluon CLI Tool (Recommended)
The `prepare_detection_dataset` tool automatically:
- Downloads the dataset
- Converts to COCO format
- Splits into train/validation/test (3:1:1 ratio)

**Basic Usage:**
```python
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole
```

**With Custom Output Path:**
```python
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole --output_path ~/data
```

**Shortened Command:**
```python
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```

**Output Structure:**
```
pothole/Annotations/
    ├── usersplit_train_cocoformat.json
    ├── usersplit_val_cocoformat.json
    └── usersplit_test_cocoformat.json
```

### Option 2: Manual Download from Kaggle
- Available at: [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)
- **Important:** Dataset comes in VOC format and requires conversion to COCO format
- Refer to these tutorials for conversion:
  - [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb)
  - [Convert Data to COCO Format](convert_data_to_coco_format.ipynb)

## Important Notes
- COCO format is strongly recommended for AutoGluon MultiModalPredictor
- The dataset is used in tutorials:
  - [Fast Finetune on COCO Format Dataset](../finetune/detection_fast_finetune_coco.ipynb)
  - [High Performance Finetune on COCO Format Dataset](../finetune/detection_high_performance_finetune_coco.ipynb)

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)

## Citation
```
@inproceedings{inoue_2018_cvpr,
    author = {Inoue, Naoto and Furuta, Ryosuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
    title = {Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```