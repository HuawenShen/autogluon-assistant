# Condensed: AutoMM Detection - Prepare Pothole Dataset

Summary: This tutorial provides implementation guidance for preparing a pothole detection dataset using AutoMM. It covers two main implementation approaches: using a CLI tool and manual download from Kaggle. The tutorial helps with dataset preparation tasks for object detection, specifically handling a 665-image pothole dataset. Key features include COCO/VOC format handling, dataset splitting (3:1:1 ratio), and file organization. It emphasizes using COCO format for AutoGluon MultiModalPredictor compatibility and includes specific CLI commands, storage requirements (1GB, preferably SSD), and expected preparation time (~3 min on AWS EC2).

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Pothole Dataset Preparation

## Key Requirements
- Disk space: 1 GB
- Preferred storage: SSD
- Typical preparation time: ~3 min on AWS EC2 with EBS

## Dataset Information
- Size: 665 images
- Domain: Road pothole detection
- Format: Available in COCO format (recommended) or VOC format
- Default split ratio: 3:1:1 (train/validation/test)

## Implementation Methods

### 1. Using CLI Tool (Recommended)
```bash
# Basic usage - current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole

# With custom output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name pothole --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```

### 2. Manual Download
Download from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)

## Important Notes
- **Strongly recommended**: Use COCO format instead of VOC format
- Generated annotation files location:
  ```
  pothole/Annotations/usersplit_train_cocoformat.json
  pothole/Annotations/usersplit_val_cocoformat.json
  pothole/Annotations/usersplit_test_cocoformat.json
  ```

## Best Practices
1. Use COCO format for compatibility with AutoGluon MultiModalPredictor
2. Refer to "Convert Data to COCO Format" tutorial for format conversion
3. Use SSD for better performance during data preparation

For customization details, refer to the "Customize AutoMM" documentation.