# Condensed: AutoMM Detection - Prepare COCO2017 Dataset

Summary: This tutorial provides implementation guidance for preparing the COCO2017 dataset specifically for AutoMM Detection tasks. It covers two main implementation approaches: using Python CLI and Bash scripts, with detailed commands for dataset download and custom path configuration. The tutorial helps with dataset setup tasks, including proper directory structure organization and format requirements for AutoGluon MultiModalPredictor. Key features covered include handling a large-scale dataset (42.7GB) with 80 classes and 123,287 images, COCO format specifications, and system requirements like SSD storage recommendations. It's particularly useful for setting up object detection datasets in the correct format and structure.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - COCO2017 Dataset Preparation

## Key Requirements
- Disk space: 42.7 GB
- Recommended: SSD over HDD
- Typical setup time: ~20 min on AWS EC2 with EBS

## Installation Methods

### 1. Python CLI Method
```bash
# Basic usage
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017

# With custom output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017 --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d coco17 -o ~/data
```

### 2. Bash Script Method
```bash
# Basic usage
bash download_coco17.sh

# With custom output path
bash download_coco17.sh ~/data
```

## Dataset Structure
After download, the `coco17` folder contains:
```
annotations/
test2017/
train2017/
unlabeled2017/
val2017/
```

## Important Notes
- COCO2017 contains:
  - 80 classes
  - 123,287 images
  - 886,284 instances
  - Median image ratio: 640 x 480
- The COCO format (.json) is the recommended format for AutoGluon MultiModalPredictor
- For format conversion:
  - Refer to "Convert Data to COCO Format" tutorial
  - See "AutoMM Detection - Convert VOC Format Dataset to COCO Format"

## Best Practices
1. Use SSD for better performance
2. Follow the COCO format for dataset organization
3. Check available disk space before starting download
4. Use bash script on Unix systems if progress monitoring is needed

For customization details, refer to the "Customize AutoMM" documentation.