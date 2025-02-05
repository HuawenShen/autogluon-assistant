# Condensed: AutoMM Detection - Prepare Watercolor Dataset

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focused on essential implementation details:

# AutoMM Detection - Prepare Watercolor Dataset

## Overview
The Watercolor dataset is a specialized object detection dataset containing:
- 1,000 training images and 1,000 testing images
- Domain: watercolor images
- Required disk space: 7.5 GB (SSD recommended)

## Dataset Download Options

### Option 1: Python CLI Tool
Use the built-in AutoGluon CLI tool:

```python
# Basic usage
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor

# Specify custom output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d watercolor -o ~/data
```

### Option 2: Bash Script
Download using the provided bash script:

```bash
# Extract in current directory
bash download_watercolor.sh

# Extract to specific path
bash download_watercolor.sh ~/data
```

## Dataset Structure
After extraction, the dataset contains:
```
watercolor/
├── Annotations
├── ImageSets
└── JPEGImages
```

## Important Implementation Notes

1. **Format Recommendation**:
   - AutoMM MultiModalPredictor strongly recommends using COCO format
   - VOC format has limited support (used by Watercolor dataset)

2. **VOC Format Requirements**:
   - Root path must contain: `Annotations`, `ImageSets`, and `JPEGImages` directories
   - Used primarily for quick testing purposes

3. **Format Conversion**:
   - For production use, convert VOC to COCO format
   - Refer to "Convert Data to COCO Format" tutorial for conversion instructions

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

For customization options, refer to the "Customize AutoMM" tutorial.