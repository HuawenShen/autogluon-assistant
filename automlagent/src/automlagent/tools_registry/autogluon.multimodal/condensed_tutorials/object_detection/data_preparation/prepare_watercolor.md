# Condensed: AutoMM Detection - Prepare Watercolor Dataset

Summary: This tutorial provides implementation guidance for preparing the Watercolor dataset for object detection tasks using AutoMM. It covers two installation methods (Python CLI and Bash script), focusing on dataset structure and format requirements. The tutorial helps with tasks related to dataset preparation, format conversion, and proper directory setup for object detection. Key features include cross-platform installation options, COCO/VOC format handling, and dataset organization with 2,000 watercolor images. It's particularly useful for understanding proper data structuring and format recommendations when working with AutoMM MultiModalPredictor for detection tasks.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Watercolor Dataset Preparation

## Key Requirements
- Disk space: 7.5 GB
- Preferred storage: SSD
- Estimated preparation time: ~8 min (on AWS EC2 with EBS)

## Dataset Installation Methods

### 1. Using Python CLI (Cross-platform)
```python
# Basic installation in current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor

# Installation with custom output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d watercolor -o ~/data
```

### 2. Using Bash Script (Unix systems)
```bash
# Basic installation
bash download_watercolor.sh

# Installation with custom path
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

## Important Notes
1. **Format Recommendation**: COCO format is strongly recommended over VOC format for AutoMM MultiModalPredictor
2. **Dataset Details**:
   - 1,000 training images
   - 1,000 testing images
   - Domain: Watercolor images

## Best Practices
1. Use COCO format when possible
2. Refer to "Convert Data to COCO Format" tutorial for format conversion
3. For VOC format usage, ensure root path contains required directories (Annotations, ImageSets, JPEGImages)

## References
- [Official Dataset Page](https://naoto0804.github.io/cross_domain_detection/)
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization: Refer to "Customize AutoMM" tutorial