# Condensed: AutoMM Detection - Prepare Pascal VOC Dataset

Summary: This tutorial provides implementation guidance for preparing the Pascal VOC dataset for object detection tasks using AutoMM. It covers two main implementation methods: using Python CLI commands or bash scripts to download and structure the dataset (8.4GB). The tutorial details the exact directory structure required for VOC format, though it recommends using COCO format for AutoMM MultiModalPredictor. It helps with dataset download automation, proper directory organization, and format conversion tasks. Key features include flexible output path specification, options for downloading VOC2007/2012 separately or combined, and handling a 20-class dataset with 16,551 training images.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection - Pascal VOC Dataset Preparation

## Key Requirements
- Disk space: 8.4 GB
- Recommended: SSD for better performance
- Estimated preparation time: ~10 min on AWS EC2 with EBS

## Dataset Download Options

### 1. Python Script Method
```python
# Basic usage - current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712

# Specify output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name voc0712 --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc -o ~/data

# Download separately
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```

### 2. Bash Script Method
```bash
# Basic usage
bash download_voc0712.sh

# With custom output path
bash download_voc0712.sh ~/data
```

## Dataset Structure
```
VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── SegmentationObject/
└── VOC2012/
    └── [same structure as VOC2007]
```

## Important Notes
1. **Format Recommendation**: COCO format is strongly recommended over VOC format for AutoMM MultiModalPredictor
2. **Dataset Details**:
   - Combined VOC2007 trainval and VOC2012 trainval for training
   - VOC2007 test for validation
   - 20 classes
   - 16,551 total training images

## Required Directory Structure for VOC Format
Minimum required folders:
```
├── Annotations/
├── ImageSets/
└── JPEGImages/
```

## References
- For COCO format conversion: See "Convert Data to COCO Format" tutorial
- For customization: Refer to "Customize AutoMM" documentation
- Additional examples available in AutoMM Examples repository