# Condensed: AutoMM Detection - Prepare Pascal VOC Dataset

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM Detection - Prepare Pascal VOC Dataset

## Overview
Pascal VOC is an object detection dataset containing 20 classes across 16,551 training images. Common benchmark setup:
- Training: VOC2007 trainval + VOC2012 trainval
- Validation: VOC2007 test

**Requirements:**
- Disk space: 8.4 GB
- Recommended: SSD for better performance
- Typical preparation time: ~10 min on AWS EC2 with EBS

## Dataset Download Options

### 1. Python Script Method
Advantages: Cross-platform compatibility
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
Advantages: Progress bar visibility (Unix systems only)
```bash
# Current directory
bash download_voc0712.sh

# Specify output path
bash download_voc0712.sh ~/data
```

## Dataset Structure
After extraction:
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
1. **Recommended Format**: COCO format is preferred over VOC format for AutoMM MultiModalPredictor
2. **VOC Format Support**: Limited support available for quick testing
   - Required directory structure:
     ```
     ├── Annotations/
     ├── ImageSets/
     └── JPEGImages/
     ```

## Additional Resources
- [AutoMM Examples Repository](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM Tutorial](../../advanced_topics/customization.ipynb)
- [COCO Dataset Preparation Guide](prepare_coco17.ipynb)
- [VOC to COCO Conversion Guide](convert_data_to_coco_format.ipynb)

## Citation
```
@Article{Everingham10,
   author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
   title = "The Pascal Visual Object Classes (VOC) Challenge",
   journal = "International Journal of Computer Vision",
   volume = "88",
   year = "2010",
   number = "2",
   month = jun,
   pages = "303--338",
}
```