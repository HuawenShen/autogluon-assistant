# Condensed: AutoMM Detection - Prepare COCO2017 Dataset

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM Detection - Prepare COCO2017 Dataset

## Overview
COCO2017 is a large-scale object detection dataset featuring:
- 80 classes
- 123,287 images
- 886,284 instances
- Median image ratio: 640 x 480

**Requirements:**
- 42.7 GB disk space (SSD recommended)
- Typical setup time: ~20 min on AWS EC2 with EBS

## Installation Options

### 1. Python Script Method
Uses `prepare_detection_dataset` CLI tool. Works across all platforms but without progress bar.

```python
# Basic usage - extract to current directory
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017

# Extract to specific path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017 --output_path ~/data

# Short form
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d coco17 -o ~/data
```

### 2. Bash Script Method
Provides progress bar visualization (Unix systems only).

```bash
# Extract to current directory
bash download_coco17.sh

# Extract to specific path
bash download_coco17.sh ~/data
```

## Directory Structure
After extraction, the `coco17` folder contains:
```
annotations/
test2017/
train2017/
unlabeled2017/
val2017/
```

## Data Format
COCO uses a specific JSON format for annotations. This format is recommended for use with AutoGluon MultiModalPredictor.

For format conversion guidance:
- See "Convert Data to COCO Format" tutorial
- See "AutoMM Detection - Convert VOC Format Dataset to COCO Format" tutorial

## Additional Resources
- [AutoMM Examples Repository](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization: Refer to "Customize AutoMM" tutorial

## Citation
```
@misc{https://doi.org/10.48550/arxiv.1405.0312,
  doi = {10.48550/ARXIV.1405.0312},
  url = {https://arxiv.org/abs/1405.0312},
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Bourdev, Lubomir and Girshick, Ross and Hays, James and Perona, Pietro and Ramanan, Deva and Zitnick, C. Lawrence and Dollár, Piotr},
  title = {Microsoft COCO: Common Objects in Context},
  publisher = {arXiv},
  year = {2014}
}
```