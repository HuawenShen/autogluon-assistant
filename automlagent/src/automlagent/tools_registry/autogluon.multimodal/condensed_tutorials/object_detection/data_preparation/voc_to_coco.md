# Condensed: AutoMM Detection - Convert VOC Format Dataset to COCO Format

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# AutoMM Detection - Convert VOC Format Dataset to COCO Format

## Overview
This tutorial demonstrates how to convert Pascal VOC format datasets to COCO format, which is the recommended format for AutoGluon MultiModalPredictor.

## Key Components
- Source: Pascal VOC dataset (XML format)
- Target: COCO format (JSON)
- Tool: `voc2coco.py` utility

## Prerequisites
Required directory structure:
```
VOCdevkit/VOC2007/
├── Annotations/
├── ImageSets/Main/
└── JPEGImages/
```

## Conversion Methods

### 1. Converting Existing Splits
For datasets with predefined splits (train.txt, val.txt, test.txt):

```bash
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```

Output files (in Annotations folder):
- train_cocoformat.json
- val_cocoformat.json
- test_cocoformat.json

### 2. Custom Split Ratios
To create new splits with custom ratios:

```bash
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007 --train_ratio 0.6 --val_ratio 0.2
```

Output files (in Annotations folder):
- usersplit_train_cocoformat.json
- usersplit_val_cocoformat.json
- usersplit_test_cocoformat.json

## Important Notes
- COCO format is strongly recommended for AutoGluon MultiModalPredictor
- VOC format has limited support (only for quick testing)
- The conversion tool shows progress during execution
- Default test ratio is calculated as: 1 - train_ratio - val_ratio

## Additional Resources
- For customization: Refer to "Customize AutoMM" documentation
- More examples: Check AutoMM Examples repository

This focused version maintains all essential implementation details while removing redundant explanations and keeping the core functionality intact.