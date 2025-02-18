# Condensed: AutoMM Detection - Convert VOC Format Dataset to COCO Format

Summary: This tutorial demonstrates how to convert Pascal VOC format datasets to COCO format using AutoGluon's multimodal CLI tool (autogluon.multimodal.cli.voc2coco). It covers implementation techniques for both pre-existing dataset splits and custom split ratio conversions, making it valuable for data preprocessing tasks in object detection workflows. Key functionalities include handling VOC's XML annotations, maintaining proper directory structures, and generating COCO-formatted JSON files for train/val/test sets. The tutorial is particularly useful for developers working with AutoMM, which prefers COCO format for optimal performance.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Converting VOC Format Dataset to COCO Format

## Key Points
- Pascal VOC uses XML format for annotations
- AutoMM strongly recommends using COCO format
- Conversion tool: `autogluon.multimodal.cli.voc2coco`

## Directory Structure
```
VOCdevkit/VOC2007/
├── Annotations/
├── ImageSets/Main/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── JPEGImages/
```

## Implementation Methods

### 1. Converting Existing Splits
```bash
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```

**Output files:**
- `train_cocoformat.json`
- `val_cocoformat.json`
- `test_cocoformat.json`

### 2. Custom Split Ratios
```bash
python3 -m autogluon.multimodal.cli.voc2coco \
    --root_dir ./VOCdevkit/VOC2007 \
    --train_ratio 0.6 \
    --val_ratio 0.2
```

**Output files:**
- `usersplit_train_cocoformat.json`
- `usersplit_val_cocoformat.json`
- `usersplit_test_cocoformat.json`

## Best Practices
1. Use COCO format for optimal compatibility with AutoMM
2. VOC format is supported but only for quick testing
3. Custom splits don't require pre-existing split files

For more examples and customization options, refer to:
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)