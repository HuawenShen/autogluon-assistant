# Condensed: Convert Data to COCO Format

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Convert Data to COCO Format

## Overview
COCO format is the standard data format for object detection tasks in AutoGluon. This guide covers the essential requirements and conversion process.

## COCO Format Requirements

### 1. Folder Structure
```
<dataset_dir>/
    images/
        <imagename0>.<ext>
        <imagename1>.<ext>
        ...
    annotations/
        train_labels.json
        val_labels.json
        test_labels.json
```

### 2. JSON Structure
Required fields in `*_labels.json`:
```javascript
{
    "images": [image],      // List of all images
    "annotations": [annotation],  // List of all annotations
    "categories": [category]     // List of all categories
}

image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str
}

category = {
    "id": int,
    "name": str,
    "supercategory": str
}

annotation = {
    "id": int,
    "image_id": int,      // References image.id
    "category_id": int,   // References category.id
    "bbox": [x,y,width,height],
    "area": float,
    "iscrowd": int       // 0 or 1
}
```

**Important Notes:**
- `"info"` and `"licenses"` fields are optional
- For prediction, only the `"images"` field is required
- All IDs must be unique within their respective lists

## Converting VOC Format to COCO

### Prerequisites
- VOC dataset structure:
```
<path_to_VOCdevkit>/
    VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
        labels.txt
    VOC2012/
        ...
```

### Conversion Command
```python
# Custom split ratios
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>

# Use dataset's provided splits
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
```

## Converting Other Formats
- Use third-party tools like [FiftyOne](https://github.com/voxel51/fiftyone) for converting from CVAT, YOLO, KITTI, etc.
- Custom conversion scripts should output the exact COCO format structure detailed above
- Ensure all IDs are properly referenced between images, annotations, and categories

This format is required for all AutoMM object detection pipelines.