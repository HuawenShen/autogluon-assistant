# Condensed: Few Shot Learning with AutoMM

*This is a condensed version that preserves essential implementation details and context.*

Here's the focused version of the tutorial:

# Few Shot Learning with AutoMM

This tutorial demonstrates few shot classification using AutoMM by leveraging foundation model features with SVM for both text and image classification tasks.

## Text Classification

### Data Preparation
```python
import pandas as pd
import os
from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/nlp_datasets/MLDoc-10shot-en.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)
dataset_path = os.path.join(download_dir)
train_df = pd.read_csv(f"{dataset_path}/train.csv", names=["label", "text"])
test_df = pd.read_csv(f"{dataset_path}/test.csv", names=["label", "text"])
```

### Few Shot Classifier Implementation
```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_text = MultiModalPredictor(
    problem_type="few_shot_classification",  # Key parameter for few shot learning
    label="label",
    eval_metric="acc",
)
predictor_fs_text.fit(train_df)
scores = predictor_fs_text.evaluate(test_df, metrics=["acc", "f1_macro"])
```

### Standard Classifier Comparison
```python
predictor_default_text = MultiModalPredictor(
    label="label",
    problem_type="classification",
    eval_metric="acc",
)
predictor_default_text.fit(train_data=train_df)
scores = predictor_default_text.evaluate(test_df, metrics=["acc", "f1_macro"])
```

## Image Classification

### Data Preparation
```python
import os
from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls/stanfordcars/"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/stanfordcars.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Download CSV files
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/train.csv
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/test.csv

# Process DataFrames
train_df_raw = pd.read_csv(os.path.join(download_dir, "train.csv"))
train_df = train_df_raw.drop(columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", 
                                     "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"])
train_df["ImageID"] = download_dir + train_df["ImageID"].astype(str)

test_df_raw = pd.read_csv(os.path.join(download_dir, "test.csv"))
test_df = test_df_raw.drop(columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", 
                                   "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"])
test_df["ImageID"] = download_dir + test_df["ImageID"].astype(str)
```

### Few Shot Classifier Implementation
```python
predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_fs_image.fit(train_df)
scores = predictor_fs_image.evaluate(test_df, metrics=["acc", "f1_macro"])
```

### Standard Classifier Comparison
```python
predictor_default_image = MultiModalPredictor(
    problem_type="classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_default_image.fit(train_data=train_df)
scores = predictor_default_image.evaluate(test_df, metrics=["acc", "f1_macro"])
```

Key Points:
- Use `problem_type="few_shot_classification"` for few shot learning tasks
- Works with both text and image data
- Typically outperforms standard classification on small datasets
- Requires data in DataFrame format with appropriate column names for labels and features

For customization options, refer to the AutoMM customization documentation.