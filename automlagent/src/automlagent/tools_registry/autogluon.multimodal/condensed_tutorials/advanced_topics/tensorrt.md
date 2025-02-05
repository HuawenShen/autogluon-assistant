# Condensed: Faster Prediction with TensorRT

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Faster Prediction with TensorRT

This tutorial demonstrates how to use TensorRT with AutoGluon-MultiModal to optimize inference speed.

## Setup

```python
import os
import numpy as np
import time
import warnings
from IPython.display import clear_output
warnings.filterwarnings('ignore')
np.random.seed(123)

# Install required packages
try:
    import tensorrt, onnx, onnxruntime
except ImportError:
    !pip install autogluon.multimodal[tests]
    !pip install -U "tensorrt>=10.0.0b0,<11.0"
    clear_output()
```

## Data Preparation

```python
# Download and load dataset
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load CSV files
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'

# Process image paths
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

```python
from autogluon.multimodal import MultiModalPredictor

hyperparameters = {
    "optimization.max_epochs": 2,
    "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
    "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
    "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
}

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=120,
)
```

## TensorRT Optimization

```python
# Baseline PyTorch prediction
batch_size = 2
n_trails = 10
sample = test_data.head(batch_size)
y_pred = predictor.predict_proba(sample)

# TensorRT optimization
model_path = predictor.path
trt_predictor = MultiModalPredictor.load(path=model_path)
trt_predictor.optimize_for_inference()
y_pred_trt = trt_predictor.predict_proba(sample)
```

Important Notes:
- `optimize_for_inference()` modifies the model for inference only
- Cannot call `predictor.fit()` after optimization
- Uses FP16 precision by default

## Performance Comparison

```python
# Compare accuracy
np.testing.assert_allclose(y_pred, y_pred_trt, atol=0.01)

# Compare metrics
metric = predictor.evaluate(test_data)
metric_trt = trt_predictor.evaluate(test_data)
metric_df = pd.DataFrame.from_dict({"PyTorch": metric, "TensorRT": metric_trt})
```

## Customization Options

To use different precision:
```python
predictor.optimize_for_inference(providers=["CUDAExecutionProvider"])
```

Key Points:
- TensorRT optimization significantly improves inference speed
- Minor accuracy differences due to FP16 precision
- Can switch to CUDA provider for full precision if needed
- Optimization is inference-only; reload model for retraining

For more examples and customization options, refer to the AutoMM Examples repository.