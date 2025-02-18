# Condensed: Faster Prediction with TensorRT

Summary: This tutorial demonstrates how to optimize AutoGluon models using TensorRT for faster inference, specifically covering the implementation of mixed-precision optimization and model conversion. It helps with tasks like configuring TensorRT integration, optimizing trained models for inference, and validating prediction accuracy post-optimization. Key features include FP16 precision optimization, execution provider configuration, batch size tuning, and performance benchmarking techniques. The tutorial provides essential code patterns for model optimization, accuracy verification, and performance evaluation while highlighting critical considerations for maintaining prediction accuracy during the optimization process.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Faster Prediction with TensorRT in AutoGluon

## Key Setup
```python
# Required packages
import tensorrt, onnx, onnxruntime
!pip install autogluon.multimodal[tests]
!pip install -U "tensorrt>=10.0.0b0,<11.0"
```

## Implementation Details

### 1. Training Configuration
```python
hyperparameters = {
    "optimization.max_epochs": 2,
    "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
    "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
    "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
}

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=120
)
```

### 2. TensorRT Optimization
```python
# Load and optimize predictor
model_path = predictor.path
trt_predictor = MultiModalPredictor.load(path=model_path)
trt_predictor.optimize_for_inference()
```

⚠️ **Important Warning**: `optimize_for_inference()` modifies internal model definition for inference only. Don't call `predictor.fit()` after optimization.

### 3. Key Configuration Options

- Default: Uses TensorRT with FP16 precision
- Alternative provider configuration:
```python
predictor.optimize_for_inference(providers=["CUDAExecutionProvider"])
```

## Best Practices

1. **Initialization**: Always run first prediction separately for memory allocation
```python
y_pred_trt = trt_predictor.predict_proba(sample)  # Initial run
```

2. **Accuracy Verification**: Check prediction consistency
```python
np.testing.assert_allclose(y_pred, y_pred_trt, atol=0.01)
```

3. **Performance Evaluation**: 
- Benchmark with multiple trials
- Compare metrics between PyTorch and TensorRT versions
- Monitor accuracy loss with mixed precision

## Critical Parameters

- `batch_size`: Affects inference speed
- `atol`: Tolerance for accuracy comparison (default: 0.01)
- `providers`: Execution provider selection for optimization

## Performance Considerations

1. Mixed precision (FP16) is used by default for better performance
2. If accuracy loss is significant:
   - Switch to CUDA execution provider
   - Adjust precision settings
3. Evaluate trade-off between speed improvement and accuracy loss

This implementation enables significant inference speed improvements while maintaining accuracy within acceptable tolerances.