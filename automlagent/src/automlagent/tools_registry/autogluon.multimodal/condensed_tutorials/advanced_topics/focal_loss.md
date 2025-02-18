# Condensed: Handling Class Imbalance with AutoMM - Focal Loss

Summary: This tutorial demonstrates implementing focal loss in AutoGluon's MultiModalPredictor to handle class imbalance problems. It covers specific techniques for dataset preparation with controlled imbalance creation, focal loss configuration through key parameters (alpha, gamma, reduction), and training setup with Swin Transformer models. The tutorial helps with tasks involving imbalanced classification problems, particularly in multimodal settings. Key features include customizing focal loss parameters, calculating appropriate class weights, and optimizing model performance through gamma value tuning. It provides practical code examples for integrating focal loss into the training pipeline and best practices for handling imbalanced datasets.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Handling Class Imbalance with AutoMM - Focal Loss

## Key Concepts
- Focal loss helps balance hard/easy samples and uneven class distributions
- Useful for improving model performance on imbalanced datasets

## Implementation Details

### 1. Dataset Preparation
```python
from autogluon.multimodal.utils.misc import shopee_dataset
train_data, test_data = shopee_dataset(download_dir)

# Create imbalanced dataset
imbalanced_train_data = []
for lb in range(4):
    class_data = train_data[train_data.label == lb]
    sample_index = np.random.choice(np.arange(len(class_data)), 
                                  size=int(len(class_data) * ds), 
                                  replace=False)
    ds /= 3  # downsample 1/3 each time
    imbalanced_train_data.append(class_data.iloc[sample_index])
```

### 2. Focal Loss Configuration
Key parameters:
- `optimization.loss_function`: Set to "focal_loss"
- `optimization.focal_loss.alpha`: Class weights list (must match number of classes)
- `optimization.focal_loss.gamma`: Controls focus on hard samples (higher = more focus)
- `optimization.focal_loss.reduction`: "mean" or "sum"

### 3. Training with Focal Loss
```python
predictor = MultiModalPredictor(label="label", 
                              problem_type="multiclass", 
                              path=model_path)

predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.loss_function": "focal_loss",
        "optimization.focal_loss.alpha": weights,
        "optimization.focal_loss.gamma": 1.0,
        "optimization.focal_loss.reduction": "sum",
        "optimization.max_epochs": 10,
    },
    train_data=imbalanced_train_data
)
```

## Best Practices
1. Calculate class weights using inverse of class sample proportions
2. Experiment with different gamma values for optimal performance
3. Use focal loss when dealing with significant class imbalances
4. Compare performance with and without focal loss to validate improvements

## Important Notes
- Class weights list length must match total number of classes
- Higher gamma values increase focus on hard samples
- Focal loss can significantly improve performance on imbalanced datasets