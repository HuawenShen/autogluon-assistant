# Condensed: Knowledge Distillation in AutoMM

Summary: This tutorial demonstrates knowledge distillation implementation in AutoMM, specifically showing how to transfer knowledge from a large BERT teacher model (12 layers) to a smaller BERT student model (6 layers). It covers practical implementation techniques including dataset preparation using QNLI data, teacher model setup, and student model training with knowledge distillation. Key functionalities include configuring model checkpoints, optimization parameters, and evaluation methods. The tutorial is particularly useful for tasks involving model compression while maintaining performance, especially in resource-constrained deployment scenarios. It provides code examples for the complete distillation pipeline using the MultiModalPredictor framework.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Knowledge Distillation in AutoMM

## Key Concepts
- Knowledge distillation transfers knowledge from large teacher models to smaller student models
- Enables deployment of smaller models while maintaining performance benefits from larger models
- Useful for scenarios with limited deployment resources

## Implementation

### 1. Data Preparation
```python
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load QNLI dataset
dataset = load_dataset("glue", "qnli")

# Prepare train/valid/test splits
train_valid_df = dataset["train"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=123)
test_df = dataset["validation"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
```

### 2. Teacher Model Setup
```python
from autogluon.multimodal import MultiModalPredictor

# Load pre-trained teacher model
teacher_predictor = MultiModalPredictor.load("ag_distillation_sample_teacher/")
```

### 3. Knowledge Distillation Implementation
```python
# Initialize student model
student_predictor = MultiModalPredictor(label="label")

# Train student with knowledge distillation
student_predictor.fit(
    train_df,
    tuning_data=valid_df,
    teacher_predictor=teacher_predictor,
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",
        "optimization.max_epochs": 2,
    }
)

# Evaluate student model
print(student_predictor.evaluate(data=test_df))
```

## Important Configurations
- Teacher model: BERT (12 layers) - `google/bert_uncased_L-12_H-768_A-12`
- Student model: BERT (6 layers) - `google/bert_uncased_L-6_H-768_A-12`

## Best Practices
1. Use a well-trained teacher model
2. Choose an appropriate student model size based on deployment constraints
3. Validate performance on test data
4. For more complex scenarios, refer to the [AutoMM Distillation Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation)

## Additional Resources
- Detailed customization: See [Customize AutoMM](customization.ipynb)
- More examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- Advanced multilingual distillation: [PAWS-X example](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py)