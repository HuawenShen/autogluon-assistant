# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides implementation guidance for Complex and Quaternion Neural Networks in SpeechBrain, focusing on specialized neural network architectures that handle complex and quaternion number operations. It covers implementation techniques for complex/quaternion tensor representations, linear operations, convolution layers, and RNN variants (LSTM, LiGRU, RNN). Key functionalities include weight sharing mechanisms, specialized layer implementations (CLinear, QLinear, CConv, QConv), normalization layers, and quaternion spinor neural networks for rotation modeling. The tutorial helps with tasks like configuring complex/quaternion networks, handling tensor operations, and implementing complete end-to-end models, particularly useful for speech processing applications. It includes practical code examples, YAML configurations, and best practices for initialization and dimensionality handling.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial content focusing on key implementation details and concepts:

# Complex and Quaternion Neural Networks in SpeechBrain

## Key Concepts

### Number Systems
- **Complex Numbers**: `z = r + ix`
  - Two-dimensional representation (real + imaginary parts)
  - Natural for speech signals and Fourier transforms

- **Quaternions**: `q = r + ix + jy + kz`
  - Four-dimensional representation (1 real + 3 imaginary parts)
  - Ideal for 3D operations and rotations

### Important Properties
- **Weight sharing**: Unique mechanism in C/Q-NNs different from traditional neural networks
- Preserves internal relationships between signal components
- Suitable for multidimensional input processing

## Implementation Details

### Tensor Representation
- SpeechBrain uses concatenated format for complex/quaternion values
- No special tensor types needed - uses standard PyTorch tensors

```python
# Complex tensor format: [r, r, r, x, x, x]
# Quaternion tensor format: [r, r, r, x, x, x, y, y, y, z, z, z]
```

### Requirements
- Feature dimension must be divisible by:
  - 2 for complex numbers
  - 4 for quaternion numbers

### Basic Usage

```python
import torch
from speechbrain.nnet.complex_networks.c_ops import get_real, get_imag

# Create tensor
T = torch.rand((1,8))

# Access components
real_part = get_real(T)
imag_part = get_imag(T)
```

## Best Practices
1. Use standard PyTorch tensors for input
2. Let the appropriate layer (C/Q-Layer) handle the complex/quaternion interpretation
3. Verify input tensor dimensions match requirements (divisible by 2 or 4)

## Important Notes
- Operations are abstracted in neural layers
- Any PyTorch tensor can be interpreted as complex/quaternion when processed by appropriate layers
- Layer determines interpretation:
  - `torch.nn.Linear` → real
  - `nnet.complex_networks.c_linear.CLinear` → complex

This condensed version maintains the essential implementation details while removing introductory and theoretical content that isn't directly related to practical usage.

Here's the condensed tutorial on Complex and Quaternion Neural Networks in SpeechBrain:

# Complex and Quaternion Products Implementation

## Key Concepts

1. Complex and quaternion numbers can be represented as real-valued matrices
2. Matrix multiplication simulates complex/quaternion products
3. Implementation follows two steps:
   - `init()`: Define and initialize weights
   - `forward()`: Implement specific product operations

## Core Implementation Details

### Complex Linear Operation
```python
def complex_linear_op(input, real_weight, imag_weight, bias):
    # Build real-valued matrix
    cat_real = torch.cat([real_weight, -imag_weight], dim=0)
    cat_imag = torch.cat([imag_weight, real_weight], dim=0)
    cat_complex = torch.cat([cat_real, cat_imag], dim=1)
    
    # Matrix multiplication
    if input.dim() == 2:
        return torch.addmm(bias, input, cat_complex) if bias.requires_grad else torch.mm(input, cat_complex)
    else:
        output = torch.matmul(input, cat_complex)
        return output + bias if bias.requires_grad else output
```

## Available Neural Network Components

### 1. Complex Convolution Layers
```python
# 1D Convolution
cnn_1d = CConv1d(input_shape=T.shape, out_channels=12, kernel_size=3)

# 2D Convolution
cnn_2d = CConv2d(input_shape=T.shape, out_channels=12, kernel_size=3)
```

### 2. Complex Linear Layer
```python
lin = CLinear(12, input_shape=T.shape, 
              init_criterion='glorot', 
              weight_init='complex')
```

### 3. Complex Normalization Layers
```python
CBN = CBatchNorm(input_shape=inp_tensor.shape)
CLN = CLayerNorm(input_shape=inp_tensor.shape)
```

## Important Notes

1. **Weight Initialization**: Complex/quaternion weights require careful initialization using `init_criterion` and `weight_init` parameters

2. **Activation Functions**: Currently uses split activation functions (real-valued functions applied to complex/quaternion signals)

3. **Dimensionality**: 
   - Output dimensions are doubled due to real/imaginary parts
   - For 2D convolutions, channel axis splits real/imaginary parts

4. **Implementation Pattern**: All layers follow the same pattern of building real-valued matrices to simulate complex/quaternion operations

5. **Available Components**:
   - 1D and 2D convolutions
   - Batch and layer normalizations
   - Linear layers
   - Recurrent cells (LSTM, LiGRU, RNN)

Here's the condensed version focusing on key implementation details and concepts:

# Complex and Quaternion Neural Networks Implementation

## Recurrent Neural Networks

### Key Implementation Details
- Complex RNNs use CLinear layers instead of standard Linear layers
- Available variants: CLSTM, CRNN, CLiGRU
- Bidirectional implementation supported

```python
from speechbrain.nnet.complex_networks.c_RNN import CLiGRU, CLSTM, CRNN

# Basic usage
lstm = CLSTM(hidden_size=12, input_shape=inp_tensor.shape, 
             weight_init='complex', bidirectional=True)
rnn = CRNN(hidden_size=12, input_shape=inp_tensor.shape, 
           weight_init='complex', bidirectional=True)
ligru = CLiGRU(hidden_size=12, input_shape=inp_tensor.shape, 
               weight_init='complex', bidirectional=True)
```

**Note**: Output dimension is 48 for bidirectional RNNs (12 complex numbers × 2 directions × 2 values per complex number)

## Quaternion Neural Networks

### Available Components
- 1D and 2D convolutions
- Batch and layer normalizations
- Linear and Spinor layers
- Recurrent cells (LSTM, LiGRU, RNN)

### Basic Implementation
```python
from speechbrain.nnet.quaternion_networks.q_CNN import QConv1d, QConv2d
from speechbrain.nnet.quaternion_networks.q_linear import QLinear
from speechbrain.nnet.quaternion_networks.q_RNN import QLiGRU, QLSTM, QRNN

# Layer definitions
cnn_1d = QConv1d(input_shape=T.shape, out_channels=12, kernel_size=3)
lin = QLinear(12, input_shape=T.shape, init_criterion='glorot', 
              weight_init='quaternion')
lstm = QLSTM(hidden_size=12, input_shape=T.shape, weight_init='quaternion', 
             bidirectional=True)
```

## Quaternion Spinor Neural Networks (SNN)

### Key Concepts
- Specialized for rotation modeling
- Uses rotation equation: $\vec{v_{output}} = q_{weight} \vec{v_{input}} q^{-1}_{weight}$
- Implemented using matrix multiplication for efficiency

### Implementation Details
```python
# Spinor layer examples
cnn_1d = QConv1d(input_shape=T.shape, out_channels=12, kernel_size=3, 
                 spinor=True, vector_scale=True)
lin = QLinear(12, input_shape=T.shape, spinor=True, vector_scale=True)
```

### Important Best Practices
1. Always use `vector_scale=True` for deep models to prevent value explosion
2. Consider using `weight_init='unitary'` for better training stability
3. Input/output should be 3D vectors (real components should be zero)

**Note**: All activation functions are currently split activation functions (real-valued functions applied to complex/quaternion signals)

Here's the condensed version of the final chunk, focusing on key implementation details:

# Implementation Summary

## Complete Model Example

### Key YAML Configuration
```yaml
model: !new:speechbrain.nnet.containers.Sequential
    input_shape: [!ref <N_batch>, null, 660]
    # Quaternion CNN layers
    conv1: !name:speechbrain.nnet.quaternion_networks.q_CNN.QConv1d
        out_channels: 16
        kernel_size: 3
    conv2: !name:speechbrain.nnet.quaternion_networks.q_CNN.QConv1d
        out_channels: 32
        kernel_size: 3
    # Pooling layer
    pooling: !new:speechbrain.nnet.pooling.Pooling1d
        pool_type: "avg"
        kernel_size: 3
    # Quaternion RNN layer
    RNN: !name:speechbrain.nnet.quaternion_networks.q_RNN.QLiGRU
        hidden_size: 64
        bidirectional: True
    # Output layers
    linear: !name:speechbrain.nnet.linear.Linear
        n_neurons: 43  # 42 phonemes + 1 blank
        bias: False
    softmax: !new:speechbrain.nnet.activations.Softmax
        apply_log: True
```

### Implementation Notes
- Complete examples available in:
  - `speechbrain/tests/integration/ASR_CTC/example_asr_ctc_experiment_complex_net.yaml`
  - `speechbrain/tests/integration/ASR_CTC/example_asr_ctc_experiment_quaternion_net.yaml`
- The example shows a quaternion-valued CNN-LiGRU model for end-to-end CTC ASR
- Model architecture combines quaternion convolutions, pooling, quaternion RNN, and standard output layers

### Citation Information
When using SpeechBrain, cite both:
- SpeechBrain V1.0 paper (2024)
- Original SpeechBrain toolkit paper (2021)

[Citation details omitted for brevity but should be included in implementation]