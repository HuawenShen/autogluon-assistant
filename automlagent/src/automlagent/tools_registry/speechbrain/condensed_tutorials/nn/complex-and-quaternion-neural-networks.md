# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial covers the implementation of Complex and Quaternion Neural Networks in SpeechBrain, focusing on specialized neural network architectures for processing complex (2D) and quaternion (4D) number representations. It provides implementation details for complex/quaternion-specific layers including linear operations, convolutions, RNNs (LSTM, LiGRU), and normalization layers, all using standard PyTorch tensors with concatenated format. Key features include weight sharing mechanisms, specialized initialization techniques, and Quaternion Spinor Neural Networks for rotation modeling. The tutorial helps with tasks requiring multidimensional signal processing, particularly in speech processing applications, and includes practical code examples and YAML configurations for building end-to-end ASR models using these architectures.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial content focusing on key implementation details and concepts:

# Complex and Quaternion Neural Networks in SpeechBrain

## Key Concepts

### Number Representations
- **Complex Numbers**: `z = r + ix`
  - Two-dimensional representation (real + imaginary parts)
  - Natural for speech signals and Fourier transforms

- **Quaternion Numbers**: `q = r + ix + jy + kz`
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
# Complex tensor format: [r1,r2,r3, x1,x2,x3]
# Quaternion tensor format: [r1,r2,r3, x1,x2,x3, y1,y2,y3, z1,z2,z3]
```

### Key Requirements
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

### Layer Processing
- Regular tensors are automatically interpreted based on the layer type:
  - `torch.nn.Linear` → real-valued processing
  - `nnet.complex_networks.c_linear.CLinear` → complex-valued processing

## Best Practices
1. Ensure input tensor dimensions match the required format
2. Use appropriate layer types for desired number representation
3. No need to manually declare complex/quaternion types - handled by layers

## Prerequisites
- SpeechBrain Introduction
- YAML understanding
- Brain Class knowledge
- Speech Features basics

This implementation abstracts complex mathematical operations into neural layers for easier usage while maintaining computational efficiency.

Here's the condensed tutorial focusing on key implementation details and concepts:

# Complex and Quaternion Neural Networks Implementation

## Core Concepts

### Matrix Representation
- Complex and quaternion numbers can be represented as real-valued matrices
- Complex number matrix format:
```
[a  -b]
[b   a]
```
- Matrix multiplication of these representations yields the corresponding algebra product

## Implementation in SpeechBrain

### Core Structure
1. **Initialization**: Define complex/quaternion weights as torch.Parameters
2. **Forward Pass**: Implement specific product operations

### Key Implementation Example: Complex Linear Operation
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

### Supported Layers
- 1D and 2D convolutions
- Batch and layer normalizations
- Linear layers
- Recurrent cells (LSTM, LiGRU, RNN)

### Important Implementation Notes
- All operations work with real numbers internally
- Complex algebra is abstracted within layers
- Split activation functions are used (real-valued functions applied to complex signals)

### Layer Examples

1. **Complex Convolution**
```python
# 1D Convolution
cnn_1d = CConv1d(input_shape=T.shape, out_channels=12, kernel_size=3)

# 2D Convolution
cnn_2d = CConv2d(input_shape=T.shape, out_channels=12, kernel_size=3)
```

2. **Complex Linear**
```python
lin = CLinear(12, input_shape=T.shape, 
              init_criterion='glorot', 
              weight_init='complex')
```

3. **Complex Normalization**
```python
CBN = CBatchNorm(input_shape=inp_tensor.shape)
CLN = CLayerNorm(input_shape=inp_tensor.shape)
```

### Critical Configurations
- Weight initialization requires special attention
- Parameters `init_criterion` and `weight_init` are crucial for all complex/quaternion layers
- For 2D convolutions, channel axis represents real and imaginary parts

### Best Practices
- Always consider the doubled size of output features (real + imaginary parts)
- Use appropriate initialization schemes as described in literature
- Follow proper normalization techniques for complex-valued numbers

Here's the condensed version focusing on key implementation details and concepts:

# Complex and Quaternion Neural Networks Implementation

## Recurrent Neural Networks

### Key Implementation Details
- Complex RNNs use CLinear layers instead of standard Linear layers
- Available variants: CLSTM, CRNN, CLiGRU
- Bidirectional implementation supported

```python
from speechbrain.nnet.complex_networks.c_RNN import CLiGRU, CLSTM, CRNN

# Example implementation
lstm = CLSTM(hidden_size=12, input_shape=input_tensor.shape, 
             weight_init='complex', bidirectional=True)
```

**Note**: Output dimension is 2x hidden_size for bidirectional RNNs (e.g., 12 complex numbers × 2 directions = 48)

## Quaternion Neural Networks

### Available Components
- 1D/2D convolutions
- Batch/layer normalizations
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
- Uses rotation equation: v_output = q_weight * v_input * q_weight^(-1)
- Implemented using matrix multiplication for efficiency

### Implementation Details
```python
# Spinor layer activation
cnn_1d = QConv1d(input_shape=T.shape, out_channels=12, kernel_size=3, 
                 spinor=True, vector_scale=True)
lin = QLinear(12, input_shape=T.shape, spinor=True, vector_scale=True)
```

### Important Best Practices
1. Always use `vector_scale=True` for deep models to prevent explosion of values
2. Consider using `weight_init='unitary'` for better learning stability
3. Input/output should be 3D vectors (real components should be zero)

**Note**: All activation functions are currently split activation functions (real-valued functions applied to complex/quaternion signals)

Here's the condensed version of the final chunk, focusing on key implementation details and practical usage:

# Implementation Example

## Complete Model Configuration

Key example implementations are available for both complex and quaternion neural networks in:
- `speechbrain/tests/integration/ASR_CTC/example_asr_ctc_experiment_complex_net.yaml`
- `speechbrain/tests/integration/ASR_CTC/example_asr_ctc_experiment_quaternion_net.yaml`

### Sample Quaternion Model Configuration
```yaml
model: !new:speechbrain.nnet.containers.Sequential
    input_shape: [!ref <N_batch>, null, 660]
    conv1: !name:speechbrain.nnet.quaternion_networks.q_CNN.QConv1d
        out_channels: 16
        kernel_size: 3
    act1: !ref <activation>
    conv2: !name:speechbrain.nnet.quaternion_networks.q_CNN.QConv1d
        out_channels: 32
        kernel_size: 3
    nrm2: !name:speechbrain.nnet.quaternion_networks.q_CNN.QConv1d
    act2: !ref <activation>
    pooling: !new:speechbrain.nnet.pooling.Pooling1d
        pool_type: "avg"
        kernel_size: 3
    RNN: !name:speechbrain.nnet.quaternion_networks.q_RNN.QLiGRU
        hidden_size: 64
        bidirectional: True
    linear: !name:speechbrain.nnet.linear.Linear
        n_neurons: 43  # 42 phonemes + 1 blank
        bias: False
    softmax: !new:speechbrain.nnet.activations.Softmax
        apply_log: True
```

This configuration demonstrates a basic quaternion-valued CNN-LiGRU model for end-to-end CTC ASR, showing how to:
- Structure the model layers
- Configure quaternion-specific components
- Set up the final classification layer

### Running the Example
```bash
cd /content/speechbrain/tests/integration/ASR_CTC/
python example_asr_ctc_experiment.py example_asr_ctc_experiment_quaternion_net.yaml
```

## Citation Information
When using SpeechBrain, cite using the following BibTeX entries:

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with {SpeechBrain} 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and et al.},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv}
}

@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and et al.},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv}
}
```