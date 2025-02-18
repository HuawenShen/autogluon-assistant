# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides implementation details for various Recurrent Neural Network (RNN) architectures in SpeechBrain, including vanilla RNN, LSTM, GRU, and LiGRU. It covers specific coding patterns for initializing and using these networks with PyTorch tensors in [batch, time, features] format. Key functionalities include bidirectional processing, multi-layer stacking, and parameter configuration. The tutorial is particularly valuable for tasks involving sequence processing, especially speech recognition, highlighting LiGRU's optimized architecture with BatchNorm integration and parameter sharing in bidirectional mode. Implementation examples demonstrate parameter structures, core equations, and best practices for handling common challenges like vanishing gradients.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Recurrent Neural Networks in SpeechBrain

## Key Concepts
- RNNs process sequences using feedback connections and memory
- Core equation: $h_t = f(x_t, h_{t−1}, θ)$
- Parameters are shared across time steps
- Input format: `[batch, time, features]`

## 1. Vanilla RNN

### Implementation
```python
from speechbrain.nnet.RNN import RNN

# Basic RNN usage
inp_tensor = torch.rand([4, 10, 20])  # [batch, time, features]
net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
out_tensor, _ = net(inp_tensor)
```

### Key Parameters
- `hidden_size`: Dimension of hidden state
- `bidirectional`: Enable bidirectional processing (doubles output features)
- `num_layers`: Stack multiple RNN layers

### Parameter Structure
- W: Input-to-hidden weights `[hidden_size, input_size]`
- U: Hidden-to-hidden weights `[hidden_size, hidden_size]`
- b: Bias terms (split into input and recurrent)

## 2. LSTM (Long-Short Term Memory)

### Core Equations
```
f_t = σ(Wf*xt + Uf*h(t-1) + bf)  # Forget gate
i_t = σ(Wi*xt + Ui*h(t-1) + bi)  # Input gate
o_t = σ(Wo*xt + Uo*h(t-1) + bo)  # Output gate
c_t = ft*c(t-1) + it*tanh(Wc*xt + Uc*h(t-1) + bc)  # Cell state
h_t = ot*tanh(ct)  # Hidden state
```

### Implementation
```python
from speechbrain.nnet.RNN import LSTM

inp_tensor = torch.rand([4, 10, 20])
net = LSTM(hidden_size=5, input_shape=inp_tensor.shape)
out_tensor, _ = net(inp_tensor)
```

## Important Notes
- RNNs can suffer from vanishing/exploding gradients
- Exploding gradients can be handled with gradient clipping
- Vanishing gradients can be mitigated with skip connections
- LSTMs better handle long-term dependencies through gating mechanisms
- Parameters are concatenated in implementation (e.g., all gate weights combined)

## Best Practices
- Use bidirectional RNNs when full sequence context is available
- Consider multiple layers for deeper feature extraction
- LSTM is generally more robust than vanilla RNN for long sequences
- Match input tensor dimensions: `[batch, time, features]`

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# RNN Variants: GRU and LiGRU Implementation

## Gated Recurrent Units (GRU)
A simplified version of LSTM using only two multiplicative gates:

### Key Equations
```
zt = σ(Wzxt + Uzht-1 + bz)      # update gate
rt = σ(Wrxt + Urht-1 + br)      # reset gate
h̃t = tanh(Whxt + Uh(ht-1 ⊙ rt) + bh)
ht = zt ⊙ ht-1 + (1-zt) ⊙ h̃t
```

### Implementation
```python
from speechbrain.nnet.RNN import GRU

# Basic GRU usage
inp_tensor = torch.rand([4, 10, 20])  # [batch, time, features]
net = GRU(hidden_size=5, input_shape=inp_tensor.shape)
out_tensor, _ = net(inp_tensor)
```

## Light Gated Recurrent Units (LiGRU)
Optimized version of GRU with single gate and improved efficiency.

### Key Modifications
1. Removes reset gate
2. Uses ReLU + BatchNorm instead of tanh
3. Shares parameters in bidirectional mode

### Key Equations
```
zt = σ(BN(Wzxt) + Uzht-1)
h̃t = ReLU(BN(Whxt) + Uhht-1)
ht = zt ⊙ ht-1 + (1-zt) ⊙ h̃t
```

### Implementation
```python
from speechbrain.nnet.RNN import LiGRU

# Basic LiGRU
net = LiGRU(hidden_size=5, input_shape=inp_tensor.shape)

# Bidirectional LiGRU (parameters shared between directions)
net_bi = LiGRU(hidden_size=5,
               input_shape=inp_tensor.shape,
               bidirectional=True)
```

### Parameter Comparison Example
```python
# Configuration for comparison
hidden_size = 512
num_layers = 4
bidirectional = True

# Model initialization
rnn = RNN(hidden_size=hidden_size, input_shape=inp_tensor.shape, 
          bidirectional=bidirectional, num_layers=num_layers)
lstm = LSTM(...)  # Similar parameters
gru = GRU(...)    # Similar parameters
ligru = LiGRU(...) # Similar parameters
```

### Key Benefits of LiGRU
- Parameter efficient (comparable to vanilla RNN)
- Maintains long-term dependency learning
- Better numerical stability through BatchNorm
- Improved performance in speech processing tasks

### Best Practices
1. Use BatchNorm with LiGRU to prevent numerical instabilities
2. Consider LiGRU for speech processing tasks where parameter efficiency is important
3. Leverage bidirectional mode in LiGRU for parameter sharing between directions

Here's the condensed version of the citation information:

## Citing SpeechBrain

When using SpeechBrain in research or business applications, include these citations:

```bibtex
# For SpeechBrain 1.0
@misc{speechbrainV1,
  title={Open-Source Conversational AI with {SpeechBrain} 1.0},
  author={Ravanelli, M. and others},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2407.00463},
}

# For original SpeechBrain
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Ravanelli, M. and others},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
}
```

**Key Point**: Include both citations when using SpeechBrain 1.0, as they reference both the original toolkit and its latest version.