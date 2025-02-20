Summary: This tutorial provides implementation guidance for Complex and Quaternion Neural Networks in SpeechBrain, focusing on specialized neural network architectures that handle complex and quaternion number operations. It covers implementation techniques for complex/quaternion tensor representations, linear operations, convolution layers, and RNN variants (LSTM, LiGRU, RNN). Key functionalities include weight sharing mechanisms, specialized layer implementations (CLinear, QLinear, CConv, QConv), normalization layers, and quaternion spinor neural networks for rotation modeling. The tutorial helps with tasks like configuring complex/quaternion networks, handling tensor operations, and implementing complete end-to-end models, particularly useful for speech processing applications. It includes practical code examples, YAML configurations, and best practices for initialization and dimensionality handling.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/complex-and-quaternion-neural-networks.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/nn/complex-and-quaternion-neural-networks.ipynb)

# Complex and Quaternion Neural Networks

This tutorial demonstrates how to use the SpeechBrain implementation of complex-valued and quaternion-valued neural networks for speech technologies. It covers the basics of highdimensional representations and the associated neural layers : Linear, Convolution, Recurrent and Normalisation.

## Prerequisites
- [SpeechBrain Introduction](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/introduction-to-speechbrain.html)
- [YAML tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/hyperpyyaml.html)
- [Brain Class tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/brain-class.html)
- [Speech Features tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/speech-features.html)

## Introduction and Background

**Complex Numbers:**
Complex numbers extend the concept of real numbers into a two-dimensional space. Comprising a real and an imaginary part, a complex number `z` is typically expressed as `z = r + ix`, where `r` is the real part and `ix` is the imaginary part. This mathematical extension finds diverse applications in real-world scenarios, offering a powerful algebraic framework for manipulating concepts in two-dimensional space, such as rotations, translations, and phase-related operations. Complex numbers naturally represent the speech signal, with the Fourier transform being a notable example that operates in the complex space, capturing amplitude and phase information.

**Quaternion Numbers:**
Quaternions generalize complex numbers to the three-dimensional space, featuring a real (`r`) and an imaginary part, which is a 3D vector (`ix + jy + kz`). A quaternion `q` can be expressed as `q = r + ix + jy + kz`. In practice, quaternions define 3D rotations and find extensive utility in physics, computer science, computer graphics, and robotics. They provide a stable and natural framework for conceiving and interpreting movements in three-dimensional space.

### Connection to Neural Networks:

As the resurgence of modern deep learning gained momentum, researchers explored the integration of complex and quaternion numbers into neural networks to address specific tasks. Complex-valued neural networks (CVNN) can directly handle the output of the Fast Fourier Transform (FFT), while quaternion neural networks (QNN) can be implemented to generate realistic robot movements.

Beyond their natural fit for certain representations, CVNN and QNN share a compelling property: **weight sharing**. The algebraic rules governing complex and quaternion numbers differ from those of real numbers, influencing the multiplication of quaternions or complex numbers. This distinction leads to a unique mechanism of **weight sharing** within Q-CVNN, as opposed to traditional dot products in real-valued networks. This mechanism has proven to be exceptionally useful for learning expressive representations of multidimensional inputs while preserving internal relationships within the signal components, such as amplitude and phase for complex numbers.

In this tutorial, we won't delve into all the intricacies of these properties due to their extensive nature. Instead, we aim to provide a detailed guide on how to effectively implement and utilize CVNN and QNN within SpeechBrain.


### Relevant bibliography
- *Andreescu, T., & Andrica, D. (2006). Complex Numbers from A to... Z (Vol. 165). Boston: Birkhäuser.*
- *Altmann, S. L. (1989). Hamilton, Rodrigues, and the quaternion scandal. Mathematics Magazine, 62(5), 291-308.*
- **Complex Neural Networks Survey:** *Hirose, A. (2012). Complex-valued neural networks (Vol. 400). Springer Science & Business Media.*
- **All about Quaternion Neural Networks:** *Parcollet, T., (2019) Quaternion Neural Networks, PhD Thesis, Avignon Université*

## SpeechBrain Representation of Complex and Quaternions

In SpeechBrain, algebraic operations are abstracted in the neural layers, freeing users from the need to focus on the initial representation. This abstraction ensures that users can manipulate real-valued tensors without explicitly declaring a specific tensor type for complex or quaternion numbers. The underlying operations are expressed in a tensor/matrix format, facilitating seamless integration with modern GPU architectures.

Practically, any PyTorch tensor generated in your recipe can be interpreted as a complex or quaternion-valued tensor, depending on the layer that processes it. For instance:
- If processed by a `torch.nn.Linear` layer, the tensor will be real.
- If processed by a `nnet.complex_networks.c_linear.CLinear` layer, the tensor will be complex.

**How are tensors interpreted and constructed?**

Let's illustrate with an example. Suppose we want to consider a tensor containing `3` complex numbers or `3` quaternions. The different parts of the numbers will be concatenated as follows:

For a complex tensor (`c_tensor`): `[r, r, r, x, x, x]`

For a quaternion tensor (`q_tensor`): `[r, r, r, x, x, x, y, y, y, z, z, z]`

This flexibility allows any tensor declared in your code to be viewed as a complex or quaternion tensor when processed by a {C/Q}-Layer in SpeechBrain, as long as the features dimension can be divided by 2 for complex numbers and 4 for quaternion numbers.

To explore this further, let's proceed with the installation of SpeechBrain.


```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH

!git clone https://github.com/speechbrain/speechbrain.git
```

Now, let's try to manipulate some Tensor to better understand the formalism. We start by instantiating a Tensor containing 8 real numbers.


```python
import torch

T = torch.rand((1,8))
print(T)
```

Then, we access the SpeechBrain libary for manipulating complex numbers and we simply display the different parts (real, imaginary).


```python
from speechbrain.nnet.complex_networks.c_ops import get_real, get_imag

print(get_real(T))
print(get_imag(T))
```

As you can see, the initial Tensor is simply splitted in 2 and the same happens with 4 and quaternions.

## Complex and quaternion products

At the core of QNN and CVNN is the product. Of course, others specificities exist such as the weight initialisation, specific normalisations, activation functions etc. Nevertheless, the basic product is central to all neural network layers : a weight matrix that multplies the input vector.

A very good thing to know is that a complex number can be represented in a real-valued matrix format:

\begin{equation}
\left(\begin{array}{rr}
a & -b \\
b & a
\end{array}\right).
\end{equation}

The same goes for a quaternion number:

\begin{equation}
\left(\begin{array}{cccc}
a & -b & -c & -d \\
b & a & -d & c \\
c & d & a & -b \\
d & -c & b & a
\end{array}\right).
\end{equation}

And even more interestingly, if we multiply two of these matrices, then we obtain the product corresponding to the considered algebra. For instance, the complex product between two complex number is defined as:

\begin{equation}
\left(\begin{array}{rr}
a & -b \\
b & a
\end{array}\right)\left(\begin{array}{lr}
c & -d \\
d & c
\end{array}\right)=\left(\begin{array}{cc}
a c-b d & -a d-b c \\
b c+a d & -b d+a c
\end{array}\right),
\end{equation}

which is equivalent to the formal definition:

\begin{equation}
(a+\mathrm{i} b)(c+\mathrm{i} d)=(a c-b d)+\mathrm{i}(a d+b c).
\end{equation}

**Ok, so how is this implemented in SpeechBrain**?

Every single layer that you can call either on the complex or quaternion libraries will follow two steps:
1. *init()*: Define the complex / quaternion weights as torch.Parameters and initialise them with the adapted scheme.
2. *forward()*: Call the corresponding operation that implements the specific product. For instance, a complex linear layer would call the `complex_linear_op()` from `speechbrain.nnet.complex_networks.c_ops`.

In practice, the `speechbrain.nnet.complex_networks.c_ops.complex_linear_op` function simply:
1. Takes the weights of the layer and builds the corresponding real-valued matrix.
2. Apply a product between the input and this matrix to simulate the complex / quaternion products.

Example:



```python
def complex_linear_op(input, real_weight, imag_weight, bias):
    """
    Applies a complex linear transformation to the incoming data.

    Arguments
    ---------
    input : torch.Tensor
        Complex input tensor to be transformed.
    real_weight : torch.Parameter
        Real part of the quaternion weight matrix of this layer.
    imag_weight : torch.Parameter
        First imaginary part of the quaternion weight matrix of this layer.
    bias : torch.Parameter
    """

    # Here we build the real-valued matrix as defined by the equations!
    cat_real = torch.cat([real_weight, -imag_weight], dim=0)
    cat_imag = torch.cat([imag_weight, real_weight], dim=0)
    cat_complex = torch.cat([cat_real, cat_imag], dim=1)

    # If the input is already [batch*time, N]

    # We do inputxconstructed_matrix to simulate the product

    if input.dim() == 2:
        if bias.requires_grad:
            return torch.addmm(bias, input, cat_complex)
        else:
            return torch.mm(input, cat_complex)
    else:
        output = torch.matmul(input, cat_complex)
        if bias.requires_grad:
            return output + bias
        else:
            return output

# We create a single complex number
complex_input = torch.rand(1, 2)

# We create two Tensors (not parameters here because we don't care about storing gradients)
# These tensors are the real_parts and imaginary_parts of the weight matrix.
# The real part is equivalent [nb_complex_numbers_in // 2, nb_complex_numbers_out // 2]
# The imag part is equivalent [nb_complex_numbers_in // 2, nb_complex_numbers_out // 2]
# Hence if we define a layer with 1 complex input and 2 complex outputs:
r_weight = torch.rand((1,2))
i_weight = torch.rand((1,2))

bias = torch.ones(4) # because we have 2 (complex) x times 2 = 4 real-values

# and we forward propagate!
print(complex_linear_op(complex_input, r_weight, i_weight, bias).shape)
```

**It is important to note that the quaternion implementation follows exactly the same approach.**

## Complex-valued Neural Networks

Once you are familiar with the formalism, you can easily derive any complex-valued neural building blocks given in `speechbrain.nnet.complex_networks`:
- 1D and 2D convolutions.
- Batch and layer normalisations.
- Linear layers.
- Recurrent cells (LSTM, LiGRU, RNN).

*According to the litterature, most of the complex and quaternion neural networks rely on split activation functions (any real-valued activation function applied over the complex/quaternion valued signal). For now, SpeechBrain follows this approach and does not offer any fully complex or quaternion activation function*.

### Convolution layers

First, let's define a batch of inputs (that could be the output of the FFT for example).



```python
from speechbrain.nnet.complex_networks.c_CNN import CConv1d, CConv2d

# [batch, time, features]
T = torch.rand((8, 10, 32))

# We define our layer and we want 12 complex numbers as output.
cnn_1d = CConv1d( input_shape=T.shape, out_channels=12, kernel_size=3)

out_tensor = cnn_1d(T)
print(out_tensor.shape)
```

As we can see, we applied a Complex-Valued 1D convolution over the input Tensor and we obtained an output Tensor whose features dimension is equal to 24. Indeed, we requested 12 `out_channels` which is equivalent to 24 real-values. Remember : **we always work with real numbers, the algebra is abstracted in the layer itself!**

The same can be done with 2D convolution.



```python
# [batch, time, fea, Channel]
T = torch.rand([10, 16, 30, 30])

cnn_2d = CConv2d( input_shape=T.shape, out_channels=12, kernel_size=3)

out_tensor = cnn_2d(T)
print(out_tensor.shape)
```

Please note that the 2D convolution is applied over the time and fea axis. The channel axis is used to be considered as the real and imaginary parts: `[10, 16, 30, 0:15] = real` and `[10, 16, 30, 15:30] = imag`.

### Linear layer

In the same manner as for convolution layers, we just need to instantiate the right module and use it!


```python
from speechbrain.nnet.complex_networks.c_linear import CLinear

# [batch, time, features]
T = torch.rand((8, 10, 32))

# We define our layer and we want 12 complex numbers as output.
lin = CLinear(12, input_shape=T.shape, init_criterion='glorot', weight_init='complex')

out_tensor = lin(T)
print(out_tensor.shape)
```

Please notice that we added the `init_criterion` and `weight_init` arguments. These two parameters that exist in **ALL** the complex and quaternion layers define how the weights are initialised. Indeed, complex and quaternion-valued weights need a carefull initialisation process as detailled in *Deep Complex Networks* by Chiheb Trabelsy et al. and `Quaternion Recurrent Neural Networks` from Titouan Parcollet et al.

### Normalization layers

One do not normalise a set of complex numbers (e.g the output of a complex-valued layers) in the same manner as a set of real-valued numbers. Due to the complexity of the task, this tutorial won't go into the details. Please note that the code is fully available in the corresponding SpeechBrain library and that it strictly follows the description first made in the paper *Deep Complex Networks* by Chiheb Trabelsy et al.

SpeechBrain supports both complex batch and layer normalisations:



```python
from speechbrain.nnet.complex_networks.c_normalization import CBatchNorm,CLayerNorm

inp_tensor = torch.rand([10, 16, 30])

# Not that by default the complex axis is the last one, but it can be specified.
CBN = CBatchNorm(input_shape=inp_tensor.shape)
CLN = CLayerNorm(input_shape=inp_tensor.shape)

out_bn_tensor = CBN(inp_tensor)
out_ln_tensor = CLN(inp_tensor)
```

### Recurrent Neural Networks

Recurrent neural cells are nothing more than multiple linear layers with a time connection. Hence, SpeechBrain provides an implementation for the complex variation of LSTM, RNN and LiGRU. As a matter of fact, these models are strictly equivalent to the real-valued ones, except that Linear layers are replaced with CLinear layers!


```python
from speechbrain.nnet.complex_networks.c_RNN import CLiGRU, CLSTM, CRNN

inp_tensor = torch.rand([10, 16, 40])

lstm = CLSTM(hidden_size=12, input_shape=inp_tensor.shape, weight_init='complex', bidirectional=True)
rnn = CRNN(hidden_size=12, input_shape=inp_tensor.shape, weight_init='complex', bidirectional=True)
ligru = CLiGRU(hidden_size=12, input_shape=inp_tensor.shape, weight_init='complex', bidirectional=True)

print(lstm(inp_tensor).shape)
print(rnn(inp_tensor).shape)
print(ligru(inp_tensor).shape)
```

Note that the output dimension is 48 as we have 12 complex numbers (24 values) times 2 directions (bidirectional RNNs).

## Quaternion Neural Networks

Luckily, QNN within SpeechBrain follow exactly the same formalism. Therefore, you can easily derive any quaternion-valued neural networks from the building blocks given in `speechbrain.nnet.quaternion_networks`:
- 1D and 2D convolutions.
- Batch and layer normalisations.
- Linear and Spinor layers.
- Recurrent cells (LSTM, LiGRU, RNN).

*According to the litterature, most of the complex and quaternion neural networks rely on split activation functions (any real-valued activation function applied over the complex/quaternion valued signal). For now, SpeechBrain follows this approach and does not offer any fully complex or quaternion activation function*.

Everything we just saw with complex neural networks still hold. Hence we can summarize everything in a single code snippet:


```python
from speechbrain.nnet.quaternion_networks.q_CNN import QConv1d, QConv2d
from speechbrain.nnet.quaternion_networks.q_linear import QLinear
from speechbrain.nnet.quaternion_networks.q_RNN import QLiGRU, QLSTM, QRNN

# [batch, time, features]
T = torch.rand((8, 10, 40))

# [batch, time, fea, Channel]
T_4d = torch.rand([10, 16, 30, 40])

# We define our layers and we want 12 quaternion numbers as output (12x4 = 48 output real-values).
cnn_1d = QConv1d( input_shape=T.shape, out_channels=12, kernel_size=3)
cnn_2d = QConv2d( input_shape=T_4d.shape, out_channels=12, kernel_size=3)

lin = QLinear(12, input_shape=T.shape, init_criterion='glorot', weight_init='quaternion')

lstm = QLSTM(hidden_size=12, input_shape=T.shape, weight_init='quaternion', bidirectional=True)
rnn = QRNN(hidden_size=12, input_shape=T.shape, weight_init='quaternion', bidirectional=True)
ligru = QLiGRU(hidden_size=12, input_shape=T.shape, weight_init='quaternion', bidirectional=True)

print(cnn_1d(T).shape)
print(cnn_2d(T_4d).shape)
print(lin(T).shape)
print(lstm(T)[0].shape) # RNNs return output + hidden so we need to filter !
print(ligru(T)[0].shape) # RNNs return output + hidden so we need to filter !
print(rnn(T)[0].shape) # RNNs return output + hidden so we need to filter !

```

### Quaternion Spinor Neural Networks

**Introduction:**
Quaternion Spinor Neural Networks (SNN) represent a specialized category within quaternion-valued neural networks (QNN). As mentioned earlier, quaternions are designed to represent rotations. In QNN layers, the fundamental operation involves the Hamilton product (`inputs x weights`), where inputs and weights are sets of quaternions. This product essentially creates a new rotation equivalent to the composition of the first rotation followed by the second.

**Rotation Composition:**
Multiplying two quaternions results in a rotation that combines the individual rotations represented by each quaternion. For instance, given `q3 = q1 x q2`, it implies that *q3 is a rotation equivalent to a rotation by q1 followed by a rotation from q2*. In the context of Spinor Neural Networks, this concept is employed to compose new rotations, not to physically rotate objects, but to predict sequential rotations. For example, predicting the next movement of a robot involves using the previous movement (represented as a quaternion) as input to produce a new quaternion as the output, capturing the expected next movement.

**Modeling Rotations with SNN:**
Spinor Neural Networks (SNN) are specifically designed to model rotations. In scenarios like robotic movements, SNNs take 3D coordinates (x, y, z) of the object before the movement as input and predict its coordinates after the movement as the output.

**Formal Rotation Equation:**
To achieve this, the standard product in all layers of the network is replaced with the following equation:

\begin{equation}
\vec{v_{output}} = q_{weight} \vec{v_{input}} q^{-1}_{weight}.
\end{equation}

This equation formally defines the rotation of a vector $\vec{v}$ by a unit quaternion $q_{weight}$ (with a norm of 1), where $q^{-1}$ represents the conjugate of the quaternion. Both left and right products in this equation are Hamilton products.

In summary, Quaternion Spinor Neural Networks are tailored to model rotations, making them particularly suitable for applications where predicting sequential rotations or movements is crucial, such as in robotics or animation.


**Ok, so how is this implemented in SpeechBrain?**

In the exact same manner than for the standard Hamilton product! Indeed, such rotation can also be represented as a matrix product:

\begin{equation}
\left(\begin{array}{ccc}
a^{2}+b^{2}-c^{2}-d^{2} & 2 b c-2 a d & 2 a c+2 b d \\
2 a d+2 b c & a^{2}-b^{2}+c^{2}-d^{2} & 2 c d-2 a b \\
2 b d-2 a c & 2 a b+2 c d & a^{2}-b^{2}-c^{2}+d^{2}
\end{array}\right).
\end{equation}

Hence, we just need to define the `quaternion_op` that follows the same usual process:
1. Compose a real-valued matrix from the different weight components
2. Apply a matrix product between the input and this rotation matrix!

[Check the code!](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.nnet.quaternion_networks.q_ops.html#speechbrain.nnet.quaternion_networks.q_ops.quaternion_linear_rotation_op)

### Turning a quaternion layer into a spinor layer

Spinor layer can be activated with a boolean parameter in all quaternion layers.
Here are a couple of examples:




```python
from speechbrain.nnet.quaternion_networks.q_CNN import QConv1d
from speechbrain.nnet.quaternion_networks.q_linear import QLinear

# [batch, time, features]
T = torch.rand((8, 80, 16))

#
# NOTE: in this case the real components must be zero as spinor neural networks
# only input and output 3D vectors ! We don't do it here for the sake of compactness
#

# We define our layers and we want 12 quaternion numbers as output (12x4 = 48 output real-values).
cnn_1d = QConv1d( input_shape=T.shape, out_channels=12, kernel_size=3, spinor=True, vector_scale=True)
lin = QLinear(12, input_shape=T.shape, spinor=True, vector_scale=True)

print(cnn_1d(T).shape)
print(lin(T).shape)
```

Two remarks on Spinor layers:
1. We need to set a vector_scale to train deep models. The vector scale is just an other set torch.Parameters that will scale down the output of each Spinor layers. Indeed, the output of a SNN layer is a set of 3D vectors that are the sum of rotated 3D vectors. Quaternion rotations do not affect the magnitude of the rotated vector. Hence, by summing over and over rotated 3D vectors, we might end up very quickly with very large values (i.e the training will explode).
2. You might consider to use `weight_init='unitary'`. Indeed, quaternion rotations are valid only if the considered quaternion is unitary. Therefore, starting with unitary weights may facilitate the learning phase!

## Putting everyting together!

We provide a minimal example for both complex and quaternion neural networks:
- `speechbrain/tests/integration/ASR_CTC/example_asr_ctc_experiment_complex_net.yaml`.
- `speechbrain/tests/integration/ASR_CTC/example_asr_ctc_experiment_quaternion_net.yaml`.

If we take a look at one of these YAML params file, we can easily distinguish how to build our model out of the different blocks!


```python
yaml_params = """
model: !new:speechbrain.nnet.containers.Sequential
    input_shape: [!ref <N_batch>, null, 660]  # input_size
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
        """
```

Here, we have a very basic quaternion-valued CNN-LiGRU model that can be used to perform end-to-end CTC ASR!


```python
%cd /content/speechbrain/tests/integration/ASR_CTC/
!python example_asr_ctc_experiment.py example_asr_ctc_experiment_quaternion_net.yaml
```

## Citing SpeechBrain

If you use SpeechBrain in your research or business, please cite it using the following BibTeX entry:

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with {SpeechBrain} 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
