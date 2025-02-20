# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates the implementation of an audio source separation system using PyTorch and SpeechBrain. It covers techniques for building a neural network-based separator that combines STFT processing (or learned convolutions), bidirectional LSTM, and mask-based separation. The tutorial helps with tasks like creating custom audio datasets, implementing a source separation model architecture, and setting up training pipelines with appropriate loss functions (L1 or SI-SNR). Key features include time-frequency domain conversion, mask generation for source isolation, phase reconstruction, and handling time dimension mismatches. The implementation provides practical code for data generation, model architecture (featuring LSTM and convolutional layers), and training configuration with specific hyperparameters for optimal separation performance.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details:

# Source Separation Tutorial - Essential Implementation

## Key Components

### 1. Data Generation
```python
# Generate synthetic data
N = 100
f_th = 200  # Frequency threshold
fs = 8000   # Sampling rate
T = 10000   # Signal length

# Create sources with different frequency ranges
source1 = torch.sin(2*np.pi*(f1/fs)*t)  # f1 < f_th
source2 = torch.sin(2*np.pi*(f2/fs)*t)  # f2 > f_th
mixture = source1 + source2

# Create datasets
train_dataset = data_utils.TensorDataset(source1[:N_train], source2[:N_train], mixture[:N_train])
test_dataset = data_utils.TensorDataset(source1[N_train:], source2[N_train:], mixture[N_train:])
```

### 2. Model Architecture
```python
class simpleseparator(nn.Module):
    def __init__(self, fft_size, hidden_size, num_sources=2):
        super().__init__()
        self.masking = nn.LSTM(
            input_size=fft_size//2 + 1, 
            hidden_size=hidden_size, 
            batch_first=True, 
            bidirectional=True
        )
        self.output_layer = nn.Linear(
            hidden_size*2, 
            num_sources*(fft_size//2 + 1)
        )
        self.fft_size = fft_size
        self.num_sources = num_sources
```

### 3. Key Processing Steps
1. Convert input to frequency domain using STFT
2. Calculate magnitude and phase
3. Process through bidirectional LSTM
4. Generate source masks
5. Reconstruct sources using inverse STFT

```python
def forward(self, inp):
    # STFT
    y = torch.view_as_real(torch.stft(inp, n_fft=self.fft_size, return_complex=True))
    
    # Magnitude and phase
    mag = torch.sqrt((y ** 2).sum(-1))
    phase = torch.atan2(y[:, :, :, 1], y[:, :, :, 0])
    
    # LSTM processing
    rnn_out = self.masking(mag.permute(0, 2, 1))[0]
    
    # Generate masks and reconstruct
    sources = []
    for n in range(self.num_sources):
        mask = lin_out[:, :, :, n]
        sourcehat_dft = (mask * mag).permute(0, 2, 1) * torch.exp(1j * phase)
        sourcehat = torch.istft(sourcehat_dft, n_fft=self.fft_size)
        sources.append(sourcehat)
```

## Important Notes
- Uses STFT for time-frequency domain conversion
- Employs mask-based separation using bidirectional LSTM
- Reconstructs sources using phase information from mixture
- Model outputs separated sources and their masks

## Configuration Parameters
- `fft_size`: Size of FFT window
- `hidden_size`: LSTM hidden dimension
- `num_sources`: Number of sources to separate (default=2)
- `batch_size`: Training batch size (10 in example)

This implementation uses SpeechBrain framework and PyTorch for source separation tasks.

Here's the condensed version focusing on key implementation details and concepts:

# Source Separation Implementation Details

## Model Training Setup

### Brain Class Implementation
```python
class SeparationBrain(sb.Brain):
    def __init__(self, train_loss, modules, opt_class):
        super(SeparationBrain, self).__init__(modules=modules, opt_class=opt_class)
        self.train_loss = train_loss
        
    def compute_forward(self, mix):
        # Get source estimates
        est_sources, _, _ = self.modules.mdl(mix)
        est_sources = torch.stack(est_sources, dim=-1)
        
        # Handle time dimension mismatch
        T_origin = mix.size(1)
        T_est = est_sources.size(1)
        if T_origin > T_est:
            est_sources = nn.functional.pad(est_sources, (0, 0, 0, T_origin - T_est))
        else:
            est_sources = est_sources[:, :T_origin, :]
        return est_sources
```

### Key Components:

1. **Loss Functions**:
   - Supports L1 loss and SI-SNR loss
   - SI-SNR implemented using permutation invariant training

2. **Training Configuration**:
```python
optimizer = lambda x: torch.optim.Adam(x, lr=0.0001)
N_epochs = 10
```

3. **Model Initialization**:
```python
separator = SeparationBrain(
    train_loss='l1',
    modules={'mdl': model},
    opt_class=optimizer
)
```

## Visualization and Analysis

### Key Visualization Components:
- Mixture spectrogram
- Estimated source masks
- Masked spectrograms
- Ground truth spectrograms

### Important Observations:
- Masks act as band-stop filters
- Each mask attempts to remove interference from other sources

## Best Practices and Tips:

1. **Time Dimension Handling**:
   - Always check and handle potential time dimension mismatches after convolution operations

2. **Loss Function Selection**:
   - L1 loss for basic training
   - SI-SNR for potentially better separation quality

3. **Suggested Experiments**:
   - Compare performance between L1 and SI-SNR losses
   - Try replacing STFT/ISTFT with learned convolutional layers
   - Analyze learned filter characteristics vs. DFT bases

This implementation provides a complete framework for training and evaluating source separation models using either traditional STFT-based or learned convolutional approaches.

Here's the condensed version focusing on key implementation details and code samples:

# Sound Source Separation Example with SpeechBrain

## Key Implementation Components

### 1. Dataset and DataLoader Setup
```python
class source_separation_dataset(Dataset):
    def __init__(self, train_mixs, train_source1s, train_source2s):
        self.mixs = train_mixs
        self.train_source1s = train_source1s
        self.train_source2s = train_source2s

    def __getitem__(self, idx):
        return self.mixs[idx], self.train_source1s[idx], self.train_source2s[idx]

# Setup data loaders
train_loader_audio = DataLoader(train_dataset_audio, batch_size=1)
valid_loader_audio = DataLoader(valid_dataset_audio, batch_size=1)
```

### 2. Audio Separator Model Architecture
```python
class audioseparator(nn.Module):
    def __init__(self, fft_size, hidden_size, num_sources=2, kernel_size=16):
        super(audioseparator, self).__init__()
        # Encoder
        self.encoder = nn.Conv1d(1, fft_size, kernel_size, stride=kernel_size//2)
        
        # MaskNet
        self.rnn = nn.LSTM(fft_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_size*2, num_sources*(fft_size))
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(fft_size, 1, kernel_size, stride=kernel_size//2)

    def forward(self, inp):
        y = nn.functional.relu(self.encoder(inp.unsqueeze(0)))
        y = y.permute(0, 2, 1)
        rnn_out = self.rnn(y)[0]
        lin_out = self.output_layer(rnn_out)
        lin_out = lin_out.reshape(lin_out.size(0), lin_out.size(1), -1, self.num_sources)
        
        # Source separation
        sources = []
        for n in range(self.num_sources):
            sourcehat_mask = nn.functional.relu(lin_out[:, :, :, n])
            sourcehat_latent = (sourcehat_mask * y[:, :T, :]).permute(0, 2, 1)
            sourcehat = self.decoder(sourcehat_latent).squeeze(0)
            sources.append(sourcehat)
        
        return sources, all_masks, y
```

### 3. Training Configuration
```python
# Model initialization
model_audio = audioseparator(fft_size=1024, hidden_size=300, kernel_size=256)

# Training parameters
optimizer = lambda x: torch.optim.Adam(x, lr=0.0002)
N_epochs = 200

# SpeechBrain trainer setup
separator = SeparationBrain(
    train_loss='si-snr',
    modules={'mdl': model_audio},
    opt_class=optimizer
)

# Training
separator.fit(epoch_counter, train_loader_audio, valid_loader_audio)
```

## Important Notes
- The model uses a combination of CNN, LSTM, and transposed convolution for source separation
- Key hyperparameters:
  - FFT size: 1024
  - Hidden size: 300
  - Kernel size: 256
  - Learning rate: 0.0002
  - Number of epochs: 200
- The implementation may introduce some artifacts in the separated audio
- Uses SI-SNR (Scale-Invariant Signal-to-Noise Ratio) as the training loss

## Usage Example
```python
# Inference
estimated_sources_test, all_masks, mag = model_audio.forward(mixture.unsqueeze(0))
# Access separated sources
source1 = estimated_sources_test[0].squeeze().detach()
source2 = estimated_sources_test[1].squeeze().detach()
```