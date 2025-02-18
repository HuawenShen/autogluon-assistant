# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides implementation guidance for speech feature extraction using SpeechBrain, focusing on Filter Banks (FBANKs) and MFCCs. It demonstrates code for computing these features with customizable parameters, including STFT computation, mel filtering, and context window handling. The tutorial helps with tasks like configuring feature extractors, adding temporal context through derivatives and window frames, and selecting appropriate parameters for speech processing. Key functionalities covered include FBANK computation with adjustable filter counts (40/80), MFCC extraction with optional derivatives, context window implementation, and best practices for modern deep learning approaches, emphasizing the preference for FBANKs over MFCCs in DNN applications.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Speech Features Tutorial

## Key Concepts
- Speech is high-dimensional (16000 samples/second at 16kHz)
- Feature extraction creates compact representations
- Modern approach: Use simple features and let neural networks learn higher-level representations
- Main features covered: Filter Banks (FBANKs) and MFCCs

## 1. Filter Banks (FBANKs)

### Implementation

```python
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank
from speechbrain.lobes.features import Fbank

# Basic FBANK computation
compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
compute_fbanks = Filterbank(n_mels=40)

# Pipeline:
STFT = compute_STFT(signal)
mag = spectral_magnitude(STFT)
fbanks = compute_fbanks(mag)

# Simplified approach using Fbank lobe
fbank_maker = Fbank()
fbanks = fbank_maker(signal)
```

**Key Parameters:**
- Typically use 40 or 80 FBANKs
- Filters can be triangular, rectangular, or gaussian
- Set `freeze=False` to allow filter tuning during training

## 2. MFCCs

```python
from speechbrain.lobes.features import MFCC

mfcc_maker = MFCC(n_mfcc=13, deltas=False, context=False)
mfccs = mfcc_maker(signal)
```

**Note:** While historically important for decorrelated features, FBANKs are now preferred as DNNs handle correlated data well.

## 3. Context Information

### Derivatives
```python
# MFCCs with derivatives
mfcc_maker = MFCC(n_mfcc=13, deltas=True, context=False)
mfccs_with_deltas = mfcc_maker(signal)  # Results in 39 features (13x3)
```

### Context Windows
```python
# MFCCs with context window
mfcc_maker = MFCC(n_mfcc=13,
                  deltas=True,
                  context=True,
                  left_frames=5,
                  right_frames=5)
mfccs_with_context = mfcc_maker(signal)  # Dimensionality: 39 * (5+5+1) = 429
```

## Best Practices
1. Modern approach: Use static features with CNNs for learnable context
2. Consider using raw features:
   - Spectrograms
   - STFT
   - Raw time-domain samples (with appropriate architectures like SincNet)
3. FBANKs are preferred over MFCCs for deep learning applications
4. Context is better learned through CNN receptive fields than hand-crafted features

## Important Parameters
- Sampling rate: typically 16kHz
- STFT parameters: win_length=25ms, hop_length=10ms, n_fft=400
- Number of mel filters: 40 or 80
- MFCC coefficients: typically 13
- Context window size: application dependent