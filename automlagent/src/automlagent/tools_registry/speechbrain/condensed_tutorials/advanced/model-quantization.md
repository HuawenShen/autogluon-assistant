# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates implementing model quantization for SpeechBrain ASR models using PyTorch, covering both dynamic and static quantization techniques. It provides implementation details for quantizing specific model components like Wav2Vec 2.0 encoders and feature extractors, with code for custom quantization wrappers, benchmarking utilities, and preprocessing functions. Key functionalities include selective module quantization, calibration for static quantization, performance measurement (WER and RTF metrics), and handling different tokenizer types. The tutorial helps with tasks like reducing model size, improving inference speed, and maintaining accuracy through careful quantization strategy selection, particularly useful for deploying ASR models with memory and latency constraints.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial content focusing on essential implementation details:

# Model Quantization in SpeechBrain

## Key Concepts

### Quantization Fundamentals
- Converts weights/activations from floating point to lower resolution values (e.g., 8-bit integers)
- Benefits: Reduced memory footprint and inference latency
- Core formula: `y = round(x/S + Z)` where:
  - S = scale factor
  - Z = zero point
  - x = original value
  - y = quantized value

### Quantization Approaches

1. **Post-Training Quantization (PTQ)**
   - Dynamic Quantization
     - Weights quantized during prep
     - Activations quantized during inference
     - Adjusts parameters at runtime
   
   - Static Quantization  
     - Uses fixed parameters
     - Requires calibration with sample data
     - Generally lower latency but may sacrifice accuracy

2. **Quantization-Aware Training (QAT)**
   - Incorporates quantization during training
   - Not covered in this tutorial

## Implementation Requirements

### Dependencies
```python
# Core dependencies
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@develop

# Model-specific dependencies (if needed)
!pip install https://github.com/kpu/kenlm/archive/master.zip
!pip install pygtrie
```

### Essential Imports
```python
import gc
import numpy as np
import torch
import torch.nn as nn
import tqdm
from collections import Counter
from copy import deepcopy
```

## Best Practices & Warnings
1. Choose quantization approach based on use case:
   - Dynamic: Better for varying input ranges
   - Static: Better for latency-critical applications
2. Consider model architecture compatibility
3. Test quantized model performance thoroughly
4. Ensure representative calibration data for static quantization

This tutorial focuses on applying PTQ to pretrained SpeechBrain ASR models using PyTorch quantization functions.

Here's the condensed version of chunk 2/4, focusing on key implementation details and concepts:

# ASR Model Quantization - Part 2

## Model Selection & Architecture
- Using Wav2Vec 2.0 model with CTC trained on CommonVoice English
- Transformer-based Encoder ASR model without decoder layer
- Optional n-gram rescoring with language model

```python
from speechbrain.inference.ASR import EncoderASR

asr_model = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-commonvoice-14-en",
    savedir="/content/pretrained_ASR/asr-wav2vec2-commonvoice-14-en",
)
```

## Quantization Strategy
### Quantizable Modules
- Dynamic: `nn.Linear`, `nn.LSTM`, `nn.GRU`, `nn.RNNCell`, `nn.GRUCell`, `nn.LSTMCell`, `nn.EmbeddingBag`, `nn.Embedding`
- Static: `nn.Linear`, `nn.Conv1d/2d/3d`, `nn.EmbeddingBag`, `nn.Embedding`

### Model-Specific Quantization Plan
1. Dynamic Quantization:
   - `encoder.wav2vec2.model.encoder.layers`
   - `encoder.enc`

2. Static Quantization:
   - `encoder.wav2vec2.model.feature_extractor`
   - `encoder.wav2vec2.model.feature_projection`

3. No Quantization:
   - `encoder.ctc_lin` (impacts WER significantly)

## Data Preparation
```python
def get_samples(root):
    audios = []
    references = []
    for book in os.listdir(root):
        for chapter in os.listdir(f"{root}/{book}"):
            for file in os.listdir(f"{root}/{book}/{chapter}"):
                if file.endswith("txt"):
                    with open(f"{root}/{book}/{chapter}/{file}", "r") as f:
                        for line in f.readlines():
                            audio_path, reference = line.split(" ", 1)
                            full_audio_path = f"{root}/{book}/{chapter}/{audio_path}.flac"
                            audios.append(read_audio(full_audio_path))
                            references.append(reference)
    return audios, references
```

## Utility Functions
```python
def get_module(model, module_string):
    curr = model.mods
    for attr in module_string.split("."):
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    return curr

def set_module(model, module_string, new_module):
    curr = model.mods
    attrs = module_string.split(".")
    for attr in attrs[:-1]:
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    if attrs[-1].isnumeric():
        curr[int(attrs[-1])] = new_module
    else:
        setattr(curr, attrs[-1], new_module)
```

### Important Notes
- Avoid over-granular quantization due to quantization/dequantization overhead
- Different submodules may respond differently to quantization methods
- Experimentation needed to find optimal quantization combination
- BatchNorm layers have specific restrictions for static quantization

Here's the condensed version focusing on key implementation details and concepts:

# Static Quantization and Benchmarking Implementation

## Static Quantization Wrapper

```python
class StaticQuant(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model 
        self.dequant = DeQuantStub()

    def forward(self, x, *args, **kwargs):
        x = self.quant(x)
        x = self.model(x, *args, **kwargs)
        # Handle tuple returns from model
        if isinstance(x, tuple):
            return tuple(self.dequant(output) for output in x)
        return self.dequant(x)
```

Key points:
- Uses `QuantStub`/`DeQuantStub` to mark quantization boundaries
- Handles both single and tuple outputs from model
- Preserves model attribute access via `__getattr__` override

## Custom Quantization Function

```python
def custom_quantize(
    model,
    dynamic_modules=None, 
    static_modules=None,
    calibration_samples=None,
    dynamic_targets=None,
    dynamic_dtype=torch.qint8,
    static_qconfig=torch.ao.quantization.default_qconfig,
):
    # Dynamic Quantization
    if dynamic_modules:
        if not dynamic_targets:
            dynamic_targets = {nn.LSTM, nn.Linear, nn.GRU, nn.RNNCell, 
                             nn.GRUCell, nn.LSTMCell}
        
        for module in dynamic_modules:
            torch.quantization.quantize_dynamic(
                get_module(model, module),
                dynamic_targets,
                dtype=dynamic_dtype,
                inplace=True
            )

    # Static Quantization  
    if static_modules:
        if not calibration_samples:
            raise Exception("Calibration samples required for static quantization")

        for module in static_modules:
            set_module(model, module, StaticQuant(get_module(model, module)))
            get_module(model, module).qconfig = static_qconfig

        torch.ao.quantization.prepare(model, inplace=True)
        
        # Calibration
        for sample in calibration_samples:
            model.transcribe_batch(sample.unsqueeze(0), torch.tensor([1.0]))
            
        torch.ao.quantization.convert(model, inplace=True)
```

Important aspects:
- Supports both dynamic and static quantization
- Allows targeting specific submodules
- Requires calibration samples for static quantization
- Configurable quantization parameters

## Benchmarking Setup

Key metrics:
1. Real-time Factor (RTF)
   - Ratio of inference time to audio length
   - RTF < 1 indicates potential real-time capability

2. Word Error Rate (WER)
   ```python
   def compute_wer(references, hypotheses):
       references = [ref.split() for ref in references] 
       hypotheses = [hyp.split() for hyp in hypotheses]
       stats = accumulatable_wer_stats(references, hypotheses, Counter())
       return stats['WER']
   ```

Best Practices:
- Use appropriate calibration samples for static quantization
- Monitor both speed (RTF) and accuracy (WER) metrics
- Consider module compatibility when choosing quantization strategy
- Test with representative input data

Here's the condensed version focusing on key implementation details and concepts:

# ASR Model Quantization - Implementation Details

## Key Functions

### Preprocessing and Generation
```python
def preprocess_input(model: EncoderASR, input):
    with torch.no_grad():
        wavs = input.unsqueeze(0).float().to(model.device)
        wav_lens = torch.tensor([1.0]).to(model.device)
        return wavs, wav_lens

def generate(model, predictions):
    # Handle different tokenizer types
    is_ctc = isinstance(model.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder)
    if isinstance(model.hparams.decoding_function, functools.partial):
        predicted_words = [
            "".join(model.tokenizer.decode_ndim(token_seq)) if is_ctc
            else model.tokenizer.decode_ids(token_seq)
            for token_seq in predictions
        ]
    else:
        predicted_words = [hyp[0].text for hyp in predictions]
    return predicted_words
```

### Benchmarking Functions
```python
def timed_transcribe(model: EncoderASR, input):
    with torch.no_grad():
        wavs, wav_lens = preprocess_input(model, input)
        start = time.time()
        encoder_out = model.mods.encoder(wavs, wav_lens)
        duration = time.time() - start
        predictions = model.decoding_function(encoder_out, wav_lens)
        return generate(model, predictions)[0], duration

def benchmark(model, samples, references):
    # Warmup phase
    for sample in samples[:10]:
        timed_transcribe(model, sample)
    
    # Actual benchmarking
    total_audio_length = sum([sample.shape[0] / 16000 for sample in samples])
    total_cpu_time = 0
    outputs = []
    
    for sample in samples:
        output, duration = timed_transcribe(model, sample)
        outputs.append(output)
        total_cpu_time += duration

    return compute_wer(references, outputs), total_cpu_time / total_audio_length
```

## Quantization Configuration


...(truncated)