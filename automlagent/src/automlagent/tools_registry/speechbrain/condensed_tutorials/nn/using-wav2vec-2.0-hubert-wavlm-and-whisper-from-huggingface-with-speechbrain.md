# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates the implementation of Wav2vec 2.0 and Whisper models within SpeechBrain using HuggingFace integrations. It covers techniques for model initialization, feature extraction, fine-tuning configurations, and encoder-decoder implementations. The tutorial helps with tasks like setting up dual optimizers for pretrained models, configuring YAML parameters, implementing ASR pipelines, and handling both frozen and fine-tunable models. Key features include encoder-only and full encoder-decoder implementations, greedy/beam search decoding, zero-shot ASR/ST capabilities, and proper integration with SpeechBrain's speech processing pipeline. The implementation details span model initialization, output dimension handling, optimizer configurations, and loss computations, with specific code examples for each component.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial content focusing on key implementation details:

# Using Wav2vec 2.0 and Whisper with SpeechBrain and HuggingFace

## Key Concepts
- Integration of HuggingFace pretrained models (Whisper, wav2vec 2.0, HuBERT, WavLM) with SpeechBrain
- Focus on using and fine-tuning pretrained models rather than pre-training
- Models can be connected to SpeechBrain's speech processing pipeline

## Important Architectures
1. **Wav2Vec 2.0**
   - Transformer-based encoder for self-supervised speech representation learning
   - Reference: [wav2vec2 paper](https://arxiv.org/abs/2006.11477)

2. **Whisper**
   - Full transformer (encoder-decoder) trained on 600k+ hours of speech
   - Reference: [whisper paper](https://cdn.openai.com/papers/whisper.pdf)

## Implementation Details

### Setup
```python
# Install required packages
!git clone https://github.com/speechbrain/speechbrain.git -b develop
!python -m pip install .
```

### Model Initialization
```python
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper

# Initialize models
model_w2v2 = Wav2Vec2("facebook/wav2vec2-base-960h", save_path='/content/pretrained/')
model_whisper = Whisper("openai/whisper-tiny", save_path='/content/pretrained/')
```

## Important Notes
1. Models are implemented as lobes in SpeechBrain:
   - `speechbrain.lobes.models.huggingface_wav2vec.py`
   - `speechbrain.lobes.models.huggingface_whisper.py`

2. Key Differences:
   - Wav2vec 2.0: Use encoder output directly
   - Whisper: Must specifically extract encoder output when using for feature extraction

3. Best Practices:
   - Models return standard PyTorch Modules
   - GPU runtime recommended for model loading
   - Input audio should be properly formatted (unsqueezed for batch processing)

## Prerequisites
- SpeechBrain Introduction
- YAML understanding
- Brain Class knowledge
- Data I/O Basics
- Understanding of pretrained models and fine-tuning

Here's the condensed version focusing on key implementation details and concepts:

# Using Wav2Vec 2.0 and Whisper Encoders in SpeechBrain

## Feature Extraction Basics

```python
# Configure encoder-only mode for Whisper
model_whisper.encoder_only = True
fea_whisper = model_whisper(source)
```

### Output Dimensions
- Wav2Vec2 Base model: 768 features
- Output frequency: 50Hz
- Shape format: [batch, time, features]
- Time dimension calculation: Based on audio length (e.g., 2.87s → 143 time steps)

## Integration into SpeechBrain Pipeline

### YAML Configuration

1. Wav2Vec2 Configuration:
```yaml
wav2vec2: !new:speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2
    source: !ref <wav2vec2_hub>
    freeze: True
    save_path: !ref <save_folder>/wav2vec2_checkpoint
```

2. Whisper Configuration:
```yaml
whisper: !new:speechbrain.lobes.models.huggingface_transformers.whisper.Whisper
    pretrained_path: !ref <wav2vec2_url>
    freeze: True
    encoder_only: True
    save_path: !ref <save_folder>/wav2vec2_checkpoint/model.pt
```

### Key Parameters
- `freeze`: Controls model fine-tuning
  - `True`: Parameters frozen
  - `False`: Enables fine-tuning
- `encoder_only`: For Whisper, extracts only encoder features
- Both models can be used as standard PyTorch layers in the pipeline

### Important Notes
- Requires basic SpeechBrain knowledge
- Can be integrated into existing ASR pipelines
- Reference implementation available in LibriSpeech ASR (CTC) recipe
- Supports on-the-fly feature computation and fine-tuning

This condensed version maintains all critical implementation details while removing redundant explanations and focusing on practical usage.

Here's the condensed tutorial focusing on key implementation details:

# Fine-tuning Pretrained Models (Whisper/Wav2vec2) Configuration Guide

## Key YAML Configuration Components

```yaml
# Core model configurations
wav2vec2_hub: "facebook/wav2vec2-large-960h-lv60-self"
whisper_hub: "openai/whisper-medium"
freeze_pretrained: False
lr_pretrained: 0.0001

# Model instantiation
wav2vec2: !new:speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2
  source: !ref <wav2vec2_hub>
  freeze: !ref <freeze_pretrained>

whisper: !new:speechbrain.lobes.models.huggingface_transformers.whisper.Whisper
  source: !ref <whisper_hub>
  freeze: !ref <freeze_pretrained>
  encoder_only: True

# Encoder configuration
enc: !new:speechbrain.lobes.models.VanillaNN.VanillaNN
  input_shape: [null, null, 1024]  # Matches output dim of LARGE wav2vec2/MEDIUM whisper
```

## Important Implementation Details

### 1. Dual Optimizer Setup
```yaml
# Separate optimizers for encoder/decoder and pretrained model
adam_opt_class: !name:torch.optim.AdamW
  lr: !ref <lr>

pretrained_opt_class: !name:torch.optim.AdamW
  lr: !ref <lr_pretrained>
```

### 2. Critical Python Implementation

```python
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        # Use either wav2vec2 or whisper
        feats = self.modules.wav2vec2(wavs)  # or
        feats = self.modules.whisper(wavs)
        x = self.modules.enc(feats)

    def init_optimizers(self):
        # Dual optimizer initialization
        self.pretrained_optimizer = self.hparams.pretrained_opt_class(
            self.modules.whisper.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

    def fit_batch(self, batch):
        # Handle both optimizers in training
        loss.backward()
        if self.check_gradients(loss):
            self.pretrained_optimizer.step()
            self.adam_optimizer.step()
        
        self.pretrained_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()
```

## Best Practices and Warnings

1. Only use dual optimizers when fine-tuning pretrained models; single optimizer suffices for frozen models
2. Ensure pretrained model output dimension (1024) matches encoder input dimension
3. Use appropriate learning rates for pretrained model fine-tuning (`lr_pretrained` typically lower than main `lr`)
4. Include pretrained models in `modules` list for GPU allocation but not in `model` list for separate optimization

Here's the condensed version focusing on key implementation details for using Whisper as a pre-trained encoder-decoder in SpeechBrain:

# Using Whisper as Full Pre-trained Encoder-Decoder

## Key Capabilities
- Full transformer support for zero-shot ASR/ST or fine-tuning
- Supported features:
  - Feature extraction
  - Encoder fine-tuning
  - Zero-shot ASR/ST
  - Encoder-decoder fine-tuning
  - Greedy and beam search decoding (with/without LM)

## Critical Configurations

```yaml
# Key YAML configurations
whisper_hub: "openai/whisper-medium"
freeze_pretrained: False
lr_pretrained: 0.0001
language: english

# Decoding parameters
timestamp_index: 50363  # First token for decoding
eos_index: 50257       # End of sequence token
max_decode_ratio: 0.5  # Max decoding steps ratio

# Whisper model configuration
whisper: !new:speechbrain.lobes.models.huggingface_transformers.whisper.Whisper
  source: !ref <whisper_hub>
  freeze: !ref <freeze_pretrained>
  encoder_only: False  # Important: Keep decoder
  save_path: !ref <save_folder>/whisper_checkpoint

# Searcher configuration
valid_greedy_searcher: !new:speechbrain.decoders.seq2seq.S2SWhisperGreedySearch
  model: !ref <whisper>
  bos_index: !ref <timestamp_index>
  eos_index: !ref <eos_index>
  min_decode_ratio: 0
  max_decode_ratio: !ref <max_decode_ratio>
```

## Implementation Details

```python
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        
        # Forward pass through encoder + decoder
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        log_probs = self.hparams.log_softmax(logits)
        
        # Greedy search during validation/test
        hyps = None
        if stage != sb.Stage.TRAIN:
            hyps, _ = self.hparams.valid_greedy_searcher(enc_out, wav_lens)
            
        return log_probs, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        log_probs, hyps, wav_lens = predictions
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        
        # NLL loss computation
        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, tokens_eos_lens,
        )
        
        # Evaluation metrics (WER/CER) during validation/test
        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens
            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )
            target_words = self.tokenizer.batch_decode(
                undo_padding(tokens, tokens_lens), 
                skip_special_tokens=True
            )
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            
        return loss
```

## Important Notes
- No separate DNN decoder needed
- No CTC loss required (uses negative log-likelihood)
- Language modeling head is automatically created
- Can be improved using beam search instead of greedy search
- Scalable to larger Whisper models

Here's the condensed version of the citation information:

## Citing SpeechBrain

When using SpeechBrain in research or business applications, cite using these references:

```bibtex
# For SpeechBrain 1.0 (Latest)
@misc{speechbrainV1,
  title={Open-Source Conversational AI with {SpeechBrain} 1.0},
  author={Ravanelli et al.},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2407.00463},
}


...(truncated)