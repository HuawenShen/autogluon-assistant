# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates the implementation of Wav2vec 2.0 and Whisper models within SpeechBrain, focusing on feature extraction, fine-tuning, and encoder-decoder architectures. It provides code examples for model initialization, configuration via YAML files, and integration into SpeechBrain pipelines. Key functionalities include using models as feature extractors, fine-tuning with separate optimizers, and implementing full encoder-decoder setups with Whisper. The tutorial covers essential parameters like freeze settings, encoder_only mode, and proper token indices configuration. It helps with tasks such as ASR model development, feature extraction, and model fine-tuning, while highlighting best practices for model selection, GPU usage, and input formatting.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and concepts:

# Using Wav2vec 2.0 and Whisper with SpeechBrain

## Key Points
- Integration of HuggingFace pretrained models (Wav2vec 2.0, Whisper, HuBERT, WavLM) with SpeechBrain
- Models can be used for feature extraction or fine-tuning
- Implemented as standard PyTorch modules in SpeechBrain lobes

## Implementation Details

### Setup
```python
# Install required packages
!pip install speechbrain
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper
```

### Model Initialization
```python
# Initialize models from HuggingFace hub
model_w2v2 = Wav2Vec2("facebook/wav2vec2-base-960h", save_path='pretrained/')
model_whisper = Whisper("openai/whisper-tiny", save_path='pretrained/')
```

## Important Considerations

1. Model Architecture Differences:
   - Wav2vec 2.0: Transformer encoder only
   - Whisper: Full transformer (encoder-decoder)

2. Feature Extraction:
   - Wav2vec 2.0: Direct output from last layer
   - Whisper: Must specifically extract encoder outputs

3. Model Locations:
   - Implementation in `speechbrain.lobes.models.huggingface_wav2vec.py`
   - Implementation in `speechbrain.lobes.models.huggingface_whisper.py`

## Best Practices
- Use GPU runtime for model loading and inference
- Ensure input audio is properly formatted (correct shape and sampling rate)
- Consider model size and computational requirements when selecting pretrained models

## Limitations
- Pre-training large SSL models not fully supported in SpeechBrain
- Resource requirements vary significantly between models

Here's the condensed version focusing on key implementation details and concepts:

# Using Wav2Vec 2.0 and Whisper Encoders in SpeechBrain

## Feature Extraction Basics

```python
# Configure encoder-only mode for Whisper
model_whisper.encoder_only = True
fea_whisper = model_whisper(source)
```

### Output Dimensions
- Wav2Vec 2.0 Base model: 768 features
- Output frequency: 50Hz
- Shape format: [batch, time, features]
- Example: For 2.87s audio → ~143 time steps

## Integration into SpeechBrain Pipeline

### Wav2Vec 2.0 YAML Configuration
```yaml
wav2vec2: !new:speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2
    source: !ref <wav2vec2_hub>
    freeze: True
    save_path: !ref <save_folder>/wav2vec2_checkpoint
```

### Whisper YAML Configuration
```yaml
whisper: !new:speechbrain.lobes.models.huggingface_transformers.whisper.Whisper
    pretrained_path: !ref <wav2vec2_url>
    freeze: True
    encoder_only: True
    save_path: !ref <save_folder>/wav2vec2_checkpoint/model.pt
```

### Key Parameters
- `freeze`: Controls model fine-tuning
  - `True`: Freezes parameters
  - `False`: Enables fine-tuning
- `encoder_only`: For Whisper, extracts only encoder features
- Both models can be used as standard PyTorch layers in the pipeline

### Important Notes
1. Models can be integrated directly into SpeechBrain pipelines for on-the-fly feature computation
2. Partial freezing is possible (encoder-only for Whisper, feature extractor for Wav2Vec 2.0)
3. Basic SpeechBrain knowledge required for implementation
4. Reference the LibriSpeech ASR (CTC) recipe for complete implementation examples

Here's the condensed tutorial focusing on key implementation details for fine-tuning Whisper or Wav2vec2 models:

# Fine-tuning Pretrained Models Configuration Guide

## Architecture Overview
```
[wav -> wav2vec2 or whisper -> Dense] = encoder
```

## Key YAML Configurations

### 1. Model Configuration
```yaml
# Pretrained model sources
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
  input_shape: [null, null, 1024]  # Matches LARGE wav2vec2 and MEDIUM whisper output
```

### 2. Optimizer Configuration
```yaml
# Dual optimizer setup for separate learning rates
adam_opt_class: !name:torch.optim.AdamW
  lr: !ref <lr>

pretrained_opt_class: !name:torch.optim.AdamW
  lr: !ref <lr_pretrained>
```

### 3. Module Management
```yaml
modules:
  wav2vec2: !ref <wav2vec2>
  whisper: !ref <whisper>
  enc: !ref <enc>
  # ... other modules

model: !new:torch.nn.ModuleList
  - [!ref <enc>, !ref <emb>, !ref <dec>, !ref <ctc_lin>, !ref <seq_lin>]
```

## Python Implementation

### Key Methods
```python
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        # Use either wav2vec2 or whisper
        feats = self.modules.wav2vec2(wavs)
        # OR
        feats = self.modules.whisper(wavs)
        x = self.modules.enc(feats)

    def init_optimizers(self):
        # Separate optimizers for pretrained and other modules
        self.pretrained_optimizer = self.hparams.pretrained_opt_class(
            self.modules.whisper.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

    def fit_batch(self, batch):
        # Handle both optimizers
        loss.backward()
        if self.check_gradients(loss):
            self.pretrained_optimizer.step()
            self.adam_optimizer.step()
        
        self.pretrained_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()
```

## Important Notes
- Use separate optimizers only when fine-tuning pretrained models
- Remove unused model references (wav2vec2/whisper) from configurations
- Learning rate scheduling is applied to both optimizers independently
- Model checkpointing includes both pretrained and custom components

Here's the condensed version focusing on key implementation details for using Whisper as a pre-trained encoder-decoder in SpeechBrain:

# Using Whisper as Full Pre-trained Encoder-Decoder

## Key Capabilities
- Full transformer support for zero-shot ASR/ST or fine-tuning
- Supported features:
  - Feature extraction
  - Encoder fine-tuning
  - Zero-shot ASR/ST
  - Full encoder-decoder fine-tuning
  - Greedy and beam search decoding (with/without LM)

## Critical Implementation Details

### YAML Configuration
```yaml
whisper_hub: "openai/whisper-medium"
freeze_pretrained: False
lr_pretrained: 0.0001

# Important decoder parameters
language: english
timestamp_index: 50363  # First token for decoding
eos_index: 50257       # End of sequence token
max_decode_ratio: 0.5  # Controls max decoding steps

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

### Python Implementation
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
        loss = self.hparams.nll_loss(log_probs, tokens_eos, tokens_eos_lens)
        
        # Evaluation metrics (WER/CER) during validation/test
        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens
            predicted_words = self.tokenizer.batch_decode(hyps, skip_special_tokens=True)
            target_words = self.tokenizer.batch_decode(
                undo_padding(tokens, tokens_lens), 
                skip_special_tokens=True
            )
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            
        return loss
```

## Important Notes
- No separate DNN decoder needed (Whisper includes one)
- No CTC loss required (uses negative log-likelihood)
- Can be improved using:
  - Beam search instead of greedy search
  - Larger Whisper models
  - Language model integration

## Best Practices
1. Set `encoder_only: False` to maintain decoder functionality
2. Configure proper token indices (timestamp_index, eos_index)
3. Adjust max_decode_ratio based on your needs
4. Consider using beam search for better performance
5. Monitor WER/CER metrics during validation

Here's the condensed version of the citation information:


...(truncated)