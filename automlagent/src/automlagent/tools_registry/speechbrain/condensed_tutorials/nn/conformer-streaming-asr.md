# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides comprehensive implementation guidance for streaming speech recognition using Conformer models in SpeechBrain. It covers essential techniques for low-latency, long-form transcription including Dynamic Chunk Training, chunked attention mechanisms, and streaming-specific architectural modifications. The tutorial helps with implementing key components like chunked attention masks, dynamic chunk convolutions, positional embeddings, and streaming contexts. It details the configuration of StreamingASR components, feature extraction, and dependency management across neural network layers. Notable features include flexible chunk size configuration, context management between processing chunks, and integration with FFmpeg for live stream processing. The implementation supports both streaming and non-streaming modes, with configurable latency-accuracy tradeoffs through parameters like chunk size and left context values. The tutorial also includes debugging tools, best practices, and alternative architecture suggestions.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and concepts:

# Streaming Speech Recognition with Conformers

## Key Implementation Details

### Purpose
- Enables low-latency, long-form transcription for live stream applications
- Uses Dynamic Chunk Training approach for streaming capability

### Core Requirements
1. Restrict attention to recent context only
2. Prevent access to future frames
3. Maintain consistency between training and inference

### Architecture Modifications

#### 1. Chunked Attention
- Preferred over causal attention for better WER
- Groups frames into chunks of size `chunk_size`
- Key parameters:
  - `chunk_size`: Number of frames per chunk
  - `left_context_chunks`: Number of past chunks allowed for attention

```python
# Example: Creating chunked attention mask
chunk_streaming_mask = make_transformer_src_mask(
    torch.empty(1, 16, 128), 
    dynchunktrain_config=DynChunkTrainConfig(4, 1)
)
```

#### Implementation Notes
- Training: Uses masking with full batches for GPU efficiency
- Inference: Processes chunk-by-chunk with state caching
- Attention mask shape: `(t, t)` boolean tensor
- Frames within chunks can attend to each other
- Limited past context reduces computational/memory costs

### Best Practices
1. Use chunked attention instead of causal attention for better performance
2. Consider trade-offs between chunk size and latency
3. Ensure consistent behavior between training and inference paths

### Available Recipes
- LibriSpeech/ASR/transducer
- VoxPopuli/ASR/transducer
- CommonVoice/ASR/transducer (French, Italian)

### Prerequisites
- Basic understanding of speech recognition concepts
- Familiarity with SpeechBrain framework

This represents the first section of the tutorial, focusing on the fundamental architectural changes needed for streaming ASR with Conformers.

Here's the condensed version focusing on key implementation details and concepts:

# Streaming ASR Implementation Details

## Chunked Attention for Inference

### Key Concepts
- Careful management of dependencies between output and input frames across layers
- Chunk boundaries remain consistent across layers
- Left context allows chunks to attend to previous chunks without exploding memory requirements

### Implementation Details
```python
# Core attention processing at inference
attention_module(concat(cached_left_chunks, input_chunk))

# cached_left_chunks shape per layer:
# (batch_size, left_context_chunks * chunk_size, emb_dim)
```

## Dynamic Chunk Convolutions (DCC)

### Types of Convolutions
1. **Vanilla Convolutions**
   - Window spans: `t-((k-1)/2)` to `t+((k-1)/2)`
   - Not suitable for streaming due to future frame dependencies

2. **Causal Convolutions**
   - Window spans: `t-(k-1)` to `t`
   - Implementation:
     - Left-pad input by `(k-1)/2` frames
     - Apply convolution
     - Truncate `(k-1)/2` output frames on left

3. **Dynamic Chunk Convolutions**
   - Masks off future chunk frames
   - Maintains chunk boundaries from attention
   - Requires caching `(k-1)/2` past frames per layer at inference

### Implementation Reference
- See `speechbrain.lobes.models.transformer.Conformer.ConvolutionModule` for detailed implementation

## Unchanged Components

### Feature Extraction
- Non-causal implementation retained
- Minimal right context needed (milliseconds)
- Use `StreamingFeatureWrapper` for handling context:
```python
from speechbrain.lobes.features import StreamingFeatureWrapper
```

### Normalization
- Training/test discrepancy between full audio and per-chunk normalization is minimal
- Can be left as-is for practical implementations

### Best Practices
1. Maintain consistent chunk boundaries across layers
2. Cache only necessary context for both attention and convolution
3. Consider feature extraction right context carefully
4. Use provided wrappers for streaming features when possible

This implementation enables efficient streaming ASR while maintaining model accuracy through careful management of context and dependencies.

Here's the condensed version focusing on key implementation details and concepts:

# Streaming ASR Implementation Details

## Positional Embeddings

### Key Implementation Points
- Uses relative positional sinusoidal encodings (`speechbrain.nnet.attention.RelPosEncXL`)
- Calculates position based on difference between query and key positions
- Maximum distance limited by attention window size
- Works well for streaming as distances remain constant regardless of stream length

```python
# Example of relative positional encoding
from speechbrain.nnet.attention import RelPosEncXL

test_pos_encoder = RelPosEncXL(64)
test_pos = test_pos_encoder.make_pos(seq_len=16)
# Shape: (batch, seq_len*2-1, emb_size)
```

## Dynamic Chunk Training

### Performance Impacts
- Smaller chunks: Lower latency but worse accuracy
- Larger left context: Better accuracy but higher computational cost

### Training Strategy
```python
# Dynamic chunk training configuration
sampler = DynChunkTrainConfigRandomSampler(
    chunkwise_prob=0.6,          # 60% chunks, 40% normal training
    chunk_size_min=8,
    chunk_size_max=32,
    limited_left_context_prob=0.8,
    left_context_chunks_min=2,
    left_context_chunks_max=16,
    test_config=DynChunkTrainConfig(32, 16),
    valid_config=None
)
```

### Best Practices
1. Random chunk size sampling during training:
   - 40% normal training (no chunks)
   - 60% chunked training with:
     - Random chunk sizes (8-32 frames)
     - 75% with limited left context (2-32 chunks)
     - 25% with full left context

2. Loss Functions:
   - Primary: RNN-T loss
   - Optional: CTC as auxiliary loss
   - Potential future exploration: encoder-decoder cross-entropy

### Important Notes
- Model can infer in both streaming and non-streaming modes
- Chunk size can be chosen at runtime
- Performance degradation vs non-streaming varies by hyperparameters and datasets
- Test different chunk sizes and left context values for your specific use case

This implementation allows for flexible streaming ASR with configurable latency-accuracy tradeoffs.

Here's the condensed version focusing on key implementation details for streaming architectures in SpeechBrain:

# Dynamic Chunk Training Implementation

## Key Components

### 1. Configuration Classes
- Uses `DynChunkTrainConfig` for describing streaming configuration per batch
- `DynChunkTrainConfigRandomSampler` for random configuration sampling during training

### 2. YAML Configuration

```yaml
# Basic streaming setup
streaming: True

# Dynamic Chunk Training Configuration
dynchunktrain_config_sampler: !new:speechbrain.utils.dynamic_chunk_training.DynChunkTrainConfigRandomSampler
   chunkwise_prob: 0.6
   chunk_size_min: 8
   chunk_size_max: 32
   limited_left_context_prob: 0.75
   left_context_chunks_min: 2
   left_context_chunks_max: 32
```

### 3. Training Script Implementation

```python
# In compute_forward
if self.hparams.streaming:
    dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(stage)
else:
    dynchunktrain_config = None

# Encoder call
x = self.modules.enc(
    src,
    dynchunktrain_config=dynchunktrain_config,
)
```

## Important Notes and Best Practices

1. Architecture Requirements:
   - Use supported architectures (e.g., Conformer)
   - Set `TransformerASR`'s `causal` parameter to `False`

2. Evaluation Considerations:
   - Only greedy search is currently supported in streaming mode
   - Configure test set evaluation with greedy search
   - Optional: Specify `valid_config` or `test_config` for streaming emulation during evaluation

3. Chunk Sizing:
   - Default chunk size is approximately 40ms of audio
   - Configurable through min/max parameters in sampler configuration

4. Debugging:
   - Use `speechbrain.utils.streaming` module for debugging features
   - Provides utilities for streaming architecture validation

This implementation enables streaming capability with minimal code changes while maintaining flexibility in configuration.

Here's the condensed version focusing on key implementation details and concepts:

# Detecting Dependencies in Neural Network Layers

## Dependency Detection Tools
- Use `speechbrain.utils.streaming.infer_dependency_matrix` to analyze frame dependencies
- Visualize using `speechbrain.utils.streaming.plot_dependency_matrix`
- Red cells indicate output dependency on input frames
- Note: Tool may show false negatives on larger models

## Key Code Examples

### 1. Non-causal Conformer
```python
noncausal_model = TransformerASR(
    tgt_vocab=64, input_size=64, d_model=64, nhead=1, d_ffn=64, 
    encoder_module="conformer", normalize_before=True,
    attention_type="RelPosMHAXL",
    num_encoder_layers=4, num_decoder_layers=0,
    causal=False
)
```

### 2. Causal Conformer
```python
causal_model = TransformerASR(
    # Same parameters as non-causal but with causal=True
    causal=True
)
```

### 3. Chunked Conformer
```python
chunked_model = TransformerASR(
    # Base parameters same as above
)
chunked_conf = DynChunkTrainConfig(chunk_size=4, left_context_size=1)
```

## Feature Extractor Components
1. Filterbank extraction (Fourier transform-based)
2. Normalization (applied per-chunk during streaming)
3. Two downsampling CNNs (stride=2, reduces time dimension by 4x)

## Important Considerations
- Chunk size must be defined at transformer level
- Need precise understanding of feature extractor's shape transformations
- Must handle left/past and right/future context correctly
- Input stride is typically 16 (1ms at 16kHz)

## Feature Extractor Configuration Example
```python
feat_extractor_hparams = {
    "compute_features": Fbank(
        sample_rate=16000,
        n_fft=512,
        n_mels=80,
        win_length=32
    ),
    "cnn": ConvolutionFrontEnd(
        input_shape=(8, 10, 80),
        num_blocks=2,
        num_layers_per_block=1,
        out_channels=(64, 32),
        kernel_sizes=(3, 3),
        strides=(2, 2)
    )
}
```


...(truncated)