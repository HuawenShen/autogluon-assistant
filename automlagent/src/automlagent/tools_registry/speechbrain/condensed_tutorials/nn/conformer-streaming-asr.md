# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial provides comprehensive implementation guidance for streaming speech recognition using Conformer models in SpeechBrain. It covers essential techniques for low-latency, long-form transcription including Dynamic Chunk Training, chunked attention mechanisms, and streaming-specific architectural modifications. The tutorial helps with implementing key components like streaming contexts, feature extraction, positional embeddings, and proper handling of dependencies between input/output frames. It details configuration of crucial parameters like chunk size and left context, along with best practices for training and inference. Notable features include support for various ASR recipes (LibriSpeech, VoxPopuli, CommonVoice), flexible streaming modes, and integration with different architectures like Branchformer and FastConformer. The implementation focuses on balancing latency and accuracy while maintaining consistency between training and inference paths.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details:

# Streaming Speech Recognition with Conformers

## Key Implementation Details

### Purpose
- Enables low-latency, long-form transcription for live stream applications
- Uses Dynamic Chunk Training approach for streaming Conformer models

### Core Requirements
1. Restrict attention to recent context only
2. Prevent access to future frames
3. Maintain consistency between training and inference

### Architecture Modifications

#### 1. Chunked Attention
- Preferred over causal attention for better WER
- Groups frames into chunks of size `chunk_size`
- Allows attention within chunks and to limited past chunks
- Controlled by parameters:
  - `chunk_size`: Number of frames per chunk
  - `left_context_chunks`: How many past chunks to attend to

```python
# Example: Creating chunk streaming mask
chunk_streaming_mask = make_transformer_src_mask(
    torch.empty(1, 16, 128), 
    dynchunktrain_config=DynChunkTrainConfig(4, 1)
)
```

#### Implementation Notes
- Training: Uses masking with full batches for GPU efficiency
- Inference: Processes chunk-by-chunk with state caching
- Attention mask shape: `(t, t)` boolean tensor
- Mask determines which frames can attend to which input frames

### Best Practices
1. Use chunked attention instead of causal attention for better performance
2. Consider trade-offs between chunk size and latency
3. Ensure consistent behavior between training and inference paths

### Available Recipes
- LibriSpeech/ASR/transducer
- VoxPopuli/ASR/transducer
- CommonVoice/ASR/transducer (French, Italian)

### Prerequisites
- Basic understanding of speech recognition
- Familiarity with SpeechBrain framework
- Python and PyTorch knowledge

This implementation serves as a foundation for building streaming ASR systems, though it may not represent state-of-the-art performance.

Here's the condensed version focusing on key implementation details and concepts:

# Streaming ASR Implementation Details

## Chunked Attention for Inference

### Key Concepts
- Careful management of dependencies between output and input frames across layers
- Chunk boundaries remain consistent across layers
- Left context size defined by number of chunks

### Implementation Details
```python
# Core attention processing at inference
attention_module(concat(cached_left_chunks, input_chunk))

# cached_left_chunks shape per layer:
# (batch_size, left_context_chunks * chunk_size, emb_dim)
```

**Best Practice**: Avoid depending on future frames across layers to maintain efficient streaming.

## Dynamic Chunk Convolution (DCC)

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
   - Caches `(k-1)/2` past frames per layer
   - Implementation reference: `speechbrain.lobes.models.transformer.Conformer.ConvolutionModule`

## Feature Extraction Considerations

### Key Points
- Non-causal feature extractor can be used
- Minimal right context needed (milliseconds scale)
- Use `StreamingFeatureWrapper` for handling context

```python
from speechbrain.lobes.features import StreamingFeatureWrapper
# Automatically handles padding and context caching
```

### Normalization
- Training vs. test discrepancy is minimal
- Per-chunk normalization acceptable in practice

**Important**: Feature extraction requires minimal future context and can be handled separately from the main streaming architecture.

This condensed version maintains the critical implementation details while removing explanatory text and redundant examples.

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
test_pos = test_pos_encoder.make_pe(seq_len=16)
# Shape: (batch, seq_len*2-1, emb_size)
```

## Dynamic Chunk Training

### Performance Impact
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
   - 40% normal training (no chunking)
   - 60% chunked training with:
     - Random chunk sizes (8-32 frames)
     - 75% with restricted left context
     - 25% with full left context

2. Loss Functions:
   - Primary: RNN-T loss
   - Optional: CTC as auxiliary loss
   - Potential future: encoder-decoder cross-entropy (currently untested)

### Important Notes
- Model can infer in both streaming and non-streaming modes
- Chunk size can be chosen at runtime
- Performance degradation vs non-streaming varies by hyperparameters and datasets
- Benchmark different chunk sizes on representative test data for your use case

This implementation allows for flexible streaming ASR with configurable latency-accuracy tradeoffs.

Here's the condensed version focusing on key implementation details for streaming architectures in SpeechBrain:

# Dynamic Chunk Training Implementation

## Key Components

### 1. Configuration Classes
- Use `DynChunkTrainConfig` to define streaming configuration for each batch
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

## Important Considerations

1. **Architecture Requirements**:
   - Use supported architectures (e.g., Conformer)
   - Set `TransformerASR`'s `causal` parameter to `False`

2. **Evaluation Settings**:
   - Only greedy search is supported in streaming context
   - Configure test set evaluation with greedy search
   - Optional: Specify `valid_config` or `test_config` for evaluation

3. **Chunk Sizing**:
   - Default chunk size ≈ 40ms of audio
   - Configurable through min/max parameters

4. **Debugging**:
   - Use `speechbrain.utils.streaming` for debugging functionality
   - Provides tools for streaming architecture verification

This implementation enables streaming capability with minimal code changes while maintaining flexibility in configuration.

Here's the condensed version focusing on key implementation details and concepts:

### Detecting Dependencies in Neural Network Layers

#### Key Tools
- `speechbrain.utils.streaming.infer_dependency_matrix`: Analyzes dependencies between input and output frames
- `speechbrain.utils.streaming.plot_dependency_matrix`: Visualizes the dependencies

#### Implementation Examples

1. **Non-causal Conformer**
```python
noncausal_model = TransformerASR(
    tgt_vocab=64, input_size=64, d_model=64, nhead=1, d_ffn=64, 
    encoder_module="conformer", normalize_before=True,
    attention_type="RelPosMHAXL",
    num_encoder_layers=4, num_decoder_layers=0,
    causal=False
)
noncausal_deps = infer_dependency_matrix(noncausal_model.encode, seq_shape=[1, 16, 64])
```

2. **Causal Conformer**
```python
causal_model = TransformerASR(
    # Same parameters as non-causal but with causal=True
    causal=True
)
```

3. **Chunked Conformer**
```python
chunked_model = TransformerASR(
    # Base parameters same as above
)
chunked_conf = DynChunkTrainConfig(chunk_size=4, left_context_size=1)
chunked_deps = infer_dependency_matrix(
    lambda x: chunked_model.encode(x, dynchunktrain_config=chunked_conf), 
    seq_shape=[1, 16, 64]
)
```

#### Feature Extractor for Inference

**Key Components:**
1. Filterbank extraction (Fourier transform-based)
2. Normalization (applied per-chunk during streaming)
3. Two downsampling CNNs (stride=2, reduces time dimension by 4)

**Important Considerations:**
- Chunk size must be defined at transformer level
- Proper handling of left/past and right/future context
- Input-output shape transformations must be tracked

**Example Configuration:**
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