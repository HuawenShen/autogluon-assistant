# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial covers SpeechBrain's text tokenization implementation using SentencePiece, focusing on BPE and Unigram tokenization methods. It provides code examples for tokenizer initialization, basic tokenization/detokenization operations, PyTorch integration, and data pipeline setup. Key functionalities include configurable vocabulary size, character coverage control, custom symbol inclusion, and both piece-based and ID-based encoding/decoding. The tutorial helps with tasks like implementing text preprocessing for speech recognition, managing large vocabularies, and integrating tokenization into SpeechBrain's data pipeline, making it particularly useful for speech-to-text applications.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# Text Tokenization in SpeechBrain

## Key Concepts
- Text tokenization helps manage large vocabularies while balancing between word-level and character-level representations
- SpeechBrain uses SentencePiece tokenizer with support for:
  - BPE (Byte Pair Encoding)
  - Unigram (Subword Regularization)

## Implementation Details

### Basic Setup
```python
from speechbrain.tokenizers.SentencePiece import SentencePiece

# Initialize tokenizer
spm = SentencePiece(
    model_dir="tokenizer_data",
    vocab_size=2000,  # Required for BPE & unigram
    annotation_train="dev-clean.csv",
    annotation_read="wrd",
    model_type="bpe",
    annotation_list_to_check=["dev-clean.csv"]
)
```

### Critical Parameters
- `model_dir`: Save directory for trained tokenizer model
- `vocab_size`: Vocabulary size (mandatory for BPE/unigram)
- `model_type`: "word", "char", "bpe", or "unigram"
- `character_coverage`: Character coverage ratio (0.98-1.0)
- `split_by_whitespace`: Controls cross-word piece extraction
- `user_defined_symbols`: Force specific vocabulary inclusion

### Usage Examples

1. Basic Tokenization/Detokenization:
```python
# Tokenize
pieces = spm.sp.encode_as_pieces('THIS IS A TEST')
ids = spm.sp.encode_as_ids('THIS IS A TEST')

# Detokenize
text_from_ids = spm.sp.decode_ids([244, 177, 3, 1, 97])
text_from_pieces = spm.sp.decode_pieces(['▁THIS', '▁IS', '▁A', '▁T', 'EST'])
```

2. PyTorch Integration:
```python
# Using with tensors
encoded_seq_ids, encoded_seq_lens = spm(
    wrd_tensor,
    lens_tensor,
    dict_int2lab,
    "encode"
)

# Decode
words_seq = spm(encoded_seq_ids, encoded_seq_lens, task="decode")
```

3. Integration with SpeechBrain's Data Pipeline:
```python
@sb.utils.data_pipeline.takes("wrd")
@sb.utils.data_pipeline.provides("wrd", "tokens_list", "tokens")
def text_pipeline(wrd):
    yield wrd
    tokens_list = spm.sp.encode_as_ids(wrd)
    yield tokens_list
    tokens = torch.LongTensor(tokens_list)
    yield tokens
```

## Best Practices
1. Set appropriate `character_coverage` (1.0 for small character sets, 0.995 for rich character sets like Japanese/Chinese)
2. Use `annotation_list_to_check` to verify tokenization accuracy
3. Consider vocabulary size carefully based on your application needs
4. For large datasets, use `num_sequences` to limit training data if needed

## Important Notes
- Pre-trained models can be loaded by specifying model path, vocab_size, and model_type
- The tokenizer integrates with SpeechBrain's data transform pipeline
- Supports both tensor-based and list-based operations for flexibility