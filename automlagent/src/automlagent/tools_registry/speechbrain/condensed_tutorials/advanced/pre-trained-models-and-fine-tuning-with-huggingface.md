# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates implementing speech processing tasks using SpeechBrain and HuggingFace, covering ASR (Automatic Speech Recognition), speaker recognition, and source separation. It provides code for loading pre-trained models, performing inference, and fine-tuning models with custom data. Key implementations include transcription using CRDNN-based ASR, speaker embedding extraction with ECAPA-TDNN, audio source separation using SepFormer, and detailed fine-tuning procedures with the EncDecFineTune brain class. The tutorial showcases essential techniques like gradient management, module configuration, hyperparameter setup, and proper data pipeline construction, making it valuable for tasks involving speech processing model adaptation and deployment.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and code samples:

# Pretrained Models and Fine-Tuning with HuggingFace

## Key Concepts
- SpeechBrain provides pre-trained models via HuggingFace Hub
- Models can be used for inference or fine-tuning
- Official pre-trained models available at: huggingface.co/speechbrain

## Implementation Details

### Setup
```python
# Install SpeechBrain
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@develop
```

### ASR Implementation
```python
from speechbrain.inference.ASR import EncoderDecoderASR

# Load pretrained ASR model
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech", 
    savedir="./pretrained_ASR", 
    hparams_file="hyperparams_develop.yaml"
)

# Transcribe audio
transcription = asr_model.transcribe_file("path/to/audio.flac")
```

### Speaker Recognition Implementation
```python
from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio

# Load speaker recognition model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="./pretrained_ecapa"
)

# Extract embeddings
signal, fs = torchaudio.load('audio.flac')
embedding = verification.encode_batch(signal)
```

## Important Notes
1. Models are downloaded once and cached in the specified `savedir`
2. Same syntax works for both remote and local models
3. The ASR pipeline includes:
   - CRDNN-based seq2seq E2E ASR model
   - RNN-based Language Model
   - SentencePiece Tokenizer

## Best Practices
- Browse available models on HuggingFace Hub before implementation
- Validate transcriptions against oracle data when available
- Use PCA for visualizing speaker embeddings when working with speaker recognition

## Visualization Example
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce embedding dimensions for visualization
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(embeddings)

# Plot results
plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
```

Here's the condensed version focusing on key implementation details and concepts:

### Speaker Verification Testing
```python
# Test files from same/different speakers
file1 = './LibriSpeech/dev-clean-2/1272/135031/1272-135031-0000.flac' # Same speaker
file2 = './LibriSpeech/dev-clean-2/1272/141231/1272-141231-0004.flac' # Same speaker
file3 = './LibriSpeech/dev-clean-2/1462/170142/1462-170142-0000.flac' # Different speaker

# Verify files
score, prediction = verification.verify_files(file1, file2)  # Same speaker
score, prediction = verification.verify_files(file1, file3)  # Different speakers
```

### Source Separation Implementation
```python
# Create artificial mixture
import torchaudio
s1, fs = torchaudio.load('./LibriSpeech/dev-clean-2/1272/135031/1272-135031-0003.flac')
s2, fs = torchaudio.load('./LibriSpeech/dev-clean-2/1462/170142/1462-170142-0001.flac')

# Resample to 8KHz (model requirement)
resampler = torchaudio.transforms.Resample(fs, 8000)
s1, s2 = resampler(s1), resampler(s2)
fs = 8000

# Create mixture
min_len = min(s1.shape[-1], s2.shape[-1])
s1, s2 = s1[:, :min_len], s2[:, :min_len]
mix = s1 + s2

# Separate using SepFormer
from speechbrain.inference.separation import SepformerSeparation
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix", 
    savedir="./pretrained_sepformer"
)
est_sources = separator.separate_batch(mix)[0]  # Remove batch dimension
```

### Fine-tuning Pretrained Models
```python
# Load pretrained ASR model
from speechbrain.inference.ASR import EncoderDecoderASR
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech",
    savedir="./pretrained_ASR",
    hparams_file="hyperparams_develop.yaml"
)

# Data Pipeline Setup
from speechbrain.dataio.dataset import DynamicItemDataset
dataset = DynamicItemDataset.from_json("data.json")
dataset = dataset.filtered_sorted(sort_key="length", select_n=100)

# Add audio pipeline
dataset.add_dynamic_item(sb.dataio.dataio.read_audio, 
                        takes="file_path", 
                        provides="signal")

# Text processing pipeline
@sb.utils.data_pipeline.takes("words")
@sb.utils.data_pipeline.provides(
    "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
def text_pipeline(words):
    yield words
    tokens_list = asr_model.tokenizer.encode_as_ids(words)
    yield tokens_list
    tokens_bos = torch.LongTensor([asr_model.hparams.bos_index] + (tokens_list))
    yield tokens_bos
    tokens_eos = torch.LongTensor(tokens_list + [asr_model.hparams.eos_index])
    yield tokens_eos
    tokens = torch.LongTensor(tokens_list)
    yield tokens
```

Key Points:
1. SepFormer model requires 8KHz sampling rate
2. When fine-tuning, reuse pretrained model's tokenizer and special tokens (BOS/EOS)
3. Dataset is sorted by length for training efficiency
4. Access pretrained model components via `asr_model.mods.keys()`
5. Training hyperparameters accessible through `asr_model.hparams`

Here's the condensed version focusing on key implementation details and concepts:

# Fine-tuning Implementation Details

## EncDecFineTune Brain Class
```python
class EncDecFineTune(sb.Brain):
    def on_stage_start(self, stage, epoch):
        # Enable gradients for fine-tuning
        if stage == sb.Stage.TRAIN:
            for module in [self.modules.enc, self.modules.emb, 
                         self.modules.dec, self.modules.seq_lin]:
                for p in module.parameters():
                    p.requires_grad = True

    def compute_forward(self, batch, stage):
        # Forward pass implementation
        batch = batch.to(self.device)
        wavs, wav_lens = batch.signal
        tokens_bos, _ = batch.tokens_bos
        
        feats = self.modules.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)
        e_in = self.modules.emb(tokens_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)
        
        return p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        p_seq, wav_lens = predictions
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        return self.hparams.seq_cost(p_seq, tokens_eos, tokens_eos_lens)
```

## Module Configuration
```python
# Configure modules from pretrained model
modules = {
    "enc": asr_model.mods.encoder.model,
    "emb": asr_model.hparams.emb,
    "dec": asr_model.hparams.dec,
    "compute_features": asr_model.mods.encoder.compute_features,
    "normalize": asr_model.mods.encoder.normalize,
    "seq_lin": asr_model.hparams.seq_lin,
}

# Configure hyperparameters
hparams = {
    "seq_cost": lambda x, y, z: sb.nnet.losses.nll_loss(
        x, y, z, label_smoothing=0.1),
    "log_softmax": sb.nnet.activations.Softmax(apply_log=True)
}

# Initialize brain with SGD optimizer
brain = EncDecFineTune(
    modules, 
    hparams=hparams, 
    opt_class=lambda x: torch.optim.SGD(x, 1e-5)
)
brain.tokenizer = asr_model.tokenizer
```

## Pretrainer Usage
```python
# Initialize Pretrainer
pretrain = Pretrainer(
    loadables={'model': model},
    paths={'model': 'speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt'}
)
pretrain.collect_files()
pretrain.load_collected()
```

### Key Points:
1. The `EncDecFineTune` class handles the fine-tuning process with forward pass and loss computation
2. Modules are transferred from pretrained model using dictionary mapping
3. Hyperparameters include loss function with label smoothing and softmax activation
4. Pretrainer class provides structured parameter transfer with support for:
   - Local paths
   - Web URLs
   - HuggingFace repositories

### Best Practices:
- Enable gradients only for modules that need fine-tuning
- Use appropriate learning rate for fine-tuning (1e-5 in example)
- Maintain proper batch handling and device placement
- Use structured parameter transfer through Pretrainer for better reproducibility