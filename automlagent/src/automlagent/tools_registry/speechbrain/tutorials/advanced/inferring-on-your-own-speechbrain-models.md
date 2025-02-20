Summary: This tutorial covers implementation techniques for inference with SpeechBrain models, focusing on three main approaches: custom functions in training scripts, pre-built interfaces like EncoderDecoderASR, and custom interfaces. It provides code examples and configurations for model initialization, inference loops, and decoder parameters. Key functionalities include audio transcription, feature processing, and integration with external interfaces via foreign_class. The tutorial details essential components like encoders, decoders, scoring mechanisms, and YAML configurations, while emphasizing best practices such as using torch.no_grad() during inference and proper model evaluation mode. It's particularly useful for tasks involving ASR model deployment, custom inference pipeline development, and integrating SpeechBrain models into external applications.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/inferring-on-your-own-speechbrain-models.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/advanced/inferring-on-your-own-speechbrain-models.ipynb)

# Inferring on your trained SpeechBrain model

In this tutorial, we will learn the different ways of inferring on a trained model. Please understand that this is not related to loading pretrained models for further training or transfer learning. If interested in these topics, refer to the corresponding [tutorial](https://colab.research.google.com/drive/1LN7R3U3xneDgDRK2gC5MzGkLysCWxuC3?usp=sharing).

## Prerequisites
- [SpeechBrain Introduction](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/introduction-to-speechbrain.html)
- [YAML tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/hyperpyyaml.html)
- [Brain Class tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/brain-class.html)
- [Pretraining tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html
)

## Context

In this example, we will consider a user that would like to use a custom pretrained speech recognizer **that has been trained by him** to transcribe some audio files. If you are interested in using online-available pretrained models, please refer to the [Pretraining tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html
). The following can be extended to any SpeechBrain supported task as we provide an homogeneous way of dealing with all of them.

## Different options available

At this point, three options are available to you:
1. Define a custom python function in your ASR class (extended from Brain). This introduces strong coupling between the training recipe and your transcripts. It is pretty convenient for prototyping and obtaining simple transcripts on your datasets. However, it is not recommended for deployment.
2. Use already available Interfaces (such as `EncoderDecoderASR`, introduction in the [pretraining tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html
)). This is probably the most elegant and convenient way. However, your model should be compliant with some constraints to fit the proposed interface.
3. Build your own Interface perfectly fitting to your custom ASR model.

**Important: All these solutions also apply to other tasks (speaker recognition, source separation ...)**

### 1. Custom function in the training script
The goal of this approach is to enable the user to call a function at the end of `train.py` that transcribes a given dataset:

```python
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Load best checkpoint for transcription !!!!!!
    # You need to create this function w.r.t your system architecture !!!!!!
    transcripts = asr_brain.transcribe_dataset(
        dataset=datasets["your_dataset"], # Must be obtained from the dataio_function
        min_key="WER", # We load the model with the lowest WER
        loader_kwargs=hparams["transcribe_dataloader_opts"], # opts for the dataloading
    )
```

As you can see, there exists a strong coupling with the training recipe due to the need for an instantiated Brain class.

**Note 1:** You can remove the `.fit()` and `.evaluate()` if you don't want to call them. This is just an example to better highlight how to use it.

**Note 2:** Here, the `.transcribe_dataset()` function takes a `dataset` object to transcribe. You could also simply use a path instead. It is **completely** up to you to implement this function as you wish.

Now: what to put in this function? Here, we will give an example based on the template, but you will need to adapt it to **your** system.

```python

def transcribe_dataset(
        self,
        dataset, # Must be obtained from the dataio_function
        min_key, # We load the model with the lowest WER
        loader_kwargs # opts for the dataloading
    ):
  
    # If dataset isn't a Dataloader, we create it.
    if not isinstance(dataset, DataLoader):
        loader_kwargs["ckpt_prefix"] = None
        dataset = self.make_dataloader(
            dataset, Stage.TEST, **loader_kwargs
        )
    
    
    self.on_evaluate_start(min_key=min_key) # We call the on_evaluate_start that will load the best model
    self.modules.eval() # We set the model to eval mode (remove dropout etc)

    # Now we iterate over the dataset and we simply compute_forward and decode
    with torch.no_grad():

        transcripts = []
        for batch in tqdm(dataset, dynamic_ncols=True):
            
            # Make sure that your compute_forward returns the predictions !!!
            # In the case of the template, when stage = TEST, a beam search is applied
            # in compute_forward().
            out = self.compute_forward(batch, stage=sb.Stage.TEST)
            p_seq, wav_lens, predicted_tokens = out
            
            # We go from tokens to words.
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )
            transcripts.append(predicted_words)
            
    return transcripts
```

The pipeline is simple: load the model -> do compute_forward -> detokenize.

### 2. Using the `EndoderDecoderASR` interface

The [EncoderDecoderASR class](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/ASR.py). interface allows you to decouple your trained model from the training recipe and to infer (or encode) on any new audio file in few lines of code. If you are not interested in ASR, you'll find many other interfaces to fit your purpose in the `interfaces.py` file. This solution must be preferred if you intend to deploy your model in a production fashion i.e. if you plan to use your model a lot and in a stable way. Of course, this will require you to slightly rework the yaml.

The class has the following methods:

- *encode_batch*: apply the encoder to an input batch and returns some encoded features.
- *transcribe_file*: transcribes the single audio file in input.
- *transcribe_batch*: transcribes the input batch.

In fact, if you fulfill few constraints that we will detail in the next paragraph, you can simply do:

```python
from speechbrain.inference.ASR import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="your_folder", hparams_file='your_file.yaml', savedir="pretrained_model")
asr_model.transcribe_file('your_file.wav')
```

Nevertheless, to allow such a generalization over all the possible EncoderDecoder ASR pipelines, you will have to consider a few constraints when deploying your system:

1. **Necessary modules.** As you can see in the `EncoderDecoderASR` class, the modules defined in your yaml file MUST contain certain elements with specific names. In practice, you need a tokenizer, a decoder, and a decoder. The encoder can simply be a `speechbrain.nnet.containers.LengthsCapableSequential` composed with a sequence of features computation, normalization and model encoding.
```python
    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = [
        "encoder",
        "decoder",
    ]
```

You also need to declare these entities in the YAML file and create the following dictionary called `modules`:

```yaml
encoder: !new:speechbrain.nnet.containers.LengthsCapableSequential
    input_shape: [null, null, !ref <n_mels>]
    compute_features: !ref <compute_features>
    normalize: !ref <normalize>
    model: !ref <enc>

ctc_scorer: !new:speechbrain.decoders.scorer.CTCScorer
    eos_index: !ref <eos_index>
    blank_index: !ref <blank_index>
    ctc_fc: !ref <ctc_lin>

coverage_scorer: !new:speechbrain.decoders.scorer.CoverageScorer
    vocab_size: !ref <output_neurons>

rnnlm_scorer: !new:speechbrain.decoders.scorer.RNNLMScorer
    language_model: !ref <lm_model>
    temperature: !ref <temperature_lm>

scorer: !new:speechbrain.decoders.scorer.ScorerBuilder
    scorer_beam_scale: 1.5
    full_scorers: [
        !ref <rnnlm_scorer>,
        !ref <coverage_scorer>]
    partial_scorers: [!ref <ctc_scorer>]
    weights:
        rnnlm: !ref <lm_weight>
        coverage: !ref <coverage_penalty>
        ctc: !ref <ctc_weight_decode>

decoder: !new:speechbrain.decoders.S2SRNNBeamSearcher
    embedding: !ref <emb>
    decoder: !ref <dec>
    linear: !ref <seq_lin>
    bos_index: !ref <bos_index>
    eos_index: !ref <eos_index>
    min_decode_ratio: !ref <min_decode_ratio>
    max_decode_ratio: !ref <max_decode_ratio>
    beam_size: !ref <test_beam_size>
    eos_threshold: !ref <eos_threshold>
    using_max_attn_shift: !ref <using_max_attn_shift>
    max_attn_shift: !ref <max_attn_shift>
    temperature: !ref <temperature>
    scorer: !ref <scorer>

modules:
    encoder: !ref <encoder>
    decoder: !ref <decoder>
    lm_model: !ref <lm_model>
```

In this case, `enc` is a CRDNN, but could be any custom neural network for instance.

  **Why do you need to ensure this?** Well, it simply is because these are the modules we call when inferring on the `EncoderDecoderASR` class. Here is an example of the `encode_batch()` function.
```python
[...]
  wavs = wavs.float()
  wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
  encoder_out = self.modules.encoder(wavs, wav_lens)
return encoder_out
```
  **What if I have a complex asr_encoder structure with multiple deep neural networks and stuffs ?** Simply put everything in a torch.nn.ModuleList in your yaml:
```yaml
asr_encoder: !new:torch.nn.ModuleList
    - [!ref <enc>, my_different_blocks ... ]
```

2. **Call to the pretrainer to load the checkpoints.** Finally, you need to define a call to the pretrainer that will load the different checkpoints of your trained model into the corresponding SpeechBrain modules. In short, it will load the weights of your encoder, language model or even simply load the tokenizer.
```yaml
pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        asr: !ref <asr_model>
        lm: !ref <lm_model>
        tokenizer: !ref <tokenizer>
    paths:
      asr: !ref <asr_model_ptfile>
      lm: !ref <lm_model_ptfile>
      tokenizer: !ref <tokenizer_ptfile>
```
The loadable field creates a link between a file (e.g. `lm` that is related to the checkpoint in `<lm_model_ptfile>`) to a yaml instance (e.g. `<lm_model>`) that is nothing more than your lm.

If you respect these two constraints, it should works! Here, we give a complete example of a yaml that is used for inference only:

```yaml

# ############################################################################
# Model: E2E ASR with attention-based ASR
# Encoder: CRDNN model
# Decoder: GRU + beamsearch + RNNLM
# Tokens: BPE with unigram
# Authors:  Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, Peter Plantinga 2020
# ############################################################################


# Feature parameters
sample_rate: 16000
n_fft: 400
n_mels: 40

# Model parameters
activation: !name:torch.nn.LeakyReLU
dropout: 0.15
cnn_blocks: 2
cnn_channels: (128, 256)
inter_layer_pooling_size: (2, 2)
cnn_kernelsize: (3, 3)
time_pooling_size: 4
rnn_class: !name:speechbrain.nnet.RNN.LSTM
rnn_layers: 4
rnn_neurons: 1024
rnn_bidirectional: True
dnn_blocks: 2
dnn_neurons: 512
emb_size: 128
dec_neurons: 1024
output_neurons: 1000  # index(blank/eos/bos) = 0
blank_index: 0

# Decoding parameters
bos_index: 0
eos_index: 0
min_decode_ratio: 0.0
max_decode_ratio: 1.0
beam_size: 80
eos_threshold: 1.5
using_max_attn_shift: True
max_attn_shift: 240
lm_weight: 0.50
coverage_penalty: 1.5
temperature: 1.25
temperature_lm: 1.25

normalize: !new:speechbrain.processing.features.InputNormalization
    norm_type: global

compute_features: !new:speechbrain.lobes.features.Fbank
    sample_rate: !ref <sample_rate>
    n_fft: !ref <n_fft>
    n_mels: !ref <n_mels>

enc: !new:speechbrain.lobes.models.CRDNN.CRDNN
    input_shape: [null, null, !ref <n_mels>]
    activation: !ref <activation>
    dropout: !ref <dropout>
    cnn_blocks: !ref <cnn_blocks>
    cnn_channels: !ref <cnn_channels>
    cnn_kernelsize: !ref <cnn_kernelsize>
    inter_layer_pooling_size: !ref <inter_layer_pooling_size>
    time_pooling: True
    using_2d_pooling: False
    time_pooling_size: !ref <time_pooling_size>
    rnn_class: !ref <rnn_class>
    rnn_layers: !ref <rnn_layers>
    rnn_neurons: !ref <rnn_neurons>
    rnn_bidirectional: !ref <rnn_bidirectional>
    rnn_re_init: True
    dnn_blocks: !ref <dnn_blocks>
    dnn_neurons: !ref <dnn_neurons>

emb: !new:speechbrain.nnet.embedding.Embedding
    num_embeddings: !ref <output_neurons>
    embedding_dim: !ref <emb_size>

dec: !new:speechbrain.nnet.RNN.AttentionalRNNDecoder
    enc_dim: !ref <dnn_neurons>
    input_size: !ref <emb_size>
    rnn_type: gru
    attn_type: location
    hidden_size: !ref <dec_neurons>
    attn_dim: 1024
    num_layers: 1
    scaling: 1.0
    channels: 10
    kernel_size: 100
    re_init: True
    dropout: !ref <dropout>

ctc_lin: !new:speechbrain.nnet.linear.Linear
    input_size: !ref <dnn_neurons>
    n_neurons: !ref <output_neurons>

seq_lin: !new:speechbrain.nnet.linear.Linear
    input_size: !ref <dec_neurons>
    n_neurons: !ref <output_neurons>

log_softmax: !new:speechbrain.nnet.activations.Softmax
    apply_log: True

lm_model: !new:speechbrain.lobes.models.RNNLM.RNNLM
    output_neurons: !ref <output_neurons>
    embedding_dim: !ref <emb_size>
    activation: !name:torch.nn.LeakyReLU
    dropout: 0.0
    rnn_layers: 2
    rnn_neurons: 2048
    dnn_blocks: 1
    dnn_neurons: 512
    return_hidden: True  # For inference

tokenizer: !new:sentencepiece.SentencePieceProcessor

asr_model: !new:torch.nn.ModuleList
    - [!ref <enc>, !ref <emb>, !ref <dec>, !ref <ctc_lin>, !ref <seq_lin>]

# We compose the inference (encoder) pipeline.
encoder: !new:speechbrain.nnet.containers.LengthsCapableSequential
    input_shape: [null, null, !ref <n_mels>]
    compute_features: !ref <compute_features>
    normalize: !ref <normalize>
    model: !ref <enc>


ctc_scorer: !new:speechbrain.decoders.scorer.CTCScorer
    eos_index: !ref <eos_index>
    blank_index: !ref <blank_index>
    ctc_fc: !ref <ctc_lin>

coverage_scorer: !new:speechbrain.decoders.scorer.CoverageScorer
    vocab_size: !ref <output_neurons>

rnnlm_scorer: !new:speechbrain.decoders.scorer.RNNLMScorer
    language_model: !ref <lm_model>
    temperature: !ref <temperature_lm>

scorer: !new:speechbrain.decoders.scorer.ScorerBuilder
    scorer_beam_scale: 1.5
    full_scorers: [
        !ref <rnnlm_scorer>,
        !ref <coverage_scorer>]
    partial_scorers: [!ref <ctc_scorer>]
    weights:
        rnnlm: !ref <lm_weight>
        coverage: !ref <coverage_penalty>
        ctc: !ref <ctc_weight_decode>

decoder: !new:speechbrain.decoders.S2SRNNBeamSearcher
    embedding: !ref <emb>
    decoder: !ref <dec>
    linear: !ref <seq_lin>
    bos_index: !ref <bos_index>
    eos_index: !ref <eos_index>
    min_decode_ratio: !ref <min_decode_ratio>
    max_decode_ratio: !ref <max_decode_ratio>
    beam_size: !ref <test_beam_size>
    eos_threshold: !ref <eos_threshold>
    using_max_attn_shift: !ref <using_max_attn_shift>
    max_attn_shift: !ref <max_attn_shift>
    temperature: !ref <temperature>
    scorer: !ref <scorer>
    

modules:
    encoder: !ref <encoder>
    decoder: !ref <decoder>
    lm_model: !ref <lm_model>

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        asr: !ref <asr_model>
        lm: !ref <lm_model>
        tokenizer: !ref <tokenizer>


```

As you can see, it is a standard YAMl file, but with a pretrainer that loads the model. It is similar to the yaml file used for training. We only have to remove all the parts that are training-specific (e.g, training parameters, optimizers, checkpointers, etc.) and add the pretrainer and `encoder`, `decoder` elements that links the needed modules with their pre-trained files.

### 3. Developing your own inference interface

While the `EncoderDecoderASR` class has been designed to be as generic as possible, your might require a more complex inference scheme that better fits your needs.  In this case, you have to develop your own interface. To do so, follow these steps:

1. Create your custom interface inheriting from `Pretrained` (code [in this file](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/interfaces.py)):


```python
class MySuperTask(Pretrained):
  # Here, do not hesitate to also add some required modules
  # for further transparency.
  HPARAMS_NEEDED = ["mymodule1", "mymodule2"]
  MODULES_NEEDED = [
        "mytask_enc",
        "my_searcher",
  ]
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Do whatever is needed here w.r.t your system
```

This will enable your class to call useful functions such as `.from_hparams()` that fetches and loads based on a HyperPyYAML file, `load_audio()` that loads a given audio file.  Likely, most of the methods that we coded in the Pretrained class will fit your need. If not, you can override them to implement your custom functionality.


2. Develop your interface and the different functionalities. Unfortunately, we can't provide a generic enough example here. You can add **any** function to this class that you think can make inference on your data/model easier and natural. For instance, we can create here a function that simply encodes a wav file using the `mytask_enc` module.
```python
class MySuperTask(Pretrained):
  # Here, do not hesitate to also add some required modules
  # for further transparency.
  HPARAMS_NEEDED = ["mymodule1", "mymodule2"]
  MODULES_NEEDED = [
        "mytask_enc",
        "my_searcher",
  ]
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Do whatever is needed here w.r.t your system
  
  def encode_file(self, path):
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        with torch.no_grad():
          rel_lens = rel_length.to(self.device)
          encoder_out = self.encode_batch(waveform, rel_lens)
        
        return encode_file
```

Now, we can use your Interface in the following way:
```python
from speechbrain.inference.my_super_task import MySuperTask

my_model = MySuperTask.from_hparams(source="your_local_folder", hparams_file='your_file.yaml', savedir="pretrained_model")
audio_file = 'your_file.wav'
encoded = my_model.encode_file(audio_file)

```

As you can see, this formalism is extremely flexible and enables you to create a holistic interface that can be used to do anything you want with your pretrained model.

We provide different generic interfaces for E2E ASR, speaker recognition, source separation, speech enhancement, etc. Please have a look [here](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/inference) if interested!


## General Pretraining Inference
In some cases, users might want to develop their inference interface in an external file. This can be done using the [foreign class](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/interfaces.py).
You can take a look at the example reported [here](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP):



```
from speechbrain.inference.interfaces import foreign_class
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
print(text_lab)
```



In this case, the inference interface is not a class written in `speechbrain.pretrained.interfaces`, but it is coded in an external file (`custom_interface.py`).

This might be useful if the interface that you need is not available in `speechbrain.pretrained.interfaces`. If you want, you can add it there. If you use the foreign_class, however,  we also give you the possibility to fetch the inference code from any other path.



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
