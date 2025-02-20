Summary: This tutorial demonstrates implementing neural network adapters for efficient model fine-tuning, focusing on LoRA and custom adapter architectures in SpeechBrain. It covers configuration techniques for selective layer adaptation using YAML configs, manual adapter insertion procedures, and implementation of custom adapters like PoolLoRA. The tutorial helps with tasks involving memory-efficient model fine-tuning, reducing trainable parameters to ~1% while maintaining performance. Key features include configurable adapter ranks, layer freezing controls, integration with PEFT library, and custom adapter development with detailed parameter management and scaling mechanisms.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/neural-network-adapters.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/nn/neural-network-adapters.ipynb)

# Neural Network Adapters for faster low-memory fine-tuning

This tutorial covers the SpeechBrain implementation of adapters such as LoRA. This includes how to integrate either SpeechBrain implemented adapters, custom adapters, and adapters from libraries such as PEFT into a pre-trained model.

## Prerequisites
- [Speech Recognition From Scratch](https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-recognition-from-scratch.html)
- A CUDA-enabled device for running this tutorial

## Introduction and Background

As pre-trained models become larger and more capable, there is growing interest in methods for adapting them for specific tasks in a memory-efficient way, within a reasonable time span. One such technique is freezing the original parameters and inserting a small number of additional parameters into the original model, which are called "adapters." These adapters can often match the performance of full fine-tuning at a fraction of the parameter count, meaning faster and more memory-efficient fine-tuning [1]. One popular technique for doing this is known as Low-Rank Adaptation (LoRA) [2].

On the software side, HuggingFace has produced a popular library for adapters called PEFT [3]. Our implementation includes some of the features of this library, as well as including the ability to integrate PEFT adapters into a SpeechBrain model. We'll start with a basic example YAML so you can just try it yourself if you learn better from experimentation.

### Relevant bibliography
1. N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, "Parameter-efficient transfer learning for NLP." In *International Conference on Machine Learning*, 2019.
2. E.J. Hu, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models." In *International Conference on Learning Representations*, 2021.
3. S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and  B. Bossan, "PEFT: State-of-the-art parameter-efficient fine-tuning methods." *GitHub Repository*, 2022.


## Too Long ; Didn't Read

The TL;DR of this tutorial is that you should use a section like this in your HyperPyYAML file to create a model with adapters:

```yaml
adapted_model: !new:speechbrain.nnet.adapters.AdaptedModel
    model_to_adapt: !ref <model>
    adapter_class: !name:speechbrain.nnet.adapters.LoRA
    all_linear: True
    unfrozen_layers: ["conv_1d_*"]
    adapter_kwargs:
        rank: 8
```

Adding this section to the YAML takes the already-defined key `model` and adds a LoRA adapter to every linear layer with the keyword argument `rank=8`.
The `all_linear` and `all_conv` arguments simply add adapters to all linear or all convolution layers respectively.
By default, this class freezes the parameters of all layers that are not adapted, but the `unfrozen_layers` argument can be used to specify the names
of layers that should also be trained, at the cost of a higher parameter count. One can specify specific layers that should be adapted with
the `target_layers` argument. These arguments both support unix-style wildcards through the use of python's `fnmatch` library.

If the TL;DR isn't enough and you need to work through an example in more detail, continue to the next section.

## Detailed Tutorial

We'll demonstrate how to use adapters on a template recipe, which includes everything necessary for full training. The first step is to pretrain a model so we can later add adapters.


```python
!git clone --depth 1 --branch v1.0.2 https://github.com/speechbrain/speechbrain.git
!python -m pip install -e speechbrain
```


```python
# In order to use speechbrain in this repo we have to add it to the path
import os, sys

sys.path.append(os.path.join(os.getcwd(), 'speechbrain'))
```


```python
%cd speechbrain/templates/speech_recognition/ASR
```


```python
!python train.py train.yaml --number_of_epochs=1 --batch_size=2 --test_scorer "!ref <valid_scorer>" --enable_add_reverb=False --enable_add_noise=False #To speed up
```

## Inference

To prove that this is working, let's just perform inference on one file. This code taken from `transcribe_file.py`


```python
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.utils.fetching import fetch, LocalStrategy

# Ensure all the needed files end up in the same place to load with the transcriber
save_dir = os.path.abspath("results/CRDNN_BPE_960h_LM/2602/save/CKPT+latest")
fetch("lm.ckpt", "speechbrain/asr-crdnn-rnnlm-librispeech", save_dir, local_strategy=LocalStrategy.SYMLINK)
fetch("tokenizer.ckpt", "speechbrain/asr-crdnn-rnnlm-librispeech", save_dir, local_strategy=LocalStrategy.SYMLINK)
fetch("inference.yaml", os.getcwd(), save_dir, local_strategy=LocalStrategy.SYMLINK)

transcriber = EncoderDecoderASR.from_hparams(source=save_dir, hparams_file="inference.yaml")
speech_file = "../data/LibriSpeech/dev-clean-2/1272/135031/1272-135031-0015.flac"
transcriber.transcribe_file(speech_file)
```

## Adding adapters

So now that we've proved that the model is at least working, let's go ahead and add adapters. We basically need to create a new yaml file adding adapters to the model and then train with this new yaml file. To do this we'll just load the old yaml file and then we'll change all the parts necessary to train the adapted model.


```python
%%writefile train_lora.patch
--- train.yaml	2024-10-07 19:23:49.839501714 -0400
+++ train_lora.yaml	2024-10-07 19:25:40.340933091 -0400
@@ -30,7 +30,7 @@
 NOISE_DATASET_URL: https://www.dropbox.com/scl/fi/a09pj97s5ifan81dqhi4n/noises.zip?rlkey=j8b0n9kdjdr32o1f06t0cw5b7&dl=1
 RIR_DATASET_URL: https://www.dropbox.com/scl/fi/linhy77c36mu10965a836/RIRs.zip?rlkey=pg9cu8vrpn2u173vhiqyu743u&dl=1
 
-output_folder: !ref results/CRDNN_BPE_960h_LM/<seed>
+output_folder: !ref results/crdnn_lora/<seed>
 test_wer_file: !ref <output_folder>/wer_test.txt
 save_folder: !ref <output_folder>/save
 train_log: !ref <output_folder>/train_log.txt
@@ -41,7 +41,7 @@
 # speechbrain HuggingFace repository. However, a local path pointing to a
 # directory containing the lm.ckpt and tokenizer.ckpt may also be specified
 # instead. E.g if you want to use your own LM / tokenizer.
-pretrained_path: speechbrain/asr-crdnn-rnnlm-librispeech
+pretrained_path: results/CRDNN_BPE_960h_LM/2602/save/CKPT+latest
 
 
 # Path where data manifest files will be stored. The data manifest files are created by the
@@ -481,10 +481,9 @@
     ctc_lin: !ref <ctc_lin>
     seq_lin: !ref <seq_lin>
     normalize: !ref <normalize>
-    lm_model: !ref <lm_model>
 
 # Gathering all the submodels in a single model object.
-model: !new:torch.nn.ModuleList
+model_pretrained: !new:torch.nn.ModuleList
     - - !ref <encoder>
       - !ref <embedding>
       - !ref <decoder>
@@ -629,8 +628,31 @@
     loadables:
         lm: !ref <lm_model>
         tokenizer: !ref <tokenizer>
-        model: !ref <model>
+        model: !ref <model_pretrained>
     paths:
         lm: !ref <pretrained_path>/lm.ckpt
         tokenizer: !ref <pretrained_path>/tokenizer.ckpt
-        model: !ref <pretrained_path>/asr.ckpt
+        model: !ref <pretrained_path>/model.ckpt
+
+new_encoder: !new:speechbrain.nnet.adapters.AdaptedModel
+    model_to_adapt: !ref <encoder>
+    adapter_class: !name:speechbrain.nnet.adapters.LoRA
+    all_linear: True
+    manual_adapter_insertion: True
+    adapter_kwargs:
+        rank: 8
+
+new_decoder: !new:speechbrain.nnet.adapters.AdaptedModel
+    model_to_adapt: !ref <decoder>
+    adapter_class: !name:speechbrain.nnet.adapters.LoRA
+    all_linear: True
+    manual_adapter_insertion: True
+    adapter_kwargs:
+        rank: 8
+
+model: !new:torch.nn.ModuleList
+    - - !ref <new_encoder>
+      - !ref <embedding>
+      - !ref <new_decoder>
+      - !ref <ctc_lin>
+      - !ref <seq_lin>

```


```python
!patch train.yaml -i train_lora.patch -o train_lora.yaml
```

Because we are loading the pretrained parameters using the pretrainer, we have to insert this code to insert the adapters after the pretrained parameters have been loaded.

This is the reason for the `manual_adapter_insertion: True` in the yaml and the following brief change to the training code:


```python
%%writefile train_lora_py.patch
--- train.py	2024-10-07 14:57:21.534381751 -0400
+++ train_lora.py	2024-10-07 19:33:12.839895913 -0400
@@ -473,6 +473,8 @@
     # the path given in the YAML file). The tokenizer is loaded at the same time.
     hparams["pretrainer"].collect_files()
     hparams["pretrainer"].load_collected()
+    hparams["new_encoder"].insert_adapters()
+    hparams["new_decoder"].insert_adapters()
 
     # Trainer initialization
     asr_brain = ASR(
```


```python
!patch train.py -i train_lora_py.patch -o train_lora.py
```

## Training the adapted model

Training works identically to before, using the updated lora file. The adapted model is designed to work as an in-place replacement. Notice how the number of trainable parameters is reduced to close to 1% of the original parameters.


```python
!python train_lora.py train_lora.yaml --number_of_epochs=1 --batch_size=2 --test_scorer "!ref <valid_scorer>" --enable_add_reverb=False --enable_add_noise=False #To speed up
```

## Custom adapter

We designed this so that you could replace the SpeechBrain adapter with a `peft` adapter:

```diff
new_encoder: !new:speechbrain.nnet.adapters.AdaptedModel
    model_to_adapt: !ref <encoder>
-   adapter_class: !name:speechbrain.nnet.adapters.LoRA
+   adapter_class: !name:peft.tuners.lora.layer.Linear
    manual_adapter_insertion: True
    adapter_kwargs:
-       rank: 16
+       r: 16
+       adapter_name: lora
```

But this trains exactly the same thing as before, so no need for us to go through the whole thing. Perhaps more interesting is designing a custom adapter:


```python
%%file pool_lora.py

import torch

class PoolLoRA(torch.nn.Module):
    def __init__(self, target_module, stride=2, rank=16, alpha=1.0):
        super().__init__()

        input_size = target_module.weight.data.shape[1]
        output_size = target_module.weight.data.shape[0]
        
        # Disable gradient for pretrained module
        self.pretrained_module = target_module
        for param in self.pretrained_module.parameters():
            param.requires_grad = False
        device = target_module.weight.device

        self.adapter_down_scale = torch.nn.AvgPool1d(kernel_size=stride)
        self.adapter_down_proj = torch.nn.Linear(
            input_size // stride, rank, bias=False, device=device
        )   
        self.adapter_up_proj = torch.nn.Linear(
            rank, output_size, bias=False, device=device
        )   
        self.adapter_up_proj.weight.data.fill_(0.0)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor):
        """Applies the LoRA Adapter.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module.

        Returns
        -------
        The linear outputs
        """
        x_pretrained = self.pretrained_module(x)

        x_downsample = self.adapter_down_proj(self.adapter_down_scale(x))
        x_pool_lora = self.adapter_up_proj(x_downsample)
        
        return x_pretrained + x_pool_lora * self.scaling
```


```python
%%writefile train_pool_lora.patch
--- train_lora.yaml	2024-10-07 22:44:02.767830301 -0400
+++ train_pool_lora.yaml	2024-10-07 22:41:30.602641301 -0400
@@ -30,7 +30,7 @@
 NOISE_DATASET_URL: https://www.dropbox.com/scl/fi/a09pj97s5ifan81dqhi4n/noises.zip?rlkey=j8b0n9kdjdr32o1f06t0cw5b7&dl=1
 RIR_DATASET_URL: https://www.dropbox.com/scl/fi/linhy77c36mu10965a836/RIRs.zip?rlkey=pg9cu8vrpn2u173vhiqyu743u&dl=1
 
-output_folder: !ref results/crdnn_lora/<seed>
+output_folder: !ref results/crdnn_pool_lora/<seed>
 test_wer_file: !ref <output_folder>/wer_test.txt
 save_folder: !ref <output_folder>/save
 train_log: !ref <output_folder>/train_log.txt
@@ -636,19 +636,21 @@
 
 new_encoder: !new:speechbrain.nnet.adapters.AdaptedModel
     model_to_adapt: !ref <encoder>
-    adapter_class: !name:speechbrain.nnet.adapters.LoRA
+    adapter_class: !name:pool_lora.PoolLoRA
     all_linear: True
     manual_adapter_insertion: True
     adapter_kwargs:
-        rank: 8
+        stride: 2
+        rank: 16
 
 new_decoder: !new:speechbrain.nnet.adapters.AdaptedModel
     model_to_adapt: !ref <decoder>
-    adapter_class: !name:speechbrain.nnet.adapters.LoRA
+    adapter_class: !name:pool_lora.PoolLoRA
     all_linear: True
     manual_adapter_insertion: True
     adapter_kwargs:
-        rank: 8
+        stride: 2
+        rank: 16
 
 model: !new:torch.nn.ModuleList
     - - !ref <new_encoder>
```


```python
!patch train_lora.yaml -i train_pool_lora.patch -o train_pool_lora.yaml
```


```python
!python train_lora.py train_pool_lora.yaml --number_of_epochs=1 --batch_size=2 --test_scorer "!ref <valid_scorer>" --enable_add_reverb=False --enable_add_noise=False #To speed up
```

## Conclusion

That's it, thanks for following along! Go forth and make cool adapters.

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
