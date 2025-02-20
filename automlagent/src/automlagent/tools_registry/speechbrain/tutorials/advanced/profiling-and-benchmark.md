Summary: This tutorial demonstrates how to implement performance profiling in SpeechBrain using PyTorch's profiler. It provides implementation knowledge for setting up profiling flags, configuring warmup periods, and analyzing performance metrics through tensorboard. The tutorial helps with tasks like monitoring model training performance, optimizing execution traces, and analyzing memory/GPU usage. Key features covered include profiler configuration parameters (profile_warmup, profile_steps), tensorboard integration for visualization, and best practices for effective profiling sessions, making it valuable for debugging and optimizing deep learning training workflows.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/profiling-and-benchmark.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/advanced/profiling-and-benchmark.ipynb)

# Performance Profiling

SpeechBrain provides a simple way of profiling any training recipe. The output will be a standard tensorboard logdir containing all the typical metrics, traces, charts etc. If you want more information, please refer to [the PyTorch documentation](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Indeed, SpeechBrain simply uses the standard Torch profiler.

## Installation
Please ensure you have installed the TensorBoard profiler:

`pip install torch_tb_profiler`

## Calling the profiler

Let's start a training with the profiler enabled. Of course, we do not want to run the full training, just a few steps. This can be achieved with:

```bash
python train.py hparams/config.yaml --profile_training --profile_warmup 10 --profile_steps 5
```

**--profile_warmup** and **--profile_steps** indicate for how long we should wait, i.e. warmup, before starting to record and the number of steps to record respectively. Waiting is useful as some PyTorch and CuDNN optimizations are usually happening on the first training steps.

## Visualizing the logs with tensorboard

The previous step will have generated profiler logs in the **{output_folder}/profiler_logs** location. The output folder is the one from your YAML. You can start tensorboard and exploring the trace and charts by executing:

```bash
tensorboard --log_dir {output_folder}/profiler_logs
```

Of course, you'll need to have tensorboard installed.

```bash
pip install tensorboard
```

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
