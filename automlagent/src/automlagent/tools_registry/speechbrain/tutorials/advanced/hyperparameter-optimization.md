Summary: This tutorial demonstrates implementing hyperparameter optimization in SpeechBrain using the hpopt utility. It covers integrating optimization contexts into existing training code, configuring search spaces through YAML files, and managing optimization trials with Orion. Key implementations include wrapping main training code in hyperparameter optimization contexts, reporting results for optimization, and defining search spaces for parameters like embedding dimensions and TDNN channels. The tutorial helps with tasks like setting up distributed hyperparameter optimization, managing checkpointing during trials, and scaling optimization across multiple GPUs. Essential features include Orion integration, parallel optimization capabilities, and proper handling of hyperparameter configurations in YAML files.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/hyperparameter-optimization.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/advanced/hyperparameter-optimization.ipynb)

# Hyperparameter Optimization

Many of the speech processing tasks implemented as part of the SpeechBrain project rely on the careful selection of hyperparameters, such as:

* The number of layers
* Normalization
* Hidden layer dimensions
* Weights within cost functions
* etc

Selecting such hyperparameters by hand can be tedious. This tutorial will show how to use the automated hyperparameter optimization techniques implemented as part of the [Oríon](https://github.com/Epistimio/orion) project to automatically fit hyperparameters in a systematic way.

## Prerequisites


### Imports


```python
import os
```

### Install SpeechBrain

SpeechBrain can be downloaded from the GitHub repository listed below.


```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH

```

### Dependency Fixes

PyYAML 6.0 is not backwards-compatible, a 5.x version is needed to support HyperPyYAML


```python
%%capture
!pip install pyyaml==5.4.1
```

### Install Oríon
Oríon can be installed using `pip` or `conda`


```python
%%capture
!pip install orion[profet]
```


```python
from speechbrain.utils import hpopt as hp
```

## Update the Recipe to Support Hyperparameter Optimization

SpeechBrain comes with a convenience wrapper called `hpopt`, which is capable of reporting objective values to Orion or to other tools.

For a complete example on how to implement it,

1. Add the following import statement to the top of your recipe:

```python
from speechbrain.utils import hpopt as hp
```

2. Wrap the main code of your recipe in a hyperparameter optimization context. Set `objective_key` to the metric that Orion will optimize.

    **Before**:

  ```python
  hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
  
  with open(hparams_file) as fin:
      hparams = load_hyperpyyaml(fin, overrides)
  
  ## ...
  
  spk_id_brain = SpkIdBrain(
      modules=hparams["modules"],
      opt_class=hparams["opt_class"],
      hparams=hparams,
      run_opts=run_opts,
      checkpointer=hparams["checkpointer"],
  )
  
  # The `fit()` method iterates the training loop, calling the methods
  # necessary to update the parameters of the model. Since all objects
  # with changing state are managed by the Checkpointer, training can be
  # stopped at any point, and will be resumed on next call.
  spk_id_brain.fit(
      epoch_counter=spk_id_brain.hparams.epoch_counter,
      train_set=datasets["train"],
      valid_set=datasets["valid"],
      train_loader_kwargs=hparams["dataloader_options"],
      valid_loader_kwargs=hparams["dataloader_options"],
  )

  ```

  **After**:

    ```python
    with hp.hyperparameter_optimization(objective_key="error") as hp_ctx: # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:]) # <-- Replace sb with hp_ctx

        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        ## ...

            spk_id_brain = SpkIdBrain(
                modules=hparams["modules"],
                opt_class=hparams["opt_class"],
                hparams=hparams,
                run_opts=run_opts,
                checkpointer=hparams["checkpointer"],
            )

            # The `fit()` method iterates the training loop, calling the methods
            # necessary to update the parameters of the model. Since all objects
            # with changing state are managed by the Checkpointer, training can be
            # stopped at any point, and will be resumed on next call.
            spk_id_brain.fit(
                epoch_counter=spk_id_brain.hparams.epoch_counter,
                train_set=datasets["train"],
                valid_set=datasets["valid"],
                train_loader_kwargs=hparams["dataloader_options"],
                valid_loader_kwargs=hparams["dataloader_options"],
            )
    ```

3. Add code to report the stats

  e.g. in `on_stage_end` when `stage == sb.Stage.VALID`

  ```python
hp.report_result(stage_stats)
```

  The **last** result reported through this function will be reported for hyperparameter optimization.

  The key specified in **objective_key** parameter needs to be present in the dictionary passed to `report_result`.

4. Add the following lines in your main hyperparameter file `train.yaml`:
```yaml
hpopt_mode: null
hpopt: null
```

5. **Optional**: Create a separate YAML file overriding any hyperparameters to be used during hyperparameter optimization that are **different** from the ones used during regular training **other than** the ones being fitted. A typical approach would reduce the number of epochs and the number of training samples.

  This step can be omitted if the number of parameters being overridden is small. In this case, they can be passed on the command line instead.

  Example:

  `hpopt.yaml`:
  ```yaml
  number_of_epochs: 1
  ckpt_enable: false
  ```
6. ❗ **Important**: Most recipes use a checkpointer to save snapshots of the model after each epoch (or on a custom schedule) to ensure that training can be resumed if it is interrupted. During hyperparameter optimization, this can cause issues because if the model's architecture (e.g. the number of layers, neurons per layer, etc) changes from one set of hyperparamter values to the next, an attempt to restore a checkpoint will fail.

  One possible solution is to make the run of the checkpointer conditional and to disable it in `hpopt.yaml`

  __Before__:
  ```python
  self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])
  ```
  __After__:
  ```python
  if self.hparams.ckpt_enable:
      self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])
  ```

  An alternative strategy is to reconfigure the checkpointer to save each run in a separate directory. For this scenario, the hyperparameter optimization wrapper can supply a variable named trial_id, which can be interpolated into the output path.

  Given below is an example of this strategy:

  `hpopt.yaml`:

  ```yaml
  number_of_epochs: 1
  ckpt_enable: False
  trial_id: hpopt
  output_folder: !ref ./results/speaker_id/<trial_id>
  ```

  `train.yaml`:

  ```yaml
  # ...
  save_folder: !ref <output_folder>/save
  # ...
  checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
      checkpoints_dir: !ref <save_folder> #<-- will contain trial_id
      recoverables:
          embedding_model: !ref <embedding_model>
          classifier: !ref <classifier>
          normalizer: !ref <mean_var_norm>
          counter: !ref <epoch_counter>
  ```

## Perform the Hyperparameter Search

### Choose and Prepare Hyperparameters

Choose the hyperparameters that you would like to optimize using Orion out of the
ones available in your hyperparameter file. The hyperparameters need to be
available at the top level in order for it to be fitted using this technique.

Consider the following sample file:

```yaml
dropout: 0.1
n_mels: 80
encoder: !new:speechbrain.lobes.models.mymodel.MyModel
    input_shape: [null, null, !ref <n_mels>]
    dropout: !ref <dropout>
    cnn_blocks: 3
```

In the above file, `n_mels` and `dropout` are available for optimization, but `cnn_blocks` is not.

To make `cnn_blocks` available for optimization, modify it as follows:

```yaml
dropout: 0.1
n_mels: 80
cnn_blocks: 3 # <-- Define at the top level
encoder: !new:speechbrain.lobes.models.mymodel.MyModel
    input_shape: [null, null, !ref <n_mels>]
    dropout: !ref <dropout>
    cnn_blocks: !ref <cnn_blocks> # <-- Introduce a reference
```

### Configure Orion
Create a `.yaml` file with the configuration for the Orion algorithm to be used.

Given below is an example:
```yaml
experiment:
    max_trials: 1000
    max_broken: 1000
    algorithms:
        tpe:
            seed: 42
            n_initial_points: 5
```


```python
config_file_content = """
experiment:
    max_trials: 3
    max_broken: 1
    algorithms:
        tpe:
            seed: 42
            n_initial_points: 5
"""
config_path = os.path.expanduser("~/config")
if not os.path.exists(config_path):
    os.mkdir(config_path)

config_file_path = os.path.join(config_path, "orion-speaker-id.yaml")
with open(config_file_path, "w") as config_file:
    print(config_file_content, file=config_file)
```

For more information on the available algorithms, please take a look at  the [Orion Repository](https://github.com/Epistimio/orion/tree/develop/src/orion/algo).

### Define the Search Space
Write a shell script calling Orion defining a search space

Example:
```sh
#!/bin/bash
HPOPT_EXPERIMENT_NAME=speaker-id
HPOPT_CONFIG_FILE=$HOME/config/orion-speaker-id.yaml
orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE python train.py hparams/$HPARAMS \
    --hpopt hpopt.yaml \
    --hpopt_mode orion \
    --emb_dim~"choices([128,256,512,768,1024])" \
    --tdnn_channels~"choices([128,256,512,768,1024])"
```

Replace `--hpopt hpopt.yaml` with `--hpopt=True` if you are not using the additional  `hpopt.yaml` file.

Consider running the standalone example below


```python
%env PYTHONPATH=/env/python:/content/speechbrain/
%cd /content/speechbrain/templates/hyperparameter_optimization_speaker_id
!orion hunt -n speaker-id -c $HOME/config/orion-speaker-id.yaml python train.py train.yaml \
  --hpopt hpopt.yaml \
  --hpopt_mode orion \
  --emb_dim~"choices([128,256,512,768,1024])" \
  --tdnn_channels~"choices([128,256,512,768,1024])"
```

## Inspecting Results

Use the `orion info` command to inspect the results of hyperparameter fitting.

The tool will output basic statistics about the hyperparameter fitting experiment, including the number of runs completed, the objective value for the best trial and the hyperparameter values corresponding to that run.

In the example below, the best objective achieved value is shown under **evaluation**, and the corresponding hyperparameter values are shown under **params**.

```
Stats
=====
completed: False
trials completed: 4
best trial:
  id: c1a71e0988d70005302ab655d7e391d3
  evaluation: 0.2384105920791626
  params:
    /emb_dim: 128
    /tdnn_channels: 128
start time: 2021-11-14 21:01:12.760704
finish time: 2021-11-14 21:13:25.043336
duration: 0:12:12.282632
```


```python
!orion info --name speaker-id
```

## Hyperparameter Optimization at Scale

### Multiple GPUs
Since Orion simply wraps the execution of the training script and launches it for each set of hyperparameters using the OS shell, training scripts that support Data-Parallel (DP) or Distributed Data Parallel (DDP) execution can be used with hyperparameter fitting without modification.

For information on how to set up DP/DDP experiments, refer to the [SpeechBrain documentation](https://speechbrain.readthedocs.io/en/latest/multigpu.html#) and the [Multi-GPU Considerations](https://speechbrain.readthedocs.io/en/latest/multigpu.html) tutorial.

### Parallel or Distributed Oríon

Oríon itself provide support for parallel and distributed hyperparameter fitting.

To use multiple parallel workers on a single node, pass the `--n-workers` parameter to the Oríon CLI.

The example below will start the experiment with three workers:
```shell
orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HOPT_CONFIG_FILE --n-workers 3 python train.py hparams/$HPARAMS \
    --hpopt hpopt.yaml \
    --hpopt_mode orion \
    --emb_dim~"choices([128,256,512,768,1024])" \
    --tdnn_channels~"choices([128,256,512,768,1024])"

```

For more advanced scenarios, including distributed hyperparameter fittig on multiple nodes, refer to the [Parallel Workers](https://orion.readthedocs.io/en/stable/user/parallel.html]) page in Oríon's official documentation.

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
