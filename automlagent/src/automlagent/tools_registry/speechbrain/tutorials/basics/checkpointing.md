Summary: This tutorial covers SpeechBrain's checkpointing system implementation, focusing on model state persistence and recovery. It demonstrates how to implement checkpoint management for PyTorch models, optimizers, and training states using the Checkpointer class. Key functionalities include basic checkpoint saving/loading, metadata management, selective checkpoint retention, and parameter transfer between models. The tutorial helps with tasks like implementing training recovery, managing model versions, and handling early stopping based on validation metrics. Essential features covered include checkpoint structure definition, metadata handling, storage management with predicates, and custom checkpoint hooks for specialized objects.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/checkpointing.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/basics/checkpointing.ipynb)

# Checkpointing

By *checkpointing*, we mean saving the model and all the other necessary state information (like optimizer parameters, which epoch, and which iteration), at a particular point in time. For experiments, this has two main motivations:
- *Recovery*. Continuing an experiment from half-way through. A compute-cluster job can run out of time or memory, or there can be some simple error, which stops the experiment script before it finishes. In that case, all progress that isn't saved to disk is lost.
- *Early stopping*. During training, performance should be monitored on a separate validation set, which gives an estimate of generalization. When training progresses, we expect validation error to decrease at first. If we train too long, though, validation error can start to increase again (due to *overfitting*). After training, we should go back to the model parameters that performed best on the validation set.

Besides, it is also important to save the trained model parameters, so that the model can be used outside the experiment script.


## The role of the SpeechBrain checkpointer

The SpeechBrain checkpointer simply orchestrates checkpointing. It keeps track of all the things which should be included in checkpoints, how each of those is saved, where the checkpoints should go, and it centralizes loading and saving.

The checkpointer doesn't actually save things to the disk itself. It either finds a suitable saving function by type (class inheritance considered), or you can provide a custom hook.

## Installing dependencies


```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH
```


```python
import speechbrain as sb
import torch
from speechbrain.utils.checkpoints import Checkpointer
```

## The SpeechBrain Checkpointer in a nutshell

Run the following code block multiple times. Each time you run the block, it trains one epoch, then ends. Running the block again is similar to restarting an experiment script.


```python
# You have a model, an optimizer and an epoch counter:
model = torch.nn.Linear(1, 1, False)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
epoch_counter = sb.utils.epoch_loop.EpochCounter(10)
# Create a checkpointer:
checkpoint_dir = "./nutshell_checkpoints"
checkpointer = Checkpointer(checkpoint_dir,
                            recoverables = {"mdl": model,
                                            "opt": optimizer,
                                            "epochs": epoch_counter})
# Now, before running the training epochs, you want to recover,
# if that is possible (if checkpoints have already been saved.)
# By default, the most recent checkpoint is loaded.
checkpointer.recover_if_possible()
# Then we run an epoch loop:
for epoch in epoch_counter:
    print(f"Starting epoch {epoch}.")
    # Training:
    optimizer.zero_grad()
    prediction = model(torch.tensor([1.]))
    loss = (prediction - torch.tensor([1.]))**2
    loss.backward()
    optimizer.step()
    print(f"Model prediction={prediction.item()}, loss={loss.item()}")
    # And finally at the end, save an end-of-epoch checkpoint:
    checkpointer.save_and_keep_only(meta={"loss":loss.item()})
    # Now, let's "crash" this code block:
    break
else:
    # After training (epoch loop is depleted),
    # we want to recover the best model:
    print("Epoch loop has finished.")
    checkpointer.recover_if_possible(min_key="loss")
    print(f"Best model parameter: {model.weight.data}")
    print(f"Achieved on epoch {epoch_counter.current}.")
```


```python
# You can use this cell to reset, by deleting all checkpoints:
checkpointer.delete_checkpoints(num_to_keep=0)
```

## What does a checkpoint look like?


The checkpointer is given a top-level directory, where all the checkpoints go:


```python
checkpoint_dir = "./full_example_checkpoints"
checkpointer = Checkpointer(checkpoint_dir)
```

Each checkpoint should contain many things like model parameters and training progress.


```python
# You have a model, an optimizer and an epoch counter:
model = torch.nn.Linear(1, 1, True)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
epoch_counter = sb.utils.epoch_loop.EpochCounter(10)
```

 Each entity to save is assigned to the checkpointer separately, with a unique key, like a name:


```python
checkpointer.add_recoverable("mdl", model)
checkpointer.add_recoverables({"opt": optimizer, "epoch": epoch_counter})
```

When a checkpoint is saved, the checkpointer creates a directory inside the top-level directory. That sub-directory represents this saved checkpoint. Inside the newly created directory each entity, that was passed to the checkpointer, gets its own file.




```python
ckpt = checkpointer.save_checkpoint()
print("The checkpoint directory was:", ckpt.path)
for key, filepath in ckpt.paramfiles.items():
    print("The entity with key", key, "was saved to:", filepath)
```

### What goes in each file?

That is upto the entities. The checkpointer finds a saving "hook" by type (class inheritance considered) and calls that hook with the object to save and a filepath.

Torch entities (Module, Optimizer) have default save and load hooks already:


```python
torch_hook = sb.utils.checkpoints.get_default_hook(torch.nn.Linear(1,1), sb.utils.checkpoints.DEFAULT_SAVE_HOOKS)
print(torch_hook.__doc__)
```

Classes can register their own default saving and loading hooks:


```python
@sb.utils.checkpoints.register_checkpoint_hooks
class Duck:
    def __init__(self):
        self.quacks = 0

    def quack(self):
        print("Quack!")
        self.quacks += 1
        print(f"I have already quacked {self.quacks} times.")

    @sb.utils.checkpoints.mark_as_saver
    def save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.quacks))

    @sb.utils.checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        # Irrelevant for ducks:
        del end_of_epoch
        del device
        with open(path) as fi:
            self.quacks = int(fi.read())

duck = Duck()
duckpointer = Checkpointer("./duckpoints", {"ducky": duck})
duckpointer.recover_if_possible()
duck.quack()
_ = duckpointer.save_checkpoint()
```

### Meta info

The checkpoint also stores a dictionary of meta information. You can put e.g. validation loss or some other metric there. By default, only the unix time is saved.


```python
# Following from the cells of "What does a checkpoint look like?"
checkpointer.save_checkpoint(meta={"loss": 15.5, "validation-type": "fast", "num-examples": 3})
ckpt = checkpointer.save_checkpoint(meta={"loss": 14.4, "validation-type": "full"})
print(ckpt.meta)
```

This meta information can be used to load the best checkpoint, not just the most recent one:


```python
ckpt = checkpointer.recover_if_possible(min_key="loss")
print(ckpt.meta)
```

There are also more advanced filters available:


```python
checkpointer.save_checkpoint(meta={"loss": 12.1, "validation-type": "fast", "num-examples": 2})
ckpt =  checkpointer.recover_if_possible(importance_key=lambda ckpt: -ckpt.meta["loss"]/ckpt.meta["num-examples"],
                                 ckpt_predicate=lambda ckpt: ckpt.meta.get("validation-type") == "fast")
print(ckpt.meta)
```

## Keeping a limited amount of checkpoints

Neural models these days can be huge, and we don't need to store every checkpoint. Checkpoints can be deleted explicitly, and the same types of filters can be used as with recovery:


```python
checkpointer.delete_checkpoints(num_to_keep=1, ckpt_predicate=lambda ckpt: "validation-type" not in ckpt.meta)
```

But for convenience, there is also a method which saves and deletes at the same time:


```python
checkpointer.save_and_keep_only(meta={"loss": 13.1, "validation-type": "full"},
                                num_to_keep = 2,
                                ckpt_predicate=lambda ckpt: ckpt.meta.get("validation-type") == "full")
```

### Pretraining / parameter transfer

Transferring parameters from a pretrained model is different from recovery, although the have some similarities.



### Finding the best checkpoint

The first step in parameter transfer is to find the ideal set of parameters to take. You can use the checkpointer for that: point an empty checkpointer at the top level checkpoints directory of an experiment, and find a checkpoint with your criterion.


```python

ckpt_finder = Checkpointer(checkpoint_dir)
best_ckpt = ckpt_finder.find_checkpoint(min_key="loss",
                                        ckpt_predicate=lambda ckpt: ckpt.meta.get("validation-type") == "full")
best_paramfile = best_ckpt.paramfiles["mdl"]
print("The best parameters were stored in:", best_paramfile)
```

### Transferring parameters

There is no generic formula for parameter transfer, and in a lot of cases you may have to write some custom code to connect the incoming parameters to the new model.

SpeechBrain has an almost trivial implementation for transferring parameters to another torch Module, which simply loads the matching layers (by name) and ignores saved parameters for which no matching layer is found:


```python
finetune_mdl = torch.nn.Linear(1,1,False) #This one doesn't have bias!
with torch.no_grad():
    print("Before:", finetune_mdl(torch.tensor([1.])))
    sb.utils.checkpoints.torch_parameter_transfer(finetune_mdl, best_paramfile)
    print("And after:", finetune_mdl(torch.tensor([1.])))
```

### Orchestrating transfer

SpeechBrain has a parameter transfer orchestrator similar to Checkpointer: `speechbrain.utils.parameter_transfer.Pretrainer`. The point is primarily to implement the parameter download-and-load for `speechbrain.pretrained.Pretrained` subclasses such as `EncoderDecoderASR` and to aid in writing easy-to-share recipes.

Similar to Checkpointer, Pretrainer handles mapping parameter files to instances, and calling the transfer code (implemented as similar hooks as checkpoint loading).


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
