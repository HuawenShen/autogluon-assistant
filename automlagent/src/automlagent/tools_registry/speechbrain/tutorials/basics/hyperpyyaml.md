Summary: This tutorial covers HyperPyYAML, a YAML extension for managing hyperparameters and configurations in SpeechBrain. It demonstrates implementation techniques for creating and referencing Python objects, handling parameter overrides, and managing model configurations using special tags (!new, !ref, !copy, !name). The tutorial helps with tasks like setting up neural network architectures, managing hyperparameters, and creating reusable configurations. Key features include arithmetic operations in references, object instantiation, deep copying, parameter referencing, and file inclusion, making it particularly useful for organizing machine learning experiments and maintaining clean separation between code and configurations.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/hyperpyyaml.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/basics/hyperpyyaml.ipynb)

# HyperPyYAML Tutorial

An essential aspect of any deep learning pipeline is the definition of hyperparameters and other metadata. These hyperparameters, in conjunction with the deep learning algorithms, govern various aspects of the pipeline, including model architecture, training, and decoding.

In SpeechBrain, we emphasize a clear distinction between hyperparameters and learning algorithms in the structure of our toolkit. To achieve this, we separate our recipes into two primary files: `train.py` and `train.yaml`.

The `train.yaml` file follows a format developed by SpeechBrain, known as "HyperPyYAML." We chose to extend YAML due to its highly readable nature for data serialization. By building upon this already user-friendly format, we have created an extended definition of hyperparameters, ensuring that our experimental code remains concise and easily readable.

Here's a brief example using PyTorch code to illustrate the use of HyperPyYAML. It's important to note that PyTorch is not a requirement for utilizing HyperPyYAML:


```python
%%capture
!pip install torch
!pip install hyperpyyaml
```


```python
import torch
from hyperpyyaml import load_hyperpyyaml

example_hyperparams = """
base_channels: 32
kernel_size: 11
padding: !ref <kernel_size> // 2

layer1: !new:torch.nn.Conv1d
  in_channels: 1
  out_channels: !ref <base_channels>
  kernel_size: !ref <kernel_size>
  padding: !ref <padding>

layer2: !new:torch.nn.Conv1d
  in_channels: !ref <base_channels>
  out_channels: !ref <base_channels> * 2
  kernel_size: !ref <kernel_size>
  padding: !ref <padding>

layer3: !new:torch.nn.Conv1d
  in_channels: !ref <base_channels> * 2
  out_channels: 1
  kernel_size: !ref <kernel_size>
  padding: !ref <padding>

model: !new:torch.nn.Sequential
  - !ref <layer1>
  - !new:torch.nn.LeakyReLU
  - !ref <layer2>
  - !new:torch.nn.LeakyReLU
  - !ref <layer3>
"""

# Create model directly by loading the YAML
loaded_hparams = load_hyperpyyaml(example_hyperparams)
model = loaded_hparams["model"]

# Transform a 2-second audio clip
input_audio = torch.rand(1, 1, 32000)
transformed_audio = model(input_audio)
print(transformed_audio.shape)

# Try a different hyperparameter value by overriding the padding value
loaded_hparams = load_hyperpyyaml(example_hyperparams, {"padding": 0})
model = loaded_hparams["model"]
transformed_audio = model(input_audio)
print(transformed_audio.shape)
```

As this example shows, HyperPyYAML allows for complex hyperparameter definitions with compositions. In addition, any value can be overridden for hyperparameter tuning. To grasp how all of this works, let's first briefly look at the basics of YAML.

## Basic YAML syntax

Enough prelude: lets talk YAML! Here's a brief example of a yaml snippet and what it would look like once loaded to python:


```python
import yaml
yaml_string = """
foo: 1
bar:
  - item1
  - item2
baz:
  item1: 3.4
  item2: True
"""
yaml.safe_load(yaml_string)
```

As you can see, YAML has built-in support for a variety of data types, including string, int, float, bool, list, and dictionary. Our HyperPyYAML format keeps all of this functionality.


```python
from hyperpyyaml import load_hyperpyyaml
load_hyperpyyaml(yaml_string)
```

Our primary additions to yaml format are added with YAML tags. Tags are added before an item definition, and are prefixed with `!`. For the purpose of illustrating how tags are used, here is an example with a minor addition that we've made, the `!tuple` tag:


```python
yaml_string = """
foo: !tuple (3, 4)
"""
load_hyperpyyaml(yaml_string)
```

Now you know the YAML basics, time to move on to our additions!

## Tags `!new:` and `!name:`
YAML tags can contain a suffix to more specifically define what type of tag it is. We use this to define a tag that is able to create any python object, not just a basic type. This tag starts with `!new:` and contains the type of the object. For example:


```python
yaml_string = """
foo: !new:collections.Counter
"""
loaded_yaml = load_hyperpyyaml(yaml_string)
loaded_yaml["foo"]
```


```python
loaded_yaml["foo"].update({"a": 3, "b": 5})
loaded_yaml["foo"]["a"] += 1
loaded_yaml["foo"]
```

Of course many python objects take arguments during creation. These arguments can be passed with a list for positional arguments or a dictionary for keyword arguments.


```python
yaml_string = """
foo: !new:collections.Counter
  - [a, b, r, a, c, a, d, a, b, r, a]
bar: !new:collections.Counter
  a: 2
  b: 1
  c: 5
"""
load_hyperpyyaml(yaml_string)
```

Another python object that is useful to create is a function object. In HyperPyYAML this can be done with the `!name:` tag. Behind the scenes, this tag uses `functools.partial` to create a new function definition with the default arguments provided. For example:


```python
yaml_string = """
foo: !name:collections.Counter
  a: 2
"""
loaded_yaml = load_hyperpyyaml(yaml_string)
loaded_yaml["foo"](b=4)
```

The default arguments can be overridden, just as a normal python function


```python
loaded_yaml["foo"](a=3, b=5)
```

## Tags `!ref` and `!copy`

Of course some hyperparameters get used in multiple places, so we added a mechanism for referring to another item called `!ref`. The node that this tag is applied to must be a string that contains the location of the node to copy. Sub-nodes can be accessed with square brackets, same as in Python. For example:


```python
yaml_string = """
foo:
  a: 3
  b: 4
bar:
  c: !ref <foo>
  d: !ref <foo[b]>
"""
load_hyperpyyaml(yaml_string)
```

The `!ref` tag can support simple arithmetic and string concatenation for basic hyperparameter combinations.


```python
yaml_string = """
folder1: abc/def
folder2: ghi/jkl
folder3: !ref <folder1>/<folder2>

foo: 1024
bar: 512
baz: !ref <foo> // <bar> + 1
"""
load_hyperpyyaml(yaml_string)
```

The `!ref` tag can also refer to objects, in which case it makes a reference to the same object, rather than a copy. If you'd prefer to make a copy instead, use the `!copy` tag.


```python
yaml_string = """
foo: !new:collections.Counter
  a: 4
bar: !ref <foo>
baz: !copy <foo>
"""
loaded_yaml = load_hyperpyyaml(yaml_string)
loaded_yaml["foo"].update({"b": 10})
print(loaded_yaml["bar"])
print(loaded_yaml["baz"])
```

## Other tags

We introduced a variety of other tags as well:
* `!tuple` to create python tuples. Note this is implicitly resolved, so you do not need to explicitly write out the tuple tag, just use parentheses as you would in Python.
* `!include` to insert other yaml files directly
* `!apply` to load and execute a python function, storing the result

We use `!apply` to set the random seed at the beginning of loading the yaml, so that the models have the same parameters each run. The result is not stored, because it starts with `__`.


```python
yaml_string = """
sum: !apply:sum
  - [1, 2]
__set_seed: !apply:torch.manual_seed [1234]
"""
load_hyperpyyaml(yaml_string)
```

## Overrides

In order to run experiments with various values for a hyperparameter, we have a system for overriding the values that are listed in the yaml file.


```python
overrides = {"foo": 7}
fake_file = """
foo: 2
bar: 5
"""
load_hyperpyyaml(fake_file, overrides)
```

As shown in this example, overrides can take an ordinary python dictionary. However, this form does not support python objects. To override a python object, overrides can also take a yaml-formatted string with the HyperPyYAML syntax.


```python
load_hyperpyyaml(fake_file, "foo: !new:collections.Counter")
```

## Conclusion

We are proud to present our HyperPyYAML syntax, and we think that it provides a readable and concise way to structure hyperparameter definitions. In addition, it removes unnecessary complexity from experiment files, allowing the algorithms to become clear. As is evident in the first example, overrides are easy, making hyperparameter tuning a cinch. Overall, we have found this package to be a valuable tool for deep learning!

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
