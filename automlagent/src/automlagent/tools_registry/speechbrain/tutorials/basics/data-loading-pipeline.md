Summary: This tutorial demonstrates the implementation of data loading and processing in SpeechBrain, specifically focusing on speech recognition tasks. It covers essential techniques for creating efficient data pipelines using DynamicItemDataset, Dynamic Item Pipelines (DIPs), and CategoricalEncoder. The tutorial helps with tasks like loading audio data, implementing speaker recognition systems, and managing batch processing with padding. Key features include parallel data loading, automatic sequence padding, speaker ID encoding, audio processing pipelines, and integration with PyTorch's DataLoader. It provides code examples for creating dataset annotations, implementing data transformations, handling variable-length sequences, and building a complete speaker recognition training pipeline using SpeechBrain's Brain class.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/data-loading-pipeline.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/basics/data-loading-pipeline.ipynb)

# Data Loading

Handling data consumes 90% of the active work time in many machine learning projects.

SpeechBrain complements standard PyTorch data loading in handling variable-length sequences, large datasets, and complex data transformation pipelines. These are typical challenges when working with speech, but SpeechBrain tries not to make assumptions about your data.

## Install dependencies



```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH
```


```python
import speechbrain
import torch
```

In this tutorial we will use MiniLibriSpeech from https://www.openslr.org/resources/31: we download the validation set in the following two cells as well as images and scripts.


```python
%%capture
# here we download the material needed for this tutorial: images and an example based on mini-librispeech
!wget https://www.dropbox.com/s/b61lo6gkpuplanq/MiniLibriSpeechTutorial.tar.gz?dl=0
!tar -xvzf MiniLibriSpeechTutorial.tar.gz?dl=0
# downloading mini_librispeech dev data
!wget https://www.openslr.org/resources/31/dev-clean-2.tar.gz
!tar -xvzf dev-clean-2.tar.gz
```

## Preface: PyTorch data loading pipeline

SpeechBrain data-IO follows and extends [PyTorch data loading](https://pytorch.org/docs/stable/data.html). This preface section recaps PyTorch data loading and does not yet consider the SpeechBrain data loading extensions.

### Overview
PyTorch data loading can run in many configurations,
but a typical approach has these basic elements:
- a [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), which loads data points one-at-a-time.
- a [collation function](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn), or ``collate_fn`` for short, which takes a list of data points and forms a batch.
- a [Sampler](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler), which determines the order in which the Dataset is iterated.
- a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), which combines the elements above (and has defaults for ``collate_fn`` and Sampler), and orchestrates the whole pipeline.



### Dataset
The role of the Dataset is to produce single data points. Typically they are loaded off the disk, but they could also come from some more complex source or in some cases just from RAM. You can write your own Dataset subclass or sometimes you can use a standardized class. The training, validation, and test subsets get their own Dataset instances.

The Dataset interface is simple; it implements
`__getitem__` and usually also `__len__`. Usually, "map-style" Datasets are used, but it's worth noting that PyTorch also has a notion of [IterableDataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets)s.

`__getitem__` can return _anything_ because data can look like _anything_. Often, though, a data point consists of multiple associated things (e.g. an image and a label, or a speech waveform and its transcription). The Dataset should return all of these associated things.

It is also relatively common for the Dataset to somehow transform the data on the fly, on the CPU.


### Collation function

The ``collate_fn`` just converts a list of examples into a PyTorch tensor batch. If the data has variable-length sequences, ``collate_fn`` usually needs to implement padding.

### Sampler
Typically users don't need to create their own sampler; the two default options are to iterate in the original Dataset order or to iterate in random order.


### DataLoader

The DataLoader takes the other elements above and many other arguments such as batch size. The DataLoader has basic defaults for all arguments (except the Dataset, of course), but it's worthwhile to understand the [args](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

The DataLoader object is iterated in the training loop, and every Dataset instance (e.g. training, validation, test) gets its own DataLoader.
```python
train_loader = DataLoader(train_data, collate_fn=PaddedBatch, batch_size=32, num_workers=2)
for batch in train_loader:
    pred = model(batch.signal)
    ...
```

The iterator that DataLoader returns can either load batches in the same process that it is created in (`num_workers=0`), or it can start a new process (`num_workers=1`) or multiple new processes (`num_workers>1`). Because of the Global Interpreter Lock, Python cannot work on two tasks at the same time in a single process. Using at least one background worker process to load data while simultaneously running training is often essential for taking full advantage of GPU compute resources.

## SpeechBrain Basic dataIO
SpeechBrain basic data loading pipeline is illustrated in the Figure below.


```python
from IPython.display import Image
Image('sbdataio.png', width=1000)
```

Note that also more advanced/flexible pipelines are possible, where users can integrate their own custom Datasets, DataLoaders, and Samplers. These are illustrated in the [dataIO Advanced tutorial](todo).

The basic dataIO pipeline is organized around three "key" blocks: **DynamicItemDataset**, **Dynamic Items Pipelines (DIPs)**  and **CategoricalEncoder** which are tightly connected.

**DynamicItemDataset** inherits from `torch.utils.data.Dataset` and has been built to work together with **Dynamic Items Pipelines** to provide a straightforward and flexible way to fetch and transform data and labels from the raw dataset stored in your disk.

**DIPs** consists of user-defined functions in which the user specifies operations applied to metadata and data contained in the dataset. E.g. reading and augmenting an audio file or encoding a sequence of words using SentencePiece tokenizer.
These functions are called inside **DynamicItemDataset** `__getitem__` method and are run in parallel on the CPU.

The **CategoricalEncoder** is a convenient abstraction we provide for multi-class classification problems and it is sub-classed into **TextEncoder** and **CTCTextEncoder** which instead can be used for sequence-to-sequence applications related to text, such as ASR.


Thanks to these abstractions, most of the work necessary to set up the data IO pipeline is parsing the dataset into suitable annotation supported by **DynamicItemDataset** (SpeechBrain supports both CSV and JSON formats).


Once this annotation is ready, a flexible and efficient pipeline can be created in few lines of code, as SpeechBrain will take care under the hood of padding and other operations.



In the following tutorial, we will explain in detail how these blocks work.
We will start from the required CSV or JSON annotation whose purpose is to represent and describe the information contained in the dataset.
For example:

*   Paths to audio files, pre-extracted features et cetera.
*   Metadata such as words spoken in the audio files, Signal-to-Noise-Ratio, sound event tags, speaker identities et cetera.

Basically any information required to train your algorithm.

### Dataset Annotation

SpeechBrain offers native support for JSON and CSV formats for **describing a dataset** and, in fact, in official recipes (such as LibriSpeech ASR recipes) we provide parsing scripts to obtain such formats.  

We can take a glimpse to how these files can be structured using the downloaded Mini-LibriSpeech example.


Each file in Mini-LibriSpeech is an utterance from a single speaker, thus either JSON and CSV formats can be used to contain the absolute path to that file, the speaker identity and the words the speaker utters. This is enough to build an Automatic Speech Recognition (ASR) system.

Below we can take a look at how the JSON file used for the trivial speaker recognition example illustrated before is structured:


```python
from parse_data import parse_to_json # parse_data is a local library downloaded at the step before
parse_to_json("./LibriSpeech/dev-clean-2") # how to build this function will be shown later
```


```python
!head data.json
```

Thus the JSON file is a dictionary where each key is unique and corresponds to a single example (the key is the example ID). Each example is in its turn a dictionary containing the path to the utterance `file_path`, the `words` spoken the identity of the speaker `spkID` and the `length` in samples of the audio in `file_path`.

the CSV file also has an equivalent structure:


```python
# csv file
from parse_data import parse_to_csv
parse_to_csv("./LibriSpeech/dev-clean-2")
!head data.csv
```

Unlike other toolkits which have a rather strict requirements on how a dataset must be specified to be able to process it **we do not impose any restriction on JSON and CSV syntax except the requirement for a different unique ID string for every single example**.

This means that the JSON file must contain a dictionary whose keys are the example ids and each entry of the dictionary contains metadata for that example. Instead, CSV files, must have at least one column called id.

These are the only strict requirements to guarantee that the JSON and CSV dataset description files will work with SpeechBrain data IO pipeline.

Users are given great flexibility in how to represent their datasets in the JSON and CSV files as their goals and applications could be different: speech separation, enhancement, ASR, diarization, VAD et cetera.

This because in SpeechBrain we aim at many different tasks and datasets.

- **Every Dataset is unique**, it can be single channel or multichannel, it can provide different metadata such as speaker IDs, or speaker positions or even
multi-modal data such as audio and video.

- **Every task is unique**, the annotation used in this example is suitable for applications like ASR and Speaker Recognition. But for diarization, for example, the user would like to have also the start and stop (in seconds, frames whatever !) of each uttererance too.

This also allows to keep the annotation very simple and focused to the particular task with only the necessary information for the current application, instead of having a cumbersone do-it-all annotation.


**TIP**

It is useful, when building the parsing script, to have a `length` or `duration` for each example containing the length of the example in seconds or samples or even frames. This allows for subsequent operations such as filtering examples that are too long (to avoid OOM issues) or sorting them for faster training. Regarding CSV files if a `duration` column is specified it is automatically cast to float when a DynamicItemDataset is built from the CSV.




As mentioned, **these files must be produced by a suitable parsing script**, which is provided in our recipes and depends on the dataset and the task.
However if one wish to use its own custom dataset a parsing script must be written in order to produce *JSON* or CSV files describing the data.

**NOTE**

A separate file for each split (training, validation/development, test) must be provided.

On the other hand, creating those files should be rather easy for most datasets as Python offers many tools for manipulating CSV and JSON files (such as [pandas](https://pandas.pydata.org/)). In fact parsing the data to JSON, for example, can be done in few lines of codes:


```python
import glob
import json
import os
import torchaudio
from pathlib import Path

dev_clean_root = "./LibriSpeech/dev-clean-2"
flac_files = glob.glob(os.path.join(dev_clean_root, "**/*.flac"), recursive=True)
print("tot flac audio files {}".format(len(flac_files)))

flac_files[0]
# in this dataset files names are spk_id-chapter_id-utterance_id.flac
text_files = glob.glob(os.path.join(dev_clean_root, "**/*.txt"), recursive=True)
# we build a dictionary with words for each utterance
words_dict = {}
for txtf in text_files:
  with open(txtf, "r") as f:
    lines = f.readlines()
  for l in lines:
    l = l.strip("\n")
    utt_id = l.split(" ")[0]
    words = " ".join(l.split(" ")[1:])
    words_dict[utt_id] = words

# we now build JSON examples

examples = {}
for utterance in flac_files:
  utt_id = Path(utterance).stem
  examples[utt_id] = {"file_path": utterance,
                      "words": words_dict[utt_id],
                      "spkID": utt_id.split("-")[0],
                      "length": torchaudio.info(utterance).num_frames}


with open("data.json", "w") as f:
  json.dump(examples, f, indent=4)

print(examples[list(examples.keys())[0]])
```

### DynamicItemDataset

**DynamicItemDataset** is at the heart of SpeechBrain data pipeline and is built on top of `torch.utils.data.Dataset`.

As the name implies it allows the **dynamical creation of new "objects"** from the entries specified in the JSON (or CSV) dataset annotation.


```python
#`creating a DynamiItemDataset instance from JSON or CSV annotation is immediate
from speechbrain.dataio.dataset import DynamicItemDataset

dataset = DynamicItemDataset.from_json("data.json") # or equivalently, DynamicItemDataset.from_csv("data.csv")
```

What does it mean dynamical creation of "objects" from the entries specified in the data.json annotation ?


```python
dataset[0]
```

As it is now, this `Dataset` object does not return anything.

Dynamical creation means exactly that, **items the user wants to be returned must be specified** in some way by the user. These items can depend on the entries specified in the data.json examples:


```python
print(examples[list(examples.keys())[0]].keys())
```

Namely from `['file_path', 'words', 'spkID', 'length']`.

For example one "dynamic item" could be the audio signal which will depend on the `'file_path'` key. Another one could be the `spkID` encoded to an integer value if one wishes to perform speaker recognition or, for ASR, it could be the words encoded by a tokenizer.

To obtain these "items" one should specify in some way a function which, when applied to the corresponding key, will provide the new item.

E.g. a function which reads the audio when applied to `'file_path'` key. in order to make the `Dataset` class return, for each example. the audio signal.


### Dynamic Item Pipelines (DIPs)

This task is handled by specifying **Dynamic Item Pipelines** for each dynamic item the user wants to get from the dataset. The user can specify an arbitrary number of pipelines.

For example, regarding the audio signal:


```python
@speechbrain.utils.data_pipeline.takes("file_path")
@speechbrain.utils.data_pipeline.provides("signal")
def audio_pipeline(file_path):
      sig = speechbrain.dataio.dataio.read_audio(file_path)
      return sig
```

We specify a function that takes the `file_path` for each example and provides a new item called `sig` which is a tensor containing the audio.


We use here some pre-built function in `speechbrain.dataio.dataio` for reading audio. But the user can also use its own.

Once specified, the pipeline must be added to the `DynamicItemDataset` object and, following, the outputs requested by the user should be specified with the `set_output_keys` method.
We request two items in the output: a new one `sig` and also `file_path` which is in the JSON annotation.


```python
dataset.add_dynamic_item(audio_pipeline)
dataset.set_output_keys(["signal", "file_path"],
    )
dataset[0]
```

Note that a more compact syntax can be used for applying a simple function like reading the audio file.


```python
dataset.add_dynamic_item(speechbrain.dataio.dataio.read_audio, takes="file_path", provides="signal")
dataset.set_output_keys(["id", "signal", "words"])
dataset[0]
```

Now the dataset object will return this new specified item "sig" as well as the `file_path` as specified in the JSON:  


```python
print(dataset[0])
import matplotlib.pyplot as plt
plt.figure(1)
plt.title("Sig item")
plt.plot(dataset[0]["signal"])
plt.show()
```

It can be seen that the `DynamicItemDataset` object returns the two specified items from the JSON annotation.

The `file_path` item, is taken directly, as it is, from the JSON annotation without further processing.
The other one is instead a new item derived from `file_path` item with the pipeline we have defined before.

There is no constraints in what can be done in the pipelines: as said the user can also use its own functions:


```python
!pip install soundfile
import soundfile as sf
@speechbrain.utils.data_pipeline.takes("file_path")
@speechbrain.utils.data_pipeline.provides("sig_numpy")
def audio_pipeline_numpy(file_path):
      sig, _ = sf.read(file_path, dtype="float32")
      return sig
```


```python
speechbrain.dataio.dataset.add_dynamic_item([dataset], audio_pipeline_numpy)
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["signal", "file_path", "sig_numpy"],
    )
dataset[0]
```

The dataset object now also returns the signal as read with the soundfile library instead of only the one with read with the built-in speechbrain function which is based on torchaudio.

**Multiple outputs can be specified by one pipeline by using python generators syntax**.

In the example below three outputs are specified, the last two depend directly from the first one (sig) and are transformed version of this latter: with a random gain factor `rand_gain_sig` and with a constant offset `offset_sig`.


```python
import random
@speechbrain.utils.data_pipeline.takes("file_path")
@speechbrain.utils.data_pipeline.provides("sig", "rand_gain_sig", "offset_sig")
def audio_pipeline(file_path):
      sig = speechbrain.dataio.dataio.read_audio(file_path)
      yield sig
      rand_gain_sig = random.random()*sig
      yield rand_gain_sig
      offset_sig = sig + 1
      yield offset_sig


speechbrain.dataio.dataset.add_dynamic_item([dataset], audio_pipeline)
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["sig", "rand_gain_sig", "offset_sig"],
    )

print(dataset[0])
import matplotlib.pyplot as plt
plt.figure(1)
plt.title("Sig item")
plt.plot(dataset[0]["sig"])

plt.title("Sig item with random gain")
plt.plot(dataset[0]["rand_gain_sig"])

plt.title("Sig item offset")
plt.plot(dataset[0]["offset_sig"])
plt.legend(["sig", "rand_gain_sig", "offset_sig"])
plt.show()
```

This toy example demonstrates that multiple items can be fetched from the same pipeline and dynamically created items can depend on other dynamically created items (`offset_sig` depends on `sig`).

But dynamic items can also depend on dynamically created items from another, pre-specified pipeline:



```python
@speechbrain.utils.data_pipeline.takes("sig")
@speechbrain.utils.data_pipeline.provides("sig_as_python_list")
def to_list_pipeline(sig):
      yield sig.numpy().tolist()


speechbrain.dataio.dataset.add_dynamic_item([dataset], to_list_pipeline)
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["sig_as_python_list"],
    )
dataset[0]["sig_as_python_list"][:10]
```

In this example we have defined a new pipeline which takes `sig` and turns it from `torch.tensor` to a python list obtaining a new dynamic item `sig_as_python_list`.


**NOTE**


Since we are requesting in the output only `sig_as_python_list` which depends itself from `sig`, dynamic items `offset_sig` and `rand_gain_sig` are not computed at all. Only `sig` is computed implicitly as it is necessary to obtain `sig_as_python_list`.


In fact under the hood `DynamicItemDataset` finds a suitable evaluation
order for the requested items by constructing a **computational graph** defined by the **pipelines**.

**An error is returned if any circular dependency is present between the pipelines**.


A **DIP** can also take multiple items/annotation keys in input, the syntax is the same as for the output items:


```python
@speechbrain.utils.data_pipeline.takes("file_path", "spkID")
@speechbrain.utils.data_pipeline.provides("sig", "spkidstring")
def multiple_dip(file_path, spkID):
      sig = speechbrain.dataio.dataio.read_audio(file_path)
      yield sig
      yield spkID

speechbrain.dataio.dataset.add_dynamic_item([dataset], multiple_dip)
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["sig", "spkidstring"],
    )
dataset[0]
```


```python
@speechbrain.utils.data_pipeline.takes("file_path", "spkID")
@speechbrain.utils.data_pipeline.provides("sig")
def multiple_dip(file_path, spkID):
      sig = speechbrain.dataio.dataio.read_audio(file_path)
      yield sig, spkID


speechbrain.dataio.dataset.add_dynamic_item([dataset], multiple_dip)
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["sig"],
    )
dataset[0] # sig now is a tuple
```

And also the same **DIP** can be used in multiple datasets.
E.g. you want usually the read audio **DIP** to be the same for validation, training and test:


```python
validation = DynamicItemDataset.from_json("data.json")
train = DynamicItemDataset.from_json("data.json")

speechbrain.dataio.dataset.add_dynamic_item([validation, train], speechbrain.dataio.dataio.read_audio, takes="file_path", provides="signal")
speechbrain.dataio.dataset.set_output_keys([validation, train], ["id", "signal", "words"])
validation[0]

```


```python
train[0]
```

### CategoricalEncoder

SpeechBrain `dataio` provides a `CategoricalEncoder` class for encoding labels which belongs to a discrete set: e.g. for speaker recognition or any other multi-class classification problem.

Given a collection of hashables (e.g. strings) it encodes
    every unique item to an integer value: `["spk0", "spk1"]` --> `[0, 1]`


  Internally the correspondence between each label to its index is handled by
  two dictionaries: `lab2ind` and `ind2lab`.

It is built to tightly integrate with `DynamicItemDataset` and `dataIO pipelines`.

For example one can obtain the encoding for speaker identities (`spkID` in JSON) from our Mini-LibriSpeech dataset by creating an instance of CategoricalEncoder and fitting it to the dataset object.


```python
from speechbrain.dataio.encoder import CategoricalEncoder
spk_id_encoder = CategoricalEncoder()
```

Since `DynamicItemDataset` right now does not return spkID we have firstly to set its output to return that dynamic item:


```python
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["spkID"],
    )
# sig is a torch.tensor with audio signal as specified before.
# REMEMBER: no need to specify the pipeline for spkID as we can read directly the value from the JSON.
dataset[0]
```

The speaker identity `spkID` is a string.

Note that in librispeech it is a string containing a unique integer so one can argue that performing the encoding here is pointless as casting to integer will suffice.

However, it could happen that it is not an unique integer but an unique string like `spk1`, `spk2` et cetera.

`spk_id_encoder` can be used for this purpose. We fit the encoder to the dataset and specify which *dynamic item* we want to encode:


```python
spk_id_encoder.update_from_didataset(dataset, "spkID")

```

**NOTE**


This will iterate over dataset fetch for each example spkID and construct
internal dicts lab2ind and ind2lab.

Because of this, it is important to call `dataset.set_output_keys` avoiding
computationally costly dynamic items (e.g. can happen if the pipeline does data augmentation) before fitting the encoder.

Setting only the key on which the encoder will be fitted is a good approach.

We can look now at how much unique speaker ids there are in the dataset by using `__len__`


```python
len(spk_id_encoder)
```

We can also take a look to the encoder internal dictionaries `lab2ind` ans `ind2lab` which contains the mappings between the labels (speaker ids in this case)  and corresponding integers encodings:


```python
 spk_id_encoder.lab2ind
# contains label --> integer encoding
```


```python
spk_id_encoder.ind2lab # contains integer encoding --> label
```

Once fitted the `CategoricalEncoder` object can be used in a suitably defined pipeline to encode the spkID key and return the encoded value:


```python
@speechbrain.utils.data_pipeline.takes("spkID")
@speechbrain.utils.data_pipeline.provides("spkid_encoded")
def spk_id_encoding(spkid):
      return torch.LongTensor([spk_id_encoder.encode_label(spkid)])

```


```python
speechbrain.dataio.dataset.add_dynamic_item([dataset], spk_id_encoding)
speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["spkid_encoded"],
    )

dataset[100]
```

### PaddedBatch and SaveableDataLoader

SpeechBrain offers a way to conveniently pad right automatically tensors of different length on multiple dimensions.
This is achieved using the `PaddedBatch` class defined in `speechbrain.dataio.batch`.

`PaddedBatch` is both a `collate_fn` and a batch object.

When a `torch.utils.data.Dataset` (and thus also a `DynamicItemDataset`) is passed to the `Brain` class `PaddedBatch` is used as the default collate function `collate_fn` and examples are automatically padded together.

As the default DataLoader the Brain class instantiates, a SpeechBrain custom DataLoader: `speechbrain.dataio.dataloader.SaveableDataLoader`.


This DataLoader is identical to the plain one except that it allows for intra-epoch saving. So if for some reason training stops in the middle of an epoch it is possible to resume from exactly that step. See the [Checkpointing Tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/checkpointing.html).
The default `collate_fn` for this DataLoader is `PaddedBatch`.



```python
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch

speechbrain.dataio.dataset.set_output_keys(
        [dataset], ["id", "spkid_encoded", "signal"],
    )
```

We set as the dynamic items we are requesting `sig` which is the audio tensor
and the speaker id encoded with the `CategoricalEncoder` object (`spkid_encoded`) defined before and the example `id`.


```python
dataloader = SaveableDataLoader(dataset, batch_size=2, collate_fn=PaddedBatch)
```


```python
batch_obj = next(iter(dataloader)) # let's look at the batch obj
```


```python
batch_obj # the dataloader returns an PaddedBatch obj now
type(batch_obj)
```

Dynamic Items can be accessed in the batch object by using `dict` syntax:


```python
batch_obj["spkid_encoded"]
```


```python
batch_obj["signal"]
```


```python
batch_obj["id"] # example ids in this batch useful for debugging
```

As said, all elements in PaddedBatch which are `torch.Tensors` are padded together by adding zeros to the right.
When these elements are accessed a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) is returned: the actual padded tensors and a `length` tensor.


```python
wav_data, length = batch_obj["signal"]
```

As it is a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) the two items *length* and *data* are also accessible as attributes:


```python
lengths = batch_obj["signal"].lengths
wav_data = batch_obj["signal"].data
```


```python
lengths
```

This length tensor contains the relative true length of each sequence.
In this example it means that the second example in the batch has not been padded (relative length == 1) while the first instead has been padded to more twice its length.

The use of relative lengths instead of absolute indexes guarantees that that these values do not change even after feature extraction: the relative true length remains the same even after STFT whatever is the window.

The absolute indexes are easy to obtain:


```python
abs_lens = (lengths*wav_data.shape[1]).long()
abs_lens
```


```python
wav_data[1][:abs_lens[1]] # no zeros
```


```python
wav_data[1][abs_lens[1]:] # zeros begins at abs_lens[0]
```

The PaddedBatch object allows for conveniently moving all dynamic items which are` torch.Tensor` to the right device using to:


```python
batch_obj = batch_obj.to("cpu")
```

Of course items which are not tensors such as `id` are not moved and are not padded. Instead they are simply returned as a list.


```python
batch_obj["id"]
```

It is also possible to iterate over the examples of `PaddedBatch`:


```python
for ex in batch_obj:
  print(ex)
```

And access a single example by its position:


```python
batch_obj.at_position(1)
```

These methods can be conveniently used in the `Brain` class `compute_forward` and `compute_objectives` methods.
As we have shown in the first example of this tutorial where a complete dataIO example was illustrated:




```python
def compute_forward(self, batch, stage):
    audio, audio_len = batch["sig"]
    # the examples are automatically padded, audio_len contains the relative
    # length of the original sequence.
    return self.modules.model(audio.unsqueeze(1)).mean(-1).unsqueeze(-1)
    
  def compute_objectives(self, logits, batch, stage):
    spk_ids, _ = batch["spkid_encoded"]
    return torch.nn.functional.cross_entropy(logits, spk_ids)
```



## Full Example: Training a simple Speaker Recognition System.

Hereafter we show how the **DynamicItemDataset**, **DIPs** and **CategoricalEncoder** can be used to build a data pipeline for Speaker Recognition.

In particular we have to:

- read the audio
- read the speaker ID from annotation and encode it to integer


We firstly instantiate the dataset from that JSON annotation


```python
dataset = DynamicItemDataset.from_json("data.json")
```

Then fit the **CategoricalEncoder** to speaker IDs (`spkID`) in the annotation.


```python
spk_id_encoder = CategoricalEncoder()
spk_id_encoder.update_from_didataset(dataset, "spkID")
```

We add the **DIP** which encodes `spkID`


```python
dataset.add_dynamic_item(spk_id_encoder.encode_label_torch, takes="spkID", provides="spk_encoded")

```

We add the **DIP** for reading the audio



```python
dataset.add_dynamic_item(speechbrain.dataio.dataio.read_audio, takes="file_path", provides="signal")

```

and set the outputs of the dataset we want to access in training loop


```python
dataset.set_output_keys(["id", "signal", "spk_encoded"])
```

We sort the dataset based on length to speed up training so that we minimize in batches the amount of padded elements.


```python
sorted_data = dataset.filtered_sorted(sort_key="length")
```

We can train now a simple classifier, by passing the dataset object directly to the **Brain** class. The **Brain** class will automatically create a SaveableDataLoader with specified `train_loader_kwargs` and will be handling the padding for you.


```python
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss


class SimpleBrain(speechbrain.Brain):
    def compute_forward(self, batch, stage):
        example_batch = batch
        x = self.modules.features(batch.signal.data)
        x = self.modules.encoder(x)
        x = self.modules.pooling(x, batch.signal.lengths)
        x = self.modules.to_output(x)
        return self.modules.softmax(x)

    def compute_objectives(self, logits, batch, stage):
        return nll_loss(logits, batch.spk_encoded.data)

modules = {"features": Fbank(left_frames=1, right_frames=1),
          "encoder": torch.nn.Sequential(torch.nn.Linear(40, 256),
                                       torch.nn.ReLU()),
          "pooling": speechbrain.nnet.pooling.StatisticsPooling(),
          "to_output": torch.nn.Linear(512, len(spk_id_encoder)),
          "softmax": speechbrain.nnet.activations.Softmax(apply_log=True)}
brain = SimpleBrain(modules, opt_class=lambda x: torch.optim.SGD(x, 0.01))
brain.fit(range(1), train_set=sorted_data,
          train_loader_kwargs={"batch_size": 16, "drop_last":True})
```

## Acknowledgements



*   Many thanks to Nasser Benabderrazik ([lenassero](https://github.com/lenassero)) who helped improving this Tutorial.





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
