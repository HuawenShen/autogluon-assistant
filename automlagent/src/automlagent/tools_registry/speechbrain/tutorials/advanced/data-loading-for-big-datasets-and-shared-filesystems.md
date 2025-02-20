Summary: This tutorial demonstrates implementing efficient data loading for large datasets using WebDataset and TAR archives, particularly useful for speech recognition tasks with SpeechBrain. It covers techniques for creating and managing data shards, configuring dynamic batching, and optimizing memory usage through sequential streaming. Key functionalities include TAR shard creation, WebDataset integration with SpeechBrain, dynamic batch sizing, and epoch handling. The implementation helps solve challenges with large-scale data loading from shared filesystems, offering code examples for dataset creation, shard writing, and dataloader configuration while emphasizing best practices for performance optimization and memory management.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/data-loading-for-big-datasets-and-shared-filesystems.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/advanced/data-loading-for-big-datasets-and-shared-filesystems.ipynb)

# Data Loading for Big Datasets and Shared Filesystems

Do you have a large dataset stored in a shared filesystem, and you want to use it for training a neural network? Is this dataset so large that it doesn't even fit into the local SSD of your computation nodes? If so, this tutorial will walk you through all the needed steps to manage reading large files from a shared filesystem.

In many compute clusters, the main data storage is a network filesystem (NFS), for example [Lustre](https://en.wikipedia.org/wiki/Lustre_(file_system)). <!-- ignore-url-check --> The NFS can serve many users concurrently and provide high data throughput from a single file. However, opening or listing many different files is slow - and doing so may slow the whole system down for everyone, not just the offending user. Speech datasets usually consist of very many small recordings. Reading every file again and again is exactly the kind of data IO that can slow down an NFS.

One solution is to copy the dataset into the **local SSD** of the computing node. This can be done relatively efficiently by compressing the dataset into a single file (e.g. `dataset.tar.gz`), copying it into the local node, and finally, uncompressing (untarring) the file. Reading files from the local SSD is very efficient and does not harm the performance of the shared filesystem.
The standard SpeechBrain data IO works well in this case, see [this tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/data-loading-pipeline.html).
However, there might be huge datasets that exceed the size of your local SSD.  

A possible workaround is to keep the data in the shared filesystem and bundle the small recordings into larger archives, which are usually called **shards**. Loading data off shards avoids opening too many files, so it is fast.

When reading data from shards, random access across the dataset is no longer possible. Data is read sequentially, from a **stream**. This requires a bit of care in preparing the experiment.

The case for sharded IO laid out above is typical in an academic compute-cluster setup. Streaming data IO can also be used in even larger scales with dedicated data servers.

In this tutorial we will use the **WebDataset** library. Alternatives and the case for WebDataset are laid out by the WebDataset developer in [this PyTorch proposal](https://github.com/pytorch/pytorch/issues/38419).



## What is WebDataset?

[WebDataset](https://github.com/webdataset/webdataset) is a sharded (streaming) data IO library that works well with PyTorch. WebDataset uses standard TAR archives as the shard format, with the simple convention that all consequtive files that have the same basename belong to the same example. So listing `data-archive/shard-0000.tar`s contents can look like:

```
> tar -t data-archives/shard-0000.tar
spk1-utt1.wav
spk1-utt1.txt
spk1-utt1.json
spk1-utt2.wav
spk1-utt2.txt
spk1-utt2.json
spk2-utt1wav
spk2-utt1.txt
spk2-utt1.json
...
```

On the Python side, the dataset interface is an [IterableDataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets), and it has a set of methods which can be chained to build a data pipeline, like:

```python
import webdataset as wds  # Note the typical import shorthand
dataset = (
      wds.WebDataset("data-archives/shard-00{00...24}.tar")  # 25 shards
      .decode()  # Automagically decode files
      .shuffle(size=1000)  # Shuffle on-the-fly in a buffer
      .batch(batchsize=10)  # Create batches
)
```

Note that WebDataset is (at least at the time of writing) a fast moving library. It is also being considered for inclusion in the PyTorch core. Again, read more [here](https://github.com/pytorch/pytorch/issues/38419) and [here](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/).



## Installing dependencies


```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH
```


```python
%%capture
!pip install "webdataset<0.2"

```


```python
import speechbrain as sb
import webdataset as wds
import torch
import glob
import pathlib
import random
```

## Creating TAR shards

The data-prep process in WebDataset is to iterate over each example in the dataset and split them into TAR shards. TAR files are a standard format, so you could create them with any standard tools. WebDataset has a couple of helpers, which can make this process a bit easier.

- Tarp, a Go-based tool which can split a TAR stream into shards, and do a couple of other streaming processing tasks. See [the GitHub page](https://github.com/webdataset/tarp). This is a separate tool, and requires a separate installation, but Go could theoretically be faster than Python.
- `wds.ShardWriter`, a Python class that can write WebDataset style `dict`s into TAR archives, splitting into multiple shards of a given size. This is the approach we will take here.




### Download some data

In this tutorial we will work with the development set of Mini Librispeech (but we will treat it like any normal training data).


```python
%%capture
!wget https://www.openslr.org/resources/31/dev-clean-2.tar.gz
!tar -xvzf dev-clean-2.tar.gz
!rm dev-clean-2.tar.gz
```

### Iterate over the data

This step will of course vary dataset to dataset. In Mini Librispeech, the data is organized by speaker and by document. We will first read all transcripts in, then shuffle those, so that consecutive examples are not from the speaker and document.




```python
DATAROOT = pathlib.Path("LibriSpeech/dev-clean-2")
SHARDSDIR = pathlib.Path("DATA-SHARDS")
SHARDSDIR.mkdir(exist_ok=True, parents=True)

# 1. Gather texts
# Note that here uttid encodes speaker and document IDs, so we don't need to
# keep track of them separately
texts = {}
for textf in DATAROOT.glob("*/*/*.trans.txt"):
    with open(textf) as fi:
        for line in fi:
            uttid, text = line.split(" ", maxsplit=1)
            texts[uttid] = text
            print(uttid, text)

# 2. Shuffle uttids
uttids = list(texts.keys())
random.shuffle(uttids)
```


```python
print(uttids)
```


```python
# 3. Create TARs
# In this example, we are only storing 100 examples / shard, because the full
# development set could probably fit in a normal shard. In practical setups
# use bigger values.
# maxcount sets the max number of examples, and maxsize
# sets the maximum size in bytes.

# 3A. Iterate over the shuffled uttids
# 3B. For each uttid, create an example dict
#   The example dict is written into a TAR stream. The special __key__
#   entry becomes the basename for this example's files, and the other
#   entries in the dict become files with different extensions.
#   E.G. with uttid "3536-23268-0007" this will write the files:
#     3536-23268-0007.audio.pth, 3536-23268-0007.text
#   There are default handlers for many extensions
#     See https://github.com/webdataset/webdataset/blob/6ee2279795b3f667bb7a5868af596990cc6efee3/webdataset/writer.py#L97

with wds.ShardWriter(f"{SHARDSDIR}/shard-%06d.tar", maxcount = 100) as writer:
    for uttid in uttids:
        spk, doc, _ = uttid.split("-")
        audio_fpath = (DATAROOT / spk / doc / uttid).with_suffix(".flac")
        audio_tensor = sb.dataio.dataio.read_audio(str(audio_fpath))
        example = {
            "__key__": uttid,
            "audio.pth": audio_tensor,
            "text": texts[uttid]
        }
        writer.write(example)

```


```python
! cd DATA-SHARDS/
```


```python
# Now we can load these shards.
# This uses the SpeechBrain batch class, but batching itself is done by
# WebDataset
dataset = (
      wds.WebDataset(str(SHARDSDIR)+"/shard-0000{00..10}.tar")
      .decode()
      .shuffle(100)
      .batched(batchsize=10,
               collation_fn=sb.dataio.batch.PaddedBatch)
)
batch = next(iter(dataset))
print(batch.text)
print(batch["audio.pth"])  # Because of the audio.pth name, attribute access doesn't work
print("How much of batch is padding [%]:",
      sb.dataio.iterators.padding_ratio(batch["audio.pth"].lengths).item()*100)
```

## WebDataset with SpeechBrain

SpeechBrain is compatible with any PyTorch data loading, so WebDataset can be used without any extensions (as we have done so far). However, three problems remain:

1.   Data in shards is typically not sorted (or is even deliberately shuffled). Consecutive utterances will be of very different lengths, and need a lot of padding.
2.   The intra-epoch checkpointing in the SaveableDataLoader does not work with IterableDatasets.
3.   Exact epochs are very difficult to acheive with Distributed Data Parallel. (This problem is not specific to WebDataset or SpeechBrain.)

These problems are solved with the following strategies and extensions:

1. SpeechBrain implements an on-the-fly dynamic batching and bucketing iterator. This works together with `webdataset.WebDataset`.
  - Bucketing puts similar length utterances in the same batch, reducing the amount of padding.
  - Dynamic batching is natural to implement at the same time as bucketing, and aims to yield batches that have similar total number of elements. Batches with short utterances have larger batch size and batches with long utterances have a smaller batch size.
  - On the fly operation is needed for streaming data loading.
2. Don't care about exact epochs. Instead measure the number of updates and set a nominal epoch length (e.g. an epoch = 2500 updates).
3. Don't care about exact restarts: when an experiment is restarted, data loading will not continue from the examples where it left off, but instead just begin again from randomly assigned shards.





### Some changes in the train data loading pipeline

- First of all, use `.rename` in the loading pipeline to get more sensibly named batch elements. This will also solve the issue (above) where `audio.pth` couldn't be accessed with typical attribute style.
- Then add a `.repeat` so that an infinite stream of data is used.
- Finally, the main change is to use `sb.dataio.iterators.dynamic_bucketed_batch` as the batching method
  - A generic iterator can be used with the `.then` method
  - See [the documentation](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.dataio.iterators.html) for arguments.
  - Since this also involves a shuffling operation, don't use the WebDataset shuffle anymore.


```python
dataset = (
      wds.WebDataset(str(SHARDSDIR)+"/shard-0000{00..10}.tar")
      .decode()
      .rename(id="__key__", signal="audio.pth", text="text")  # Mention all, even text.
      .repeat()
      .then(sb.dataio.iterators.dynamic_bucketed_batch,
            len_key = "signal",  # Which batch element's length to consider
            sampler_kwargs={
                "target_batch_numel":16000*45.,  # Add examples till they total 45 seconds
                "max_batch_numel":   16000*60.   # ... but so that they don't go over 60 seconds
            }
      )
)

batch = next(iter(dataset))
print("Batch size:", len(batch))
print("How much of batch is padding [%]:",
      sb.dataio.iterators.padding_ratio(batch.signal.lengths).item()*100)
```

### More complex data loading pipelines

- You can use `.map()` to implement arbitrary processing.



```python
text_mapping = {"<PADDING>": 0}
index = 1
for example in wds.WebDataset(str(SHARDSDIR)+"/shard-0000{00..10}.tar").decode():
    for word in example["text"].split():
        if word not in text_mapping:
            text_mapping[word] = index
            index += 1

def text_to_index(sample):
    """Adds text_vec entry, a LongTensor for text"""
    sample["text_vec"] = torch.LongTensor(
        [text_mapping[word] for word in sample["text"].split()]
    )
    return sample

```


```python
dataset = (
      wds.WebDataset(str(SHARDSDIR)+"/shard-0000{00..10}.tar")
      .decode()
      .rename(id="__key__", signal="audio.pth", text="text")
      .map(text_to_index)
      .repeat()
      .then(sb.dataio.iterators.dynamic_bucketed_batch,
            len_key = "signal",  # Which batch element's length to consider
            sampler_kwargs={
                "target_batch_numel":16000*45.,  # Add examples till they total 45 seconds
                "max_batch_numel":   16000*60.   # ... but so that they don't go over 60 seconds
            }
      )
)
batch = next(iter(dataset))
print(batch.text[0])
print(batch.text_vec.data[0])
```

### How to handle the DataLoader

- Since we have a dataset that returns batches (as opposed to single examples), the DataLoader should set `batch_size=None`
  - The `Brain` class (and the underlying `sb.dataio.dataloader.make_dataloader`) will set this automatically if your Dataset is from WebDataset).
- To acheive the nominal epochs, SpeechBrain has [`sb.dataio.dataloader.LoopedLoader`](https://github.com/speechbrain/speechbrain/blob/4022f5307ae23f1415e44a9c8b8b9cc5994a945b/speechbrain/dataio/dataloader.py#L258)
  - The `Brain` class (and the underlying `sb.dataio.dataloader.make_dataloader`) will use this if you specify `looped_nominal_epoch` in `train_loader_kwargs` (when calling `.fit()`).
  - The `Brain` class will also automatically add this to the checkpointer, so that it is saved in checkpoints (and it also works for intra-epoch checkpoints).


```python
dataloader = sb.dataio.dataloader.make_dataloader(dataset, looped_nominal_epoch=5)
for epoch in range(1,6):
    print("Epoch", epoch)
    for ind, batch in enumerate(dataloader, start=1):
        print("\tBatch", ind, ": batch size", len(batch))
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
