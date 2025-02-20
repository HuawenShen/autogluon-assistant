Summary: This tutorial covers SpeechBrain's text tokenization implementation using SentencePiece, focusing on BPE and Unigram tokenization methods. It provides code examples for tokenizer initialization, basic tokenization/detokenization operations, PyTorch integration, and data pipeline setup. Key functionalities include configurable vocabulary size, character coverage control, custom symbol inclusion, and both piece-based and ID-based encoding/decoding. The tutorial helps with tasks like implementing text preprocessing for speech recognition, managing large vocabularies, and integrating tokenization into SpeechBrain's data pipeline, making it particularly useful for speech-to-text applications.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/text-tokenizer.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/advanced/text-tokenizer.ipynb)

# Text Tokenization

## Why do we need tokenization?

Almost all languages have a huge number of possible words. Machine learning tasks that process text have thus to support large vocabularies that might contain several thousands of words.  Dealing with such a large vocabulary, however, is critical.  The input and output embeddings  (e.g. one-hot-vectors) are normally huge vectors, leading to and increase memory consumption and memory usage. More importantly,  learning with such extremely sparse and high-dimensional embeddings might be sub-optimal.

A naive alternative can be to simply use characters instead of words.
The latter approach alleviates some of the aforementioned issues, but
it requires processing a longer sequence  (that is critical as well from a machine learning point of view).

Can we find a middle ground between words and characters? Yes, this is what the tokenizer is trying to do.

One popular technique called **rule-based tokenization** (e.g. [spaCy](https://spacy.io)) allows splitting the text into smaller chunks based on grammar rules, spaces, and punctuation. Unfortunately, this approach is language-dependent and must be set for each language considered ...

Another solution to get the best of both word-level and character-level tokenizations is a hybrid solution named **subword tokenization** relying on the principle that frequently-used words should not be split into smaller subwords, but rare words should be decomposed into meaningful (i.e. more frequent) subwords.


SpeechBrain currently relies on a custom integration of the [*SentencePiece tokenizer*](https://github.com/google/sentencepiece) which treats the input as a raw input stream. The following tokenizer algorithms are supported:
1. [BPE](https://web.archive.org/web/20230319172720/https://www.derczynski.com/papers/archive/BPE_Gage.pdf).
2. [Unigram](https://arxiv.org/pdf/1804.10959.pdf) (Subword Regularization).


The *SentencePiece tokenizer* is available at `speechbrain.tokenizer.SentencePiece`. In the following, we will describe all the aforementioned techniques, but first of all, let's install SpeechBrain.




```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH

# Clone SpeechBrain repository
!git clone https://github.com/speechbrain/speechbrain/
%cd /content/speechbrain/
```

Let's also download a csv file to train our tokenizer.



```python
%%capture
!wget https://www.dropbox.com/s/atg0zycfbacmwqi/dev-clean.csv
```

## Train sentencepiece tokenizer within SpeechBrain
SentencePiece is a class that can be instantiated with few parameters


*   **model_dir**: it is the directory where the trained tokenizer model is saved. The model will be saved as *`model_dir/model_type_vocab_size.model`*
*   **vocab_sizes**: It is the vocabulary size for the chosen tokenizer type (BPE, Unigram). The vocab_size is optional for character tokenization and mandatory for BPE & unigram tokenization.
* **csv_train**: It is the path of the csv file which is used to learn the tokenizer.
* **csv_read**: It is the data entry (csv header) which contains the word sequence in the csv file.
* **model_type**: It can be: word, char, bpe, or unigram tokenization.

Let's now apply it to our dev-clean.csv.


```python
import torch
from speechbrain.tokenizers.SentencePiece import SentencePiece
```


```python
spm = SentencePiece(model_dir="tokenizer_data",
                    vocab_size=2000,
                    annotation_train="dev-clean.csv",
                    annotation_read="wrd",
                    model_type="bpe",
                    annotation_list_to_check=["dev-clean.csv"])
```


```python
%less tokenizer_data/2000_bpe.vocab
```

As you can see, SetencePiece lib is an unsupervised text tokenizer and detokenizer.  Some of the tokens have `_` symbols representing spaces. The sentence piece detokenization will simply merge the sequence of tokens and replace `_` with spaces.

### Advanced parameters
* `character_coverage`: it is the number of characters covered by the model (value between [0.98 - 1]). default: 1.0 for languages with a small character set. It can be set to 0.995 for languages with rich characters set like Japanese or Chinese.
* `bos_id/eos_id/pad_id/unk_id`: allow users to define specefic index for `bos/eos/pad and unk` tokens
* `split_by_whitespace`: this parameter allows sentencepiece to extract crossword pieces and consider space as a unique token.
* `num_sequences`: use at most `num_sequences` to train the tokenize (limit the training text for large datasets).
* `csv_list_to_check`: List of csv files used for checking the accuracy of recovering words from the tokenizer.
* `user_defined_symbols`: it is a string list (separated by comma ',') which force the insertion of specific vocabulary.

As an example, if we set the `character_coverage` to `0.98` and reduce the `vocab_size`:


```python
spm = SentencePiece(model_dir="tokenizer_data",
                    vocab_size=500,
                    annotation_train="dev-clean.csv",
                    annotation_read="wrd",
                    model_type="unigram",
                    character_coverage=0.98,
                    annotation_list_to_check=["dev-clean.csv"])
```

As we can see, we are not able to recover all the words from the text because some characters are missing.

## Loading a pre-trained sentence piece tokenizer within SpeechBrain
Loading the sentencepiece tokenizer is very simple. We just need to specify the path of the model,  the `vocab_size`, and the `model_type`:



```python
spm = SentencePiece(model_dir="tokenizer_data",
                    vocab_size=2000,
                    model_type="bpe")
```

Now, we can directly use the tokenizer loaded from `tokenizer_data/2000_bpe.model`. This feature is very useful to replicate results. As an example, you can upload your tokenizer to the internet and someone else can download it to obtain the same tokenization as you.

## How to use the sentencepiece

The SentencePiece object is available at `speechbrain.tokenizer.SentencePiece.sp`. By accessing this object, you can easily perform tokenization and detokenization. If interested in all the features of SentencePiece, please feel free to read the [official tutorial](https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=uzBiPAm4ljor)

Let's try to tokenize and detokenize some text!


```python
# Encode as pieces
print(spm.sp.encode_as_pieces('THIS IS A TEST'))
# Encode as ids
print(spm.sp.encode_as_ids('THIS IS A TEST'))
```


```python
# Decode from ids
print(spm.sp.decode_ids([244, 177, 3, 1, 97]))
# Decode from pieces
print(spm.sp.decode_pieces(['▁THIS', '▁IS', '▁A', '▁T', 'EST']))
```

## Use SpeechBrain SentencePiece with Pytorch
We designed our SentencePiece wrapper to be used jointly to our data transform pipeline [(see the tutorial)](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/data-loading-pipeline.html) and therefore deal with tensors.
For that purpose, two options are available:
1. Option 1: Generating token tensors directly from a word tensors + an external dictionary named `int2lab` (which maps your tensors to words).
1. Option 2: If you use our DynamicDataset, the DynamicItem will automatically generate the token tensors.


### Example for option 1


```python
# INPUTS
# word vocab
dict_int2lab = {1: "HELLO", 2: "WORLD", 3: "GOOD", 4:"MORNING"}
# wrd tensors
wrd_tensor = torch.Tensor([[1, 2, 0], [3,4,2]])
# relative lens tensor (will help for dealing with padding)
lens_tensor = torch.Tensor([0.75, 1.0])
```

Our SentencePiece can be called like any other pytorch function with the tensors passed to the __call__ method. Parameters are given as:
batch : it is a word_ids tensor (i.e. your words). Shape: [batch_size, max_seq_lenght]
batch_lens: it is a relative length tensor. shape: [batch_size]
int2lab: dictionary which maps the word_ids to the word.
task:
"encode": convert the word batch tensor into a token tensor.
"decode": convert the token tensor into a list of word sequences.
"decode_from_list": convert a list of token sequences to a list of word sequences.


```python
encoded_seq_ids, encoded_seq_lens = spm(
        wrd_tensor,
        lens_tensor,
        dict_int2lab,
        "encode",
    )
# tokens tensor
print(encoded_seq_ids)
# relative lens token tensor
print(encoded_seq_lens)
```

Then we can simply decode it by simply specifying `"decode"` to the function!


```python
# decode from torch tensors (batch, batch_lens)
words_seq = spm(encoded_seq_ids, encoded_seq_lens, task="decode")
print(words_seq)
```

### Example for option 2

**Note:** please first read our dataio [tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/data-loading-pipeline.html) to perfectly grasp the next lines.

Here, we use a tokenizer to tokenize on-the-fly the text obtained from a .csv file. In the following example, we combined  it with the data_io pipeline of SpeechBrain.

First, we define a DynamicItemDataset from our csv file:


```python
import speechbrain as sb
train_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path="dev-clean.csv",
)
```


```python
%less dev-clean.csv
```

Then, we define the text_pipeline (i.e. what is called for each sample gathered in a mini-batch). In the text_pipeline, we simply call our tokenizer to obtain the tokenized text!


```python
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = spm.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens
```

Some more SpeechBrain stuff to finalize the data pipeline:


```python
train_set.add_dynamic_item(text_pipeline)
train_set.set_output_keys(["wrd", "tokens", "tokens_list"])
```

Finally, we create a data loader that contains the defined transformation (i.e. tokenizer).


```python
train_dataloader = sb.dataio.dataloader.make_dataloader(train_set, batch_size=1)
```

Now, we can simply get our tokenized samples !!


```python
b = next(iter(train_dataloader))
print(b.wrd)
print(b.tokens)
print(b.tokens_list)
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
