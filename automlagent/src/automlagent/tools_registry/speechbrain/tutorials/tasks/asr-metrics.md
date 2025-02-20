Summary: This tutorial provides implementation details for various Automatic Speech Recognition (ASR) evaluation metrics using SpeechBrain. It covers code implementations for traditional metrics like Word Error Rate (WER) and Character Error Rate (CER), as well as advanced semantic metrics including Part-of-speech Error Rate (POSER), Lemma Error Rate (LER), Embedding Error Rate (EmbER), BERTScore, and SemDist. The tutorial demonstrates how to configure and use these metrics with specific dependencies (SpeechBrain, Spacy, Flair), handle embeddings, and integrate language models. It's particularly useful for tasks involving ASR evaluation, semantic similarity assessment, and natural language processing metric implementations.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/tasks/asr-metrics.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/tasks/asr-metrics.ipynb)

# Metrics for Speech Recognition

Estimating the accuracy of a speech recognition model is not a trivial problem. The Word Error Rate (WER) and Character Error Rate (CER) metrics are standard, but some research has been trying to develop alternatives that better correlate with human evaluation (such as SemDist).

This tutorial introduces some alternative ASR metrics and their flexible integration into SpeechBrain, which can help you research, use or develop new metrics, with copy&paste-ready hyperparameters.

SpeechBrain v1.0.1 via [PR #2451](https://github.com/speechbrain/speechbrain/pull/2451) introduced support and tooling for the metrics suggested by [Qualitative Evaluation of Language Model Rescoring in Automatic Speech Recognition](https://www.isca-archive.org/interspeech_2022/roux22_interspeech.pdf). **We recommend that you read this, as some of the metrics won't be explained in detail here.**


```python
%%capture
# Installing SpeechBrain via pip
BRANCH = 'develop'
!python -m pip install git+https://github.com/speechbrain/speechbrain.git@$BRANCH
%pip install spacy
%pip install flair
```

Some boilerplate and test data downloading follows...


```python
from hyperpyyaml import load_hyperpyyaml
from collections import defaultdict
```


```python
%%capture
!wget https://raw.githubusercontent.com/thibault-roux/hypereval/main/data/Exemple/refhyp.txt -O refhyp.txt
```


```python
!head refhyp.txt
```


```python
refs = []
hyps = []

# some preprocessing for the example file + load uposer mapping to a test file

def split_norm_text(s: str):
    # s = s.replace("' ", "'")

    if s != "":
        return s.split(" ")

    return s

with open("refhyp.txt") as f:
    for refhyp in f.read().splitlines():
        if len(refhyp) <= 1:
            continue

        refhyp = refhyp.split("\t")
        refs.append(split_norm_text(refhyp[0]))
        hyps.append(split_norm_text(refhyp[1]))

with open("uposer.json", "w") as wf:
    wf.write("""[
    ["ADJ", "ADJFP", "ADJFS", "ADJMP", "ADJMS"],
    ["NUM", "CHIF"],
    ["CCONJ", "COCO", "COSUB"],
    ["DET", "DETFS", "DETMS", "DINTFS", "DINTMS"],
    ["X", "MOTINC"],
    ["NOUN", "NFP", "NFS", "NMP", "NMS"],
    ["PRON", "PDEMFP", "PDEMFS", "PDEMMP", "PDEMMS", "PINDFP", "PINDFS",
    "PINDMP", "PINDMS", "PPER1S", "PPER2S", "PPER3FP", "PPER3FS", "PPER3MP",
    "PPER3MS", "PPOBJFP", "PPOBJFS", "PPOBJMP", "PPOBJMS", "PREF", "PREFP",
    "PREFS", "PREL", "PRELFP", "PRELFS", "PRELMP", "PRELMS"],
    ["ADP", "PREP"],
    ["VERB", "VPPFP", "VPPFS", "VPPMP", "VPPMS"],
    ["PROPN", "XFAMIL"],
    ["PUNCT", "YPFOR"]
]
""")
```

## Word Error Rate (WER)

The usual WER metric, which is derived from the Levenshtein distance between the **words** of the reference and hypothesis (i.e. ground truth and prediction respectively). The output is often presented as a percentage, but it can actually exceed 100%, e.g. if you have a lot of insertions.

Of course, what WER is achievable is depends _very_ heavily on the dataset, and on the language to an extent. On some easy datasets, it can get as low as 1%, and good models on harder datasets can struggle to reach 15%, or even worse in challenging conditions.

The WER is defined as the following (where `#` means "number of"):

$\dfrac{\#insertions + \#substitutions + \#deletions}{\#refwords}$

To understand what exactly is an insertion/subtitution/deletion, you should understand the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), an edit distance.  
Roughly speaking, an insertion is a word your model has predicted but does not exist in the reference, a substitution is a word your model has gotten wrong or spelled incorrectly, and a deletion is a word your model has incorrectly omitted.

A limitation of the WER is that all errors are weighed equally. For example, a typo from "processing" to "procesing" does not meaningfully alter meaning, but an error from "car" to "scar" might drastically alter meaning, yet both are considered a single-word and single-character error. This can result in drastic discrepancies between the WER/CER and human evaluation.


```python
wer_hparams = load_hyperpyyaml("""
wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats
""")
```


```python
wer_hparams["wer_stats"].clear()
wer_hparams["wer_stats"].append(
    ids=list(range(len(refs))),
    predict=hyps,
    target=refs,
)
wer_hparams["wer_stats"].summarize()
```

## Character Error Rate (CER)

The typical CER measure, for reference. The CER works the same as the WER, but instead operates at character level (not word or token level).  
Ultimately, the CER penalizes various errors differently. Small typos (e.g. missed accents) would result in a full substitution error with the WER, but only result in one character substitution error with the CER. This isn't necessarily an upside since single-character errors can still alter meaning.

This is slower to run as the edit distance needs to be computed over a comparatively much longer sequence.


```python
cer_hparams = load_hyperpyyaml("""
cer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats
    split_tokens: True
""")
```


```python
cer_hparams["cer_stats"].clear()
cer_hparams["cer_stats"].append(
    ids=list(range(len(refs))),
    predict=hyps,
    target=refs,
)
cer_hparams["cer_stats"].summarize()
```

## Part-of-speech Error Rate (POSER)


```python
poser_hparams = load_hyperpyyaml("""
wer_stats_dposer: !new:speechbrain.utils.metric_stats.ErrorRateStats

uposer_dict: !apply:speechbrain.utils.dictionaries.SynonymDictionary.from_json_path
    path: ./uposer.json
wer_stats_uposer: !new:speechbrain.utils.metric_stats.ErrorRateStats
    equality_comparator: !ref <uposer_dict>

pos_tagger: !apply:speechbrain.lobes.models.flair.FlairSequenceTagger.from_hf
    source: "qanastek/pos-french"
    save_path: ./pretrained_models/
""")
```


```python
refs_poser = poser_hparams["pos_tagger"](refs)
hyps_poser = poser_hparams["pos_tagger"](hyps)
```


```python
print(" ".join(refs_poser[0]))
print(" ".join(hyps_poser[0]))
```

### dPOSER

Instead of computing WER on input words, we extract (preferably all) the parts-of-speech of the input sentences. The WER is then computed over the sequence of labels.


```python
poser_hparams["wer_stats_dposer"].clear()
poser_hparams["wer_stats_dposer"].append(
    ids=list(range(len(refs))),
    predict=hyps_poser,
    target=refs_poser,
)
poser_hparams["wer_stats_dposer"].summarize()
```


### uPOSER

The cited paper proposes a variant (uPOSER) with broad POS categories, in case that the used POS model has very specific categories. This can simply be implemented by using a synonym dictionary that groups up equivalent labels easily.


```python
poser_hparams["wer_stats_uposer"].clear()
poser_hparams["wer_stats_uposer"].append(
    ids=list(range(len(refs))),
    predict=hyps_poser,
    target=refs_poser,
)
poser_hparams["wer_stats_uposer"].summarize()
```

## Lemma Error Rate (LER)

Instead of computing the WER over words, we compute the WER over lemmatized words.


```python
%%capture
!spacy download fr_core_news_md
```


```python
ler_hparams = load_hyperpyyaml("""
ler_model: !apply:speechbrain.lobes.models.spacy.SpacyPipeline.from_name
    name: fr_core_news_md
    exclude: ["tagger", "parser", "ner", "textcat"]

wer_stats_ler: !new:speechbrain.utils.metric_stats.ErrorRateStats
""")
```


```python
refs_ler = ler_hparams["ler_model"].lemmatize(refs)
hyps_ler = ler_hparams["ler_model"].lemmatize(hyps)
```


```python
print(" ".join(refs_ler[0]))
print(" ".join(hyps_ler[0]))
```


```python
ler_hparams["wer_stats_ler"].clear()
ler_hparams["wer_stats_ler"].append(
    ids=list(range(len(refs))),
    predict=hyps_ler,
    target=refs_ler,
)
ler_hparams["wer_stats_ler"].summarize()
```

## Embedding Error Rate (EmbER)

Typical WER calculation, except that we weight the penalty of each word substitution if the words are deemed similar enough. This allows you to reduce the impact of e.g. minor spelling errors that do not alter the meaning much.

Setup for this is slightly more involved but the gist of it is that you need:
- A regular `ErrorRateStats` object which you will `.append()` to,
- The embeddings that you will be using, e.g. using the `FlairEmbeddings` wrapper,
- The EmbER configuration, which will point to the embedding (here binding to `ember_embeddings.embed_word`),
- The `WeightedErrorRateStats` which piggy backs over the base `ErrorRateStats` and plugs into the EmbER similarity function defined just above.


```python
ember_hparams = load_hyperpyyaml("""
wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats

ember_embeddings: !apply:speechbrain.lobes.models.flair.embeddings.FlairEmbeddings.from_hf
    embeddings_class: !name:flair.embeddings.FastTextEmbeddings
    source: facebook/fasttext-fr-vectors
    save_path: ./pretrained_models/

ember_metric: !new:speechbrain.utils.metric_stats.EmbeddingErrorRateSimilarity
    embedding_function: !name:speechbrain.lobes.models.flair.embeddings.FlairEmbeddings.embed_word
        - !ref <ember_embeddings>
    low_similarity_weight: 1.0
    high_similarity_weight: 0.1
    threshold: 0.4

weighted_wer_stats: !new:speechbrain.utils.metric_stats.WeightedErrorRateStats
    base_stats: !ref <wer_stats>
    cost_function: !ref <ember_metric>
    weight_name: ember
""")
```


```python
ember_hparams["wer_stats"].clear()
ember_hparams["wer_stats"].append(
    ids=list(range(len(refs))),
    predict=hyps,
    target=refs,
)
ember_hparams["weighted_wer_stats"].clear()
ember_hparams["weighted_wer_stats"].summarize()
```

## BERTScore

In a nutshell, BERTScore works by comparing the cosine similarity of *all* targets and predicted embeddings, as obtained from a BERT-like LM encoder. This works rather well because the embeddings are trained to embed information from their context.

This is best explained by the code and documentation of the metric itself.


```python
bertscore_hparams = load_hyperpyyaml("""
bertscore_model_name: camembert/camembert-large
bertscore_model_device: cuda

bertscore_stats: !new:speechbrain.utils.bertscore.BERTScoreStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: !ref <bertscore_model_name>
        save_path: pretrained_models/
        device: !ref <bertscore_model_device>
        num_layers: 8
""")
```


```python
bertscore_hparams["bertscore_stats"].clear()
bertscore_hparams["bertscore_stats"].append(
    ids=list(range(len(refs))),
    predict=hyps,
    target=refs,
)
bertscore_hparams["bertscore_stats"].summarize()
```

## Sentence Semantic Distance: SemDist

Estimated using the cosine similarity of a single embedding for every sentence, e.g. obtained by averaging of LM embeddings over all tokens.

Here, lower is better. The score is normalized by x1000 by default for readability.


```python
semdist_hparams = load_hyperpyyaml("""
semdist_model_name: camembert/camembert-large
semdist_model_device: cuda

semdist_stats: !new:speechbrain.utils.semdist.SemDistStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: !ref <semdist_model_name>
        save_path: pretrained_models/
        device: !ref <semdist_model_device>
    method: meanpool
""")
```


```python
semdist_hparams["semdist_stats"].clear()
semdist_hparams["semdist_stats"].append(
    ids=list(range(len(refs))),
    predict=hyps,
    target=refs,
)
semdist_hparams["semdist_stats"].summarize()
```


```python
semdist_hparams["semdist_stats"].scores[:5]
```

## Some comparisons

This was a bit thrown together, if you've run everything without running out of RAM congratulations :)


```python
for i in range(10):
    ref = " ".join(refs[i])
    hyp = " ".join(hyps[i])

    print(f"""\
=== REF: {ref}
=== HYP: {hyp}
WER:                  {wer_hparams['wer_stats'].scores[i]['WER']:.3f}%
CER:                  {cer_hparams['cer_stats'].scores[i]['WER']:.3f}%
dPOSER:               {poser_hparams['wer_stats_dposer'].scores[i]['WER']:.3f}%
uPOSER:               {poser_hparams['wer_stats_uposer'].scores[i]['WER']:.3f}%
EmbER:                {ember_hparams['weighted_wer_stats'].scores[i]['WER']:.3f}%
BERTScore recall:     {bertscore_hparams['bertscore_stats'].scores[i]['recall']:.5f}
BERTScore precision:  {bertscore_hparams['bertscore_stats'].scores[i]['precision']:.5f}
SemDist mean (x1000): {semdist_hparams['semdist_stats'].scores[i]['semdist']:.5f}
""")
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
