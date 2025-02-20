Summary: This tutorial demonstrates the implementation of environmental corruption techniques in speech processing using SpeechBrain's time_domain module. It covers two main implementations: AddNoise and AddReverb classes for simulating real-world acoustic conditions. The tutorial provides code examples for adding noise with configurable SNR ranges and applying reverberation through impulse responses, both using CSV-based audio file management. Key functionalities include batch processing support, customizable corruption intensities, and mathematical modeling of signal corruption (y[n] = x[n] * h[n] + n[n]). This knowledge is particularly useful for tasks involving speech augmentation, acoustic simulation, and robust speech processing system development.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/environmental-corruption.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/preprocessing/environmental-corruption.ipynb)

# Environmental Corruption

In realistic speech processing scenarios, the signals captured by microphones are often corrupted by unwanted elements such as **noise** and **reverberation**. This challenge is particularly pronounced in **distant-talking** (far-field) situations, where the speaker and the reference microphone are positioned at a considerable distance. Examples of such scenarios include signals recorded by popular devices like Google Home, Amazon Echo, Kinect, and similar devices.

A common strategy in neural speech processing involves starting with clean speech recordings and artificially introducing noise and reverberation to simulate real-world conditions. This process is known as **environmental corruption** or *speech contamination*.

Starting with clean signals allows for the controlled introduction of various types of noise and reverberation, making environmental corruption a potent regularization technique. This regularization helps neural networks generalize better when exposed to real-world, noisy conditions during testing.

The environmental corruption process transforms a clean signal $x[n]$ into a noisy and reverberant signal using the equation:

$y[n] = x[n] * h[n] + n[n]$

where $n[n]$ represents a noise sequence, and $h[n]$ is an impulse response that introduces the reverberation effect.

In the following sections, we will delve into the details of how this transformation is carried out. Before that, let's download some signals that will be essential for the rest of the tutorial.





```python
%%capture
!wget https://www.dropbox.com/s/vwv8xdr7l3b2tta/noise_sig.csv
!wget https://www.dropbox.com/s/aleer424jumcs08/noise2.wav
!wget https://www.dropbox.com/s/eoxxi2ezr8owk8a/noise3.wav
!wget https://www.dropbox.com/s/pjnub2s5hql2vxs/rir1.wav
!wget https://www.dropbox.com/s/nyno6bqbmiy2rv8/rirs.csv
!wget https://www.dropbox.com/s/u8qyvuyie2op286/spk1_snt1.wav
```


```python
%%capture
# Installing SpeechBrain
BRANCH = 'develop'
!git clone https://github.com/speechbrain/speechbrain.git -b $BRANCH
%cd /content/speechbrain/
!python -m pip install .
```

A clean speech signal looks like this:


```python
import matplotlib.pyplot as plt
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio

clean = read_audio('/content/spk1_snt1.wav').squeeze()

# Plots
plt.subplot(211)
plt.plot(clean)
plt.xlabel('Time')

plt.subplot(212)
plt.specgram(clean,Fs=16000)
plt.xlabel('Time')
plt.ylabel('Frequency')

Audio(clean, rate=16000)
```

## 1. Additive Noise

In SpeechBrain, we designed a class able to contaminate a speech signal with noise (`speechbrain.augment.time_domanin.AddNoise`). This class takes in input a csv file that itemizes a list of noise signals:


```
ID, duration, wav, wav_format, wav_opts
noise2, 5.0, noise2.wav, wav,
noise3, 1.0, noise3.wav, wav,
```
When called, `AddNoise` samples from this noise collection and adds the selected noise into the clean signal with a random **Signal-to-Nose Ratio** (SNR).




```python
import torch
from speechbrain.augment.time_domain import AddNoise

noisifier = AddNoise('tests/samples/annotation/noise.csv', replacements={'noise_folder': 'tests/samples/noise'})
noisy = noisifier(clean.unsqueeze(0), torch.ones(1))

# Plots
plt.subplot(211)
plt.plot(noisy.squeeze())
plt.xlabel('Time')

plt.subplot(212)
plt.specgram(noisy.squeeze(),Fs=16000)
plt.xlabel('Time')
plt.ylabel('Frequency')

Audio(noisy.squeeze(0), rate=16000)

```

The amount of noise can be tuned with the **snr_low** and **snr_high** parameters that define the sampling range for the SNR. The length vector is needed because we can process in parallel batches of signals with different lengths. The length vector contains relative lengths for each sentence composing the batch (e.g, for two examples we can have lenght=[0.8 1.0] where 1.0 is the length of the longest sentence in the batch).


## 2. Reverberation
When speaking into a room, our speech signal is **reflected multi-times** by the walls, floor, ceiling, and by the objects within the acoustic environment. Consequently, the final signal recorded by a distant microphone will contain multiple **delayed replicas** of the original signal. All these replicas interfere with each other and significantly affect the intelligibility of the speech signal.

Such a **multi-path propagation** is called reverberation. Within a given room enclosure, the reverberation between a source and a receiver is modeled by an **impulse response**:



```python
rir = read_audio('/content/rir1.wav')

# Impulse response
plt.subplot(211)
plt.plot(rir[0:8000])
plt.xlabel('Time')
plt.ylabel('h(t)')

# Zoom on early reflections
plt.subplot(212)
plt.plot(rir[2150:2500])
plt.xlabel('Time')
plt.ylabel('h(t)')
```

The impulse response is a complete description of the changes that the sounds undergo when traveling from a source to a receiver. In particular, each peak in the impulse response corresponds to a replica reaching the receiver. The first peak corresponds to the **direct path**. Then, we can see the **first-order reflections** on walls, ceiling, floor (see the second picture).

Globally, the impulse response follows an exponential decay. This decay is faster in a dry room characterized by low reverberation-time and it is slower in a large and empty environment.

The reverberation is added by performing a **convolution** between a clean signal and an impulse response. In SpeechBrain, this operation is performed by `speechbrain.processing.speech_augmentation.AddReverb`.

When called, `AddRev` samples an impulse response from a given csv file:

```
ID, duration, wav, wav_format, wav_opts
rir1, 1.0, rir1.wav, wav,
....
```


```python
from speechbrain.augment.time_domain import AddReverb

reverb = AddReverb('tests/samples/annotation/RIRs.csv', replacements={'rir_folder': 'tests/samples/RIRs'})
reverbed = reverb(clean)

# Plots
plt.subplot(211)
plt.plot(reverbed.squeeze())
plt.xlabel('Time')

plt.subplot(212)
plt.specgram(reverbed.squeeze(),Fs=16000)
plt.xlabel('Time')
plt.ylabel('Frequency')

Audio(reverbed.squeeze(0), rate=16000)

```

Reverberation is a convolutive noise that "smooths" the signal in the time (see the long tails that appear in regions that were silent in the clean signal) and frequency domain.

The amount of reverberation is controlled by the parameter **rir_scale_factor**. If rir_scale_factor < 1, the impulse response is compressed (less reverb), while if rir_scale_factor > 1 the impulse response is dilated (more reverb). Feel free to play with it in the previous example!

## References
[1] M. Ravanelli, P. Svaizer, M. Omologo, "Realistic Multi-Microphone Data Simulation for Distant Speech Recognition",  in Proceedings of Interspeech 2016 [ArXiv](https://arxiv.org/abs/1711.09470)

[2] M. Ravanelli, M. Omologo, "Contaminated speech training methods for robust DNN-HMM distant speech recognition", in Proceedings of  INTERSPEECH 2015. [ArXiv](https://arxiv.org/abs/1710.03538)

[3] M. Ravanelli, M. Omologo, "On the selection of the impulse responses for distant-speech recognition based on contaminated speech training", in Proceedings of  INTERSPEECH 2014. [ArXiv](https://isca-speech.org/archive/archive_papers/interspeech_2014/i14_1028.pdf)

[4] M. Ravanelli, A. Sosi, P. Svaizer, M.Omologo, "Impulse response estimation for robust speech recognition in a reverberant environment",   in Proceeding of the European Signal Processing Conference, EUSIPCO 2012. [ArXiv](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569588145.pdf)



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
