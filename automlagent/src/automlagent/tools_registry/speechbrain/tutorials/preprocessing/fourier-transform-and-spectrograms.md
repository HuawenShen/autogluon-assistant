Summary: This tutorial provides implementation details for audio signal processing using Fourier transforms in Python/PyTorch. It covers three main techniques: basic Fourier Transform (FFT), Short-Time Fourier Transform (STFT), and spectrogram computation. The tutorial helps with tasks like audio signal analysis, time-frequency representation, and signal reconstruction. Key features include efficient FFT implementation (O(NlogN)), windowing techniques, STFT parameter optimization, spectrogram visualization, and handling critical parameters like sample rate and window length. It emphasizes best practices for window selection, overlap settings, and visualization techniques, particularly useful for speech processing applications.

<!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->
<!-- The contents are initialized from tutorials/notebook-header.md -->

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/fourier-transform-and-spectrograms.ipynb)
to execute or view/download this notebook on
[GitHub](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/preprocessing/fourier-transform-and-spectrograms.ipynb)

# Fourier Transforms and Spectrograms

In speech and audio processing, the signal in the time-domain is often transformed into another domain. Ok, but why do we need to transform an audio signal?

Some speech characteristics/patterns of the signal (e.g, *pitch*, *formats*) might not be very evident when looking at the audio in the time-domain. With properly designed transformations, it might be easier to extract the needed information from the signal itself.

The most popular transformation is the **Fourier Transform**, which turns the time-domain signal into an equivalent representation in the **frequency domain**. In the following sections, we will describe the Fourier transforms along with other related transformations such as **Short-Term Fourier Transform** (STFT) and **spectrograms**.

## 1. Fourier Transform
The Fourier transform of a time-discrete sequences $f[n]={f[0],f[1],..f[N-1]}$ is called Discrete Fourier Transform (DFT) and it is defined in this way:

$F_{k} = \sum_{n=0}^{N-1} f_{n} e^{-j\frac{2\pi}{N}kn}$

The inverse transformation, called Inverse Discrete Fourier Transform (IDFT), maps the frequnecy-domain signal $F_k$ into a time-domain one $f_n$:

$f_{n} = \sum_{k=0}^{N-1} F_{k} e^{j\frac{2\pi}{N}kn}$

The two representations are equivalent and we are not losing information when applying them. It is just a different way to represent the same signal.


#### What is the intuition?
The idea behind the Fourier transform is to represent the signal as a **weighted sum of complex sinusoids with increasing frequency**.
The complex exponential $e^{j\frac{2\pi}{N}kn}$, for instance, dermines the frequnecy of this "complex sinoudoid":

$e^{j\frac{2\pi}{N}kn} = cos(\frac{2\pi}{N}kn) +j sin(\frac{2\pi}{N}kn)$.

The term $F_{k}$, instead, is another **complex number** that determines the amplitude and shift (phase) of the frequency components.
It can be shown that with N complex sinusoids with proper **amplitude** and **phase**, we can model any signal. In other words, the complex sinusoids are the basic bricks that compose your signal. If you properly combine many of them like in a LEGO building, you can create all the signals you want (both periodic and non-periodic).

The transformation has $O(N^2)$ complexity because for each element k of the frequency representation $F_k$ we have to loop over all the N elements of the sequence. This makes it impossible to compute DFT and IDFT of long sequences.

Fortunately,  there are algorithms called **Fast-Fourier Transform (FFT)** that can compute it with $O(Nlog(N))$. The FFT splits the input sequences into small chunks and combines their DTFs.

This concept of "complex sinusoids" might be quite difficult to digest. Nevertheless, on-line you can find excellent material full of cool graphical animations to help you with that (see the tutorials in the reference). For now, let's just consider the Fourier transform as a **linear transformation** that maps real-valued sequences into complex-valued ones.

Before computing some DTFTs, let's download some speech signal and install speechbrain:


```python
%%capture
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


```python
import torch
import matplotlib.pyplot as plt
from speechbrain.dataio.dataio import read_audio

signal = read_audio('/content/spk1_snt1.wav')
print(signal.shape)

# fft computation
fft = torch.fft.fft(signal.squeeze(), dim=0)
print(fft)
print(fft.shape)
```

As you can see, the input signal is real (and thus the imaginary part is filled with zeros). The DFT is a tensor containing both the real and the imaginary parts of the transformation.

Let's now compute the magnitude and phase of the DFT and plot them:


```python
# Real and Imaginary parts
real_fft = fft.real
img_fft = fft.imag

mag = torch.sqrt(torch.pow(real_fft,2) + torch.pow(img_fft,2))
phase = torch.arctan(img_fft/real_fft)

plt.subplot(211)
x_axis = torch.linspace(0, 16000, mag.shape[0])
plt.plot(x_axis, mag)

plt.subplot(212)
plt.plot(x_axis, phase)
plt.xlabel('Freq [Hz]')

```

There are few interesting things to notice from the plots:


*   The plot of the magnitude is symmetric. The last element of the x-axis corresponds to the sampling frequency $f_s$, which in this case is 16kHz. Due to this symmetry, it is only necessary to plot the magnitude from 0 to $fs/2$. This frequency is called Nyquist frequency.
*   The plot of the phase is very noisy. This is expected too. The phase is notoriously not easy to interpret and estimate.

Let's not plot the magnitude from 0 to the Nyquist frequency:


```python
half_point = mag[0:].shape[0]//2
x_axis = torch.linspace(0, 8000, half_point)
plt.plot(x_axis, mag[0:half_point])
plt.xlabel('Frequency')
```

We can see that most of the energy of a speech signal is concentrated in the lower part of the spectrum. Many important phonemes like vowels, in fact, have most of their energy in this part of the spectrum.

Moreover, we can notice some peaks in the magnitude spectrum. Let's zoom in to see them more clearly:



```python
plt.plot(mag[0:4000])
plt.xlabel('Frequency')
```

The peaks corresponds to pitch (i.e, the frequency at which our vocal cords are vibrating) and formats (which corresponds to the resonant frequency of our vocal tract).

Let's now try to go back to the time domain:


```python
signal_rec = torch.fft.ifft(fft, dim=0)
signal_rec = signal_rec # real part
signal_orig = signal

# Plots
plt.subplot(211)
plt.plot(signal_orig)

plt.subplot(212)
plt.plot(signal_rec)
plt.xlabel('Time')

print(signal_orig[0:10])
print(signal_rec[0:10])
```

As you can see from the plot, the signal can be recunstructed in the time domain. Due to some numerical round-off errros, the two signals are very similar but not identical (see the print of the first 10 samples).

## 2. Short-Term Fourier Transform (STFT)
Speech is a "dynamic" signal that evolves over time. It could thus make sense to introduce a mixed time-frequency representation that can show how the frequency components of speech are evolving over time. Such a representation is called Short-Term Fourier Transform.

The SFTF is computed in this way:

1. Split the time signal into multiple chunks using overlapped sliding windows (e.g, hamming, hanning, blackman).
2. For each small chunk compute the DFT
3. Combine all the DFT into a single representation

Let's now compute an STFT of a speech signal:


```python
from speechbrain.processing.features import STFT

signal = read_audio('/content/spk1_snt1.wav').unsqueeze(0) # [batch, time]

compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400) # 25 ms, 10 ms
signal_STFT = compute_STFT(signal)

print(signal.shape)
print(signal_STFT.shape)
```

*   The first dimension of the STFT representation is the batch axis (SpeechBrain expects it because it is designed to process in parallel multiple signals).
* The third is the frequency resolution. It corresponds to half of the fft points ($n_{fft}$) because, as we have seen before, the fft is symmetric.
* The last dimension gathers the real and the imaginary parts of the STFT representation.


Similar to the Fourier transform, the STFT has an inverse transformation called **Inverse Short-Term Fourier Transform (ISTFT)**. With properly-designed windows,  we can have a perfect reconstruction of the original signal:


```python
from speechbrain.processing.features import ISTFT

compute_ISTFT = ISTFT(sample_rate=16000, win_length=25, hop_length=10)
signal_rec = compute_ISTFT(signal_STFT)
signal_rec = signal_rec.squeeze() # remove batch axis for plotting

# Plots
plt.subplot(211)
plt.plot(signal_orig)

plt.subplot(212)
plt.plot(signal_rec)
plt.xlabel('Time')
```

## 3. Spectrogram
As we have seen before, the magnitude of the Fourier transform is more informative than the phase. We can thus take the magnitude of the STFT representation and obtain the so-called spectrogram. The spectrogram is one of the most popular speech representations.

Let's see how a spectrogram looks like:


```python
spectrogram = signal_STFT.pow(2).sum(-1) # power spectrogram
spectrogram = spectrogram.squeeze(0).transpose(0,1)

spectrogram_log = torch.log(spectrogram) # for graphical convenience

plt.imshow(spectrogram_log.squeeze(0), cmap='hot', interpolation='nearest', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

```

The spectrogram is a 2D representation that can be plotted as an image (yellow areas correspond to time-frequency points with high magnitude).
From the spectrogram, you can see how the frequency components are evolving over time. For instance, you can clearly distinguish vowels (whose frequency pattern is characterized by multiple lines corresponding to pitch and formants)  and fricatives (characterized by the presence of continuous high-frequency components). Normally, we plot the power spectrogram that corresponds to the squared magnitude of the STFT.

The time and frequency resolution of the spectrogram depends on the length of the window used for computing the STFT.

For instance, if we increase the length of the window, we can have a higher resolution in frequency (but a lower resolution in time):



```python
signal = read_audio('/content/spk1_snt1.wav').unsqueeze(0) # [batch, time]

compute_STFT = STFT(sample_rate=16000, win_length=50, hop_length=10, n_fft=800)
signal_STFT = compute_STFT(signal)

spectrogram = signal_STFT.pow(2).sum(-1)
spectrogram = spectrogram.squeeze(0).transpose(0,1)
spectrogram = torch.log(spectrogram)

plt.imshow(spectrogram.squeeze(0), cmap='hot', interpolation='nearest', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

```

Vice-versa, we can have a larger time resolution at the price of a reduced frequency resolution:


```python
signal = read_audio('/content/spk1_snt1.wav').unsqueeze(0) # [batch, time]

compute_STFT = STFT(sample_rate=16000, win_length=5, hop_length=5, n_fft=800)
signal_STFT = compute_STFT(signal)

spectrogram = signal_STFT.pow(2).sum(-1)
spectrogram = spectrogram.squeeze(0).transpose(0,1)
spectrogram = torch.log(spectrogram)

plt.imshow(spectrogram.squeeze(0), cmap='hot', interpolation='nearest', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
```

Despite being very informative, the spectrogram is not invertible. When computing it, in fact, we are only using the magnitude of the STFT and not the phase.

The spectrogram is the starting point for computing some popular speech features, such ad FilterBanks (FBANKs) and Mel-Frequency Cepstral Coefficients (MFCCs) that are the object of [another tutorial]().

## References

[1] L. R. Rabiner, Ronald W. Schafer,  “Digital Processing of Speech Signals”, Prentice-Hall, 1978

[2] S. K. Mitra Digital Signal Processing: A Computer-Based Approach [slides](http://doctord.webhop.net/courses/bei/ece410/mitra_2e/toc.htm)

[3] <https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/>

[4] <https://sites.northwestern.edu/elannesscohn/2019/07/30/developing-an-intuition-for-fourier-transforms/>




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
