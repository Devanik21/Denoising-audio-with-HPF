# Denoising Audio With HPF

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Denoising-audio-with-HPF?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Denoising-audio-with-HPF?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Signal-level audio restoration using classical DSP — High-Pass Filter design, spectral denoising, and quality measurement.

---

**Topics:** `audio-denoising` · `audio-processing` · `butterworth-filter` · `digital-signal-processing` · `dsp` · `high-pass-filter` · `machine-learning` · `neural-networks` · `python` · `signal-processing`

## Overview

This project implements a comprehensive audio denoising pipeline using classical Digital Signal Processing techniques: High-Pass Filtering (HPF), spectral subtraction, and Wiener filtering. It is designed to restore speech quality in recordings contaminated by low-frequency hum (electrical noise at 50/60 Hz and harmonics), broadband Gaussian noise, and impulsive noise artefacts.

The HPF implementation covers four classical IIR filter families — Butterworth (maximally flat passband), Chebyshev Type I (equiripple passband), Chebyshev Type II (equiripple stopband), and Elliptic (equiripple in both bands) — all designed with SciPy's `signal.iirdesign` and applied via second-order sections (SOS) for numerical stability. The cutoff frequency, passband ripple, and stopband attenuation are fully configurable.

Beyond HPF, the pipeline includes a Wiener filter implementation in the frequency domain and a spectral subtraction method that estimates the noise spectrum from silent frames and subtracts it from the noisy spectrum. Before-and-after waveform plots, spectrogram comparisons, and SNR measurements quantify the denoising effect for each method.

---

## Motivation

Audio denoising is a fundamental challenge in communications, hearing aids, podcast production, speech recognition preprocessing, and archival restoration. Understanding the mathematics of IIR filter design and spectral estimation is essential for any signal processing engineer. This project was built to make those concepts concrete and measurable.

---

## Architecture

```
Noisy WAV input
        │
  Frame segmentation (windowed, 50% overlap)
        │
  ┌──────────────────────────────────────┐
  │  Method 1: HPF (Butterworth/Cheby)   │
  │  Method 2: Wiener Filter (freq domain)│
  │  Method 3: Spectral Subtraction       │
  └──────────────────────────────────────┘
        │
  Overlap-add reconstruction
        │
  SNR measurement: 10 log₁₀(P_signal / P_noise)
        │
  Output WAV + Spectrogram comparison
```

---

## Features

### IIR Filter Design Suite
Four classical HPF families implemented with SciPy: Butterworth, Chebyshev I/II, and Elliptic — with configurable cutoff frequency, filter order, ripple, and attenuation parameters.

### Second-Order Sections (SOS) Filtering
All filters applied via SOS decomposition (sosfilt) rather than direct-form coefficients, preventing numerical instability for high-order filters.

### Wiener Filter (Frequency Domain)
Optimal Wiener filter estimated from a noise-only reference segment, applied in the STFT domain to minimise mean-square error between clean and denoised speech.

### Spectral Subtraction
Power spectral density of noise estimated from silent frames (VAD-detected) and subtracted from the noisy spectrum, with a flooring constant to prevent musical noise artefacts.

### Before/After Spectrogram Comparison
Side-by-side mel spectrogram and wideband spectrogram plots for noisy and denoised audio, highlighting suppressed noise components.

### SNR Measurement
Quantitative SNR (dB) computed before and after denoising using the clean reference signal; also estimates segmental SNR for speech-specific evaluation.

### Batch Processing
Command-line batch mode processes an entire directory of audio files with the selected method and saves denoised outputs with naming convention.

### Frequency Response Visualisation
Bode magnitude and phase plots for designed filters, showing passband flatness and stopband attenuation with annotated −3dB cutoff frequency.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **SciPy Signal** | Filter design and application | iirdesign, sosfilt, spectrogram, welch |
| **LibROSA** | Audio feature extraction | Mel spectrogram, STFT, load/save |
| **NumPy** | Array operations | FFT, SNR computation, framing |
| **Matplotlib** | Visualisation | Waveform, spectrogram, Bode plots |
| **Soundfile** | Audio I/O | WAV/FLAC read/write with correct sample rates |
| **Streamlit (optional)** | Interactive UI | Filter parameter sliders and spectrogram display |

> **Key packages detected in this repo:** `streamlit` · `librosa` · `soundfile` · `scipy` · `numpy` · `pywavelets`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/Denoising-audio-with-HPF.git
cd Denoising-audio-with-HPF
python -m venv venv && source venv/bin/activate
pip install scipy librosa numpy matplotlib soundfile streamlit

# Denoise a single file
python denoise.py --input noisy_speech.wav --method hpf --cutoff 100

# Interactive UI
streamlit run app.py
```

---

## Usage

```bash
# HPF denoising
python denoise.py --input audio.wav --method hpf --filter butterworth --order 4 --cutoff 80

# Wiener filter
python denoise.py --input audio.wav --method wiener --noise_frames 20

# Spectral subtraction
python denoise.py --input audio.wav --method spectral_sub --alpha 2.0 --beta 0.01

# Batch processing
python batch_denoise.py --input_dir ./noisy/ --output_dir ./clean/ --method hpf
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `--method` | `hpf` | Denoising method: hpf, wiener, spectral_sub |
| `--filter` | `butterworth` | HPF type: butterworth, cheby1, cheby2, ellip |
| `--order` | `4` | IIR filter order |
| `--cutoff` | `80` | HPF cutoff frequency in Hz |
| `--noise_frames` | `20` | Number of silent frames for noise estimation |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
Denoising-audio-with-HPF/
├── README.md
├── requirements.txt
├── app.py
├── .devcontainer/devcontainer.json
└── ...
```

---

## Roadmap

- [ ] Deep learning denoising: RNNoise and SEGAN model integration for comparison
- [ ] Real-time audio stream processing via PyAudio with sub-frame latency
- [ ] Multi-band noise estimation for non-stationary noise environments
- [ ] Evaluation on standard speech enhancement benchmarks (NOIZEUS, VoiceBank-DEMAND)
- [ ] PESQ and STOI objective speech quality metrics for rigorous comparison

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

All filter designs assume a sample rate of 16kHz by default. Adjust the sample_rate parameter for other audio formats. The Wiener filter requires a clean noise reference segment for optimal performance.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
