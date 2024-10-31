import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt, wiener
import pywt
from io import BytesIO

# Advanced band-pass filter with adjustable smoothness
def advanced_band_pass_filter(audio, sr, lowcut=300, highcut=3000):
    sos = butter(6, [lowcut, highcut], btype='band', fs=sr, output='sos')
    return sosfilt(sos, audio)

# Apply spectral gating to remove noise
def spectral_gate(audio, sr, noise_reduction_factor=0.5):
    stft = librosa.stft(audio)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_magnitude = np.median(magnitude, axis=1, keepdims=True) * noise_reduction_factor
    gated_magnitude = np.maximum(magnitude - noise_magnitude, 0)
    return librosa.istft(gated_magnitude * np.exp(1j * phase))

# Apply wavelet denoising
def wavelet_denoise(audio, wavelet='db1', level=2):
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(audio)))  # Universal threshold
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)

# Streamlit app layout
st.title("Advanced Audio Denoising App")
st.write("Upload a noisy audio file to apply advanced noise cancellation techniques.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load the uploaded audio
    noisy_audio, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav", start_time=0)

    # Apply advanced band-pass filter
    filtered_audio = advanced_band_pass_filter(noisy_audio, sr)
    
    # Apply spectral gating
    gated_audio = spectral_gate(filtered_audio, sr)
    
    # Apply wavelet denoising
    denoised_audio = wavelet_denoise(gated_audio)

    # Display original and processed audio side-by-side
    st.subheader("Original Noisy Audio")
    st.audio(uploaded_file, format="audio/wav")

    st.subheader("Denoised Audio with Advanced Noise Cancellation")
    denoised_audio_bytes = BytesIO()
    sf.write(denoised_audio_bytes, denoised_audio, sr, format='WAV')
    st.audio(denoised_audio_bytes, format="audio/wav")

    # Provide download links for original and denoised audio
    st.download_button(
        label="Download Original Audio",
        data=uploaded_file,
        file_name="original_noisy_audio.wav",
        mime="audio/wav"
    )

    st.download_button(
        label="Download Denoised Audio",
        data=denoised_audio_bytes.getvalue(),
        file_name="denoised_audio.wav",
        mime="audio/wav"
    )

    st.write("Original and denoised audio files are available for download.")
