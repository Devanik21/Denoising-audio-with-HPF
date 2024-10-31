import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt
from io import BytesIO

# Function for band-pass filtering
def band_pass_filter(audio, sr, lowcut=1, highcut=2000):
    sos = butter(4, [lowcut, highcut], btype='band', fs=sr, output='sos')
    return sosfilt(sos, audio)

# Function to denoise audio (placeholder for ML model)
def denoise_audio(audio):
    return audio * 0.8  # Reduces noise as a placeholder

# Streamlit app layout
st.title("Audio Denoising App")
st.write("Upload a noisy audio file to apply band-pass filtering and denoising.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load the uploaded audio
    noisy_audio, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav", start_time=0)

    # Apply band-pass filter
    filtered_audio = band_pass_filter(noisy_audio, sr)
    
    # Denoise the audio (placeholder for ML model)
    denoised_audio = denoise_audio(filtered_audio)

    # Display original and processed audio side-by-side
    st.subheader("Original Noisy Audio")
    st.audio(uploaded_file, format="audio/wav")

    st.subheader("Denoised Audio")
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
