import streamlit as st
import requests
import io
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image
import soundfile as sf


# Streamlit app title
st.title("Voice-to-Image Generator")

# Function to send audio data to the backend and receive generated image
# Function to send audio data to the backend and receive generated image
def send_to_backend(audio_data, sample_rate):
    backend_url = "http://127.0.0.1:5000/process_audio"  # Replace with your actual backend URL
    sf.write("audio.wav", audio_data, sample_rate)
    files = {"audio": ("audio.wav", open("audio.wav", "rb"))}
    response = requests.post(backend_url, files=files)

    # Receive and display the generated image from the backend
    if response.status_code == 200:
        result = response.json()  # Assuming the backend returns JSON with transcribed text
        transcribed_text = result.get("transcribed_text", "")
        generated_image = Image.open(io.BytesIO(result["image"]))

        st.image(generated_image, caption="Generated Image", use_column_width=True)
        st.write("Transcribed Text:", transcribed_text)

    else:
        st.error("Error communicating with the backend.")


# Streamlit app to record audio using the browser's microphone
def app():
    st.write("Click the 'Record' button and speak into the microphone.")

    record_button = st.button("Record")

    if record_button:
        st.write("Recording...")
        sample_rate = 44100
        duration = 10  # You can adjust the duration as needed

        audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()

        # Function to send audio data to the backend and receive generated image
def send_to_backend(audio_data, sample_rate):
    backend_url = "http://127.0.0.1:5000/process_audio"  # Replace with your actual backend URL
    sf.write("audio.wav", audio_data, sample_rate)
    files = {"audio": ("audio.wav", open("audio.wav", "rb"))}
    response = requests.post(backend_url, files=files)

    # Receive and display the generated image from the backend
    if response.status_code == 200:
        result = response.json()  # Assuming the backend returns JSON with transcribed text
        transcribed_text = result.get("transcribed_text", "")
        generated_image = Image.open(io.BytesIO(result["image"]))

        st.image(generated_image, caption="Generated Image", use_column_width=True)
        st.write("Transcribed Text:", transcribed_text)

    else:
        st.error("Error communicating with the backend.")


# Run the Streamlit app
if __name__ == "__main__":
    app()
