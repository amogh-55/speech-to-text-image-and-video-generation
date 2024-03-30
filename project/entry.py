import os
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from PIL import Image
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import pickle

st.experimental_set_query_params(timeout=600)
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the StableDiffusionPipeline
modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    modelid, revision="fp16", torch_dtype=torch.float32, use_auth_token="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"
)

# Check for available GPU and use it if available
if torch.cuda.is_available():
    # Choose the first available GPU (you can modify this if you want a specific GPU)
    pipe.to(torch.device("cuda:0"))

# Function to pickle the StableDiffusionPipeline model
def pickle_model(pipe):
    model_filename = "stable_diffusion_model.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(pipe, model_file)
    return model_filename

def record_audio():
    # Set the sample rate and duration for recording
    sample_rate = 44100
    duration = 10

    # Record audio from the microphone using sounddevice
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()

    # Save the recorded audio to a temporary WAV file using soundfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        sf.write(temp_wav.name, audio_data, sample_rate)

    return temp_wav.name

def transcribe_audio_from_mic(audio_file_path):
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file_path) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Record the audio
        audio = recognizer.record(source)

        try:
            # Use the Google Web Speech API to transcribe the audio
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Error in requesting results from Google Web Speech API: {e}"

def generate(prompt):
    torch.cuda.empty_cache()
    print(prompt)
    with autocast("cuda" if torch.cuda.is_available() else "cpu"):
        response = pipe(prompt, guidance_scale=8.5)
        # Print the keys in the response to understand its structure
        print("Response keys:", response.keys())
        
        # Replace 'sample' with the correct key based on the response structure
        image = response["images"][0]

    # Save the image
    image_path = "generatedimage.png"
    image.save(image_path)

    # Display the image using PIL
    img = Image.open(image_path)
    img.show()

# Streamlit UI
st.title("Audio Transcription and Image Generation App")

# Record audio button
recording = st.button("Record Audio")

if recording:
    with st.spinner("Recording audio... Please speak now."):
        audio_file_path = record_audio()

    # Transcribe audio
    with st.spinner("Transcribing audio..."):
        transcribed_text = transcribe_audio_from_mic(audio_file_path)

    st.write("Transcribed Text:", transcribed_text)

    # Generate image button
generate_image = st.button("Generate Image from Transcription")

if generate_image:
    with st.spinner("Generating image..."):
        generated_image_path = generate(transcribed_text)
        st.image(generated_image_path, caption="Generated Image", use_column_width=True)

