import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from PIL import Image
import sounddevice as sd
import soundfile as sf  # Import soundfile for writing audio

import numpy as np
import tempfile
import pickle

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the StableDiffusionPipeline
modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    modelid, revision="fp16", torch_dtype=torch.float32, use_auth_token="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"
)

# Check for available GPU and use it if available
if torch.cuda.is_available():
    # Choose the first available GPU (you can modify this if you want a specific GPU)
    pipe.to(torch.device("cuda:0"))
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, falling back to CPU")



def record_audio():
    # Set the sample rate and duration for recording
    sample_rate = 44100
    duration = 10  # You can adjust the duration as needed

    # Record audio from the microphone using sounddevice
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()

    # Save the recorded audio to a temporary WAV file using soundfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        sf.write(temp_wav.name, audio_data, sample_rate)

    return temp_wav.name

def transcribe_audio_from_mic():
    recognizer = sr.Recognizer()

    # Record audio from the microphone
    audio_file_path = record_audio()

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

# Example usage
transcribed_text = transcribe_audio_from_mic()

# Print the transcribed text
print("Transcribed Text:", transcribed_text)

# Print the transcribed text
print("Transcribed Text:", transcribed_text)

# Use the transcribed text as a prompt to generate an image



def generate(prompt):
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
#audio_file_path = "C:/Users/vaval/Desktop/StableDiffusionApp-main/StableDiffusionApp-main/audio.mp3";  # Replace with the actual path to your audio file
#transcribed_text = transcribe_audio(audio_file_path)


# Use the transcribed text as a prompt to generate an image
generate(transcribed_text)

