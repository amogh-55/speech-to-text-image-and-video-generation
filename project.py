from flask import Flask, request, send_file
from pydub import AudioSegment
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from io import BytesIO

app = Flask(__name__)

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

# ...

@app.route("/process_audio", methods=["POST"])
def process_audio():
    try:
        # Get the uploaded audio file
        audio_file = request.files["audio"]
        audio_data = audio_file.read()

        # Convert audio data to AudioSegment
        audio_segment = AudioSegment.from_wav(BytesIO(audio_data))

        # Perform audio processing (you can replace this with your own logic)
        # For demonstration purposes, the audio is simply converted to uppercase
        transcribed_text = audio_segment.export(format="wav").read().upper()

        # Generate image based on transcribed text
        generated_image = generate_image(transcribed_text)

        # Convert the generated image to bytes
        img_bytes = BytesIO()
        generated_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return {"transcribed_text": transcribed_text, "image": img_bytes.read().decode("latin1")}

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return "Error processing audio", 500

# ...

def generate_image(prompt):
    with autocast("cuda" if torch.cuda.is_available() else "cpu"):
        response = pipe(prompt, guidance_scale=8.5)
        # Replace 'sample' with the correct key based on the response structure
        image = response["images"][0]

    return image

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
