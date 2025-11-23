import os
import requests
from gtts import gTTS
from moviepy.editor import *
import random
import string

# =========================
# FREE SYSTEM — NO API KEYS
# =========================

# Free HuggingFace text model
HF_TEXT_MODEL = "tiiuae/falcon-7b-instruct"

# Free HuggingFace image model
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


# =========================
# Generate YouTube Script
# =========================
def generate_script():
    prompt = """
    Write a 45-second YouTube short script in Hindi on an interesting fact.
    Style: engaging, fast-paced, simple language.
    """
    
    response = requests.post(
        "https://api-inference.huggingface.co/models/" + HF_TEXT_MODEL,
        headers={"Authorization": "Bearer " + os.getenv("HF_TOKEN", "")},
        json={"inputs": prompt}
    )

    text = response.json()[0]["generated_text"]
    return text


# =========================
# Generate AI Image
# =========================
def generate_image(prompt, filename="image.png"):
    response = requests.post(
        "https://api-inference.huggingface.co/models/" + HF_IMAGE_MODEL,
        headers={"Authorization": "Bearer " + os.getenv("HF_TOKEN", "")},
        json={"inputs": prompt},
    )

    with open(filename, "wb") as f:
        f.write(response.content)

    return filename


# =========================
# Convert Script → Voice
# =========================
def generate_voice(script, filename="voice.mp3"):
    tts = gTTS(script, lang="hi")
    tts.save(filename)
    return filename


# =========================
# Create Video
# =========================
def create_video(img_path, audio_path, output="final.mp4"):

    image_clip = ImageClip(img_path).set_duration(AudioFileClip(audio_path).duration)
    audio_clip = AudioFileClip(audio_path)

    final = image_clip.set_audio(audio_clip)
    final.write_videofile(output, fps=24)

    return output


# =========================
# MAIN PIPELINE
# =========================
def run():
    print("✔ Generating Hindi script...")
    script = generate_script()

    print("✔ Creating image...")
    img = generate_image("beautiful cinematic scene, ultra detailed", "img.png")

    print("✔ Generating voice...")
    audio = generate_voice(script, "voice.mp3")

    print("✔ Rendering final video...")
    video = create_video(img, audio, "final_output.mp4")

    print("🎉 DONE — Video Ready:", video)


if __name__ == "__main__":
    run()
