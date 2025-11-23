import os
import json
import requests
from moviepy.editor import *
from moviepy.video.tools.drawing import color_gradient
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText?key={GEMINI_KEY}"
    body = { "prompt": { "text": prompt } }
    r = requests.post(url, json=body)
    data = r.json()
    try:
        return data["candidates"][0]["output"]
    except:
        return "Error script"

def generate_script():
    prompt = """
    Create a viral 45 sec Hindi YouTube short fact script.
    Make it engaging and simple.
    """
    return gemini(prompt)


# ---------------- TEXT TO IMAGE (PIL instead of ImageMagick) ---------------- #

def text_to_image(text, out_path):
    img = Image.new("RGB", (1080,1920), color=(0,0,0))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("DejaVuSans.ttf", 60)

    lines = []
    words = text.split(" ")
    line = ""
    for w in words:
        if draw.textlength(line + " " + w, font=font) < 900:
            line += " " + w
        else:
            lines.append(line)
            line = w
    lines.append(line)

    y = 200
    for line in lines:
        draw.text((100,y), line, font=font, fill=(255,255,255))
        y += 100

    img.save(out_path)


# ---------------- VIDEO CREATION ---------------- #

def create_video(script):
    os.makedirs("output", exist_ok=True)

    voice_path = "output/voice.mp3"
    gTTS(script, lang="hi").save(voice_path)

    # Generate text image
    img_path = "output/text.png"
    text_to_image(script, img_path)

    # Create video background
    img_clip = ImageClip(img_path).set_duration(45)
    audio = AudioFileClip(voice_path)

    final = img_clip.set_audio(audio)

    out = "output/final.mp4"
    final.write_videofile(out, fps=24, codec="libx264", audio_codec="aac")
    return out


def run():
    print("Generating script...")
    script = generate_script()

    print("Creating video...")
    video = create_video(script)

    print("DONE:", video)

if __name__ == "__main__":
    run()
