import os
import json
import google.generativeai as genai
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")  # Free text model


def generate_script():
    print("Generating script using Gemini...")
    prompt = """
    Write a short 40–60 second motivational YouTube video script in Hindi.
    Must include:
    - powerful opening hook  
    - emotional lines  
    - climax / message  
    - call to action  
    """
    response = model.generate_content(prompt)
    return response.text


def generate_image(text):
    print("Generating image using Gemini Vision-free API style prompt...")
    image_name = "visual.png"

    # Simple PIL-based poster design
    img = Image.new("RGB", (1280, 720), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)

    lines = []
    words = text.split()
    line = ""

    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=font) < 1100:
            line = test
        else:
            lines.append(line)
            line = w
    lines.append(line)

    y = 200
    for ln in lines[:6]:
        draw.text((60, y), ln, font=font, fill=(255, 255, 255))
        y += 60

    img.save(image_name)
    return image_name


def generate_voice(script):
    print("Generating voice using gTTS...")
    voice_path = "voice.mp3"
    tts = gTTS(text=script, lang="hi")
    tts.save(voice_path)
    return voice_path


def make_video(image_path, audio_path):
    print("Creating video using moviepy...")
    audio = AudioFileClip(audio_path)
    clip = ImageClip(image_path).set_duration(audio.duration)
    clip = clip.set_audio(audio)
    clip.write_videofile("output.mp4", fps=24)
    return "output.mp4"


def upload_to_youtube(video_path, title, desc):
    print("Uploading to YouTube...")

    creds = Credentials.from_authorized_user_info(json.loads(YOUTUBE_TOKEN_JSON))
    youtube = build("youtube", "v3", credentials=creds)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title, "description": desc},
            "status": {"privacyStatus": "public"}
        },
        media_body=MediaFileUpload(video_path, mimetype="video/mp4")
    )

    response = request.execute()
    print("Uploaded!", response)


def run():
    script = generate_script()
    image = generate_image(script)
    voice = generate_voice(script)
    video = make_video(image, voice)
    upload_to_youtube(video, "AI Gemini Auto Video", script)


if __name__ == "__main__":
    run()
