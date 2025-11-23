import os
import json
import logging
import tempfile
from pathlib import Path

import openai
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip

# YouTube upload imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON")  # must be JSON string of credentials

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Script generation will fail without it.")
else:
    openai.api_key = OPENAI_API_KEY

def generate_script(prompt=None, language="hi"):
    logger.info("Generating script (OpenAI)...")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment.")
    if not prompt:
        prompt = "Write a short motivational YouTube script in simple Hindi (approx 100-160 words). Include a strong opening line and a call to action."

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes short YouTube video scripts."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    logger.info("Script generated.")
    return text

def generate_image_from_text(text, out_path="visual.png", size=(1280,720)):
    logger.info("Generating simple visual (PIL)...")
    img = Image.new("RGB", size, color=(24,24,24))
    draw = ImageDraw.Draw(img)

    # Choose a default font. GitHub runner may not have fancy fonts; use default.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    # Wrap and draw text
    margin = 40
    max_width = size[0] - margin*2
    lines = []
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if draw.textsize(test, font=font)[0] <= max_width:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)

    y = 80
    for ln in lines[:20]:  # limit lines to avoid overflow
        draw.text((margin, y), ln, font=font, fill=(255,255,255))
        y += font.getsize(ln)[1] + 8

    img.save(out_path)
    logger.info("Saved visual to %s", out_path)
    return out_path

def generate_voice(text, out_path="voice.mp3", lang="hi"):
    logger.info("Generating voice (gTTS)...")
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    logger.info("Saved voice to %s", out_path)
    return out_path

def make_video(image_path, audio_path, out_path="output.mp4"):
    logger.info("Making video (moviepy)...")
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    clip = ImageClip(image_path).set_duration(duration)
    clip = clip.set_audio(audio)
    clip.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    logger.info("Video created: %s", out_path)
    return out_path

def upload_to_youtube(video_path, title, description):
    if not YOUTUBE_TOKEN_JSON:
        logger.info("YOUTUBE_TOKEN_JSON not provided; skipping upload.")
        return None

    logger.info("Uploading to YouTube...")
    creds_info = json.loads(YOUTUBE_TOKEN_JSON)
    creds = Credentials.from_authorized_user_info(creds_info)
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {"title": title, "description": description, "categoryId": "22"},
        "status": {"privacyStatus": "private"},
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    try:
        response = request.execute()
        logger.info("Upload complete. Video id: %s", response.get("id"))
    except Exception as e:
        logger.error("YouTube upload failed: %s", e)
        raise
    return response

def run():
    try:
        script = generate_script()
    except Exception as e:
        logger.error("Script generation failed: %s", e)
        return

    # Create temporary working directory
    tmp = tempfile.mkdtemp(prefix="ytbot_")
    img_path = os.path.join(tmp, "visual.png")
    audio_path = os.path.join(tmp, "voice.mp3")
    video_path = os.path.join(tmp, "output.mp4")

    generate_image_from_text(script, out_path=img_path)
    generate_voice(script, out_path=audio_path, lang="hi")
    make_video(img_path, audio_path, out_path=video_path)

    title = "Auto Video - " + (script.splitlines()[0][:60] if script else "AI Video")
    desc = script + "\n\nCreated automatically."

    # Try upload
    try:
        upload_to_youtube(video_path, title, desc)
    except Exception:
        logger.info("Upload skipped or failed; video is available at: %s", video_path)

if __name__ == "__main__":
    run()
