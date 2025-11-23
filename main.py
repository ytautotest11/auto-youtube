import os
import json
import base64
from openai import OpenAI
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydub import AudioSegment
from moviepy.editor import ImageClip, AudioFileClip

# Load keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# 1) SCRIPT GENERATION
# -------------------------------
def generate_script():
    print("📝 Generating script...")

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a short motivational YouTube script."},
            {"role": "user", "content": "Write a powerful motivational script in 40 seconds."}
        ]
    )

    return r.choices[0].message["content"]


# -------------------------------
# 2) IMAGE GENERATION
# -------------------------------
def generate_image():
    print("🖼 Generating image...")

    r = client.images.generate(
        model="gpt-image-1",
        prompt="Cinematic motivational background, dramatic lighting, ultra realistic",
        size="1024x1024"
    )

    img_b64 = r.data[0].b64_json
    img_bytes = base64.b64decode(img_b64)

    path = "visual.png"
    with open(path, "wb") as f:
        f.write(img_bytes)

    return path


# -------------------------------
# 3) VOICE GENERATION
# -------------------------------
def generate_voice(text):
    print("🎤 Generating voice...")

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    path = "voice.mp3"
    with open(path, "wb") as f:
        f.write(speech.read())

    return path


# -------------------------------
# 4) VIDEO GENERATION
# -------------------------------
def make_video(image_path, audio_path):
    print("🎬 Creating video...")

    audio = AudioSegment.from_file(audio_path)
    duration = audio.duration_seconds

    clip = ImageClip(image_path).set_duration(duration)
    clip = clip.set_audio(AudioFileClip(audio_path))

    out_path = "output_video.mp4"
    clip.write_videofile(out_path, fps=24)

    return out_path


# -------------------------------
# 5) UPLOAD TO YOUTUBE
# -------------------------------
def upload_to_youtube(video_path, title, desc):
    print("⬆ Uploading to YouTube...")

    creds = Credentials.from_authorized_user_info(json.loads(YOUTUBE_TOKEN_JSON))
    youtube = build("youtube", "v3", credentials=creds)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": desc
            },
            "status": {"privacyStatus": "public"},
        },
        media_body=video_path
    )

    response = request.execute()
    print("🎉 Upload completed:", response)


# -------------------------------
# MAIN BOT
# -------------------------------
def run():
    script = generate_script()
    img = generate_image()
    voice = generate_voice(script)
    video = make_video(img, voice)

    upload_to_youtube(video, "AI Auto Video", script)


if __name__ == "__main__":
    run()