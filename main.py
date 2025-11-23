# main.py
"""
Auto YouTube video generator using Hugging Face Inference API (FREE).
Workflow:
 1) Generate short script (text)
 2) Generate image from prompt
 3) Generate TTS audio from text
 4) Combine image + audio into mp4 via moviepy
 5) Upload to YouTube (optional if credentials set)

Secrets / ENV expected:
 - HF_API_KEY                 (Hugging Face API token)
 - CLIENT_SECRET_JSON         (optional) YouTube client secret JSON string
 - YOUTUBE_TOKEN_JSON         (optional) YouTube token JSON string (for upload)
"""

import os
import json
import time
import base64
import requests
from pathlib import Path
from moviepy.editor import ImageClip, AudioFileClip

# Optional: Google API (used only if YT credentials provided)
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
except Exception:
    # google libs may not be installed; upload will be skipped if not present
    Credentials = None
    build = None

HF_API_KEY = os.getenv("HF_API_KEY")  # Must be set in GitHub Actions secrets

# Models to use (Hugging Face)
TEXT_MODEL = "google/flan-t5-large"                 # text generation
IMAGE_MODEL = "stabilityai/stable-diffusion-2"      # image generation
TTS_MODEL = "espnet/kan-bayashi_ljspeech_vits"      # text-to-speech (wav)

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def hf_text_generate(prompt: str, model: str = TEXT_MODEL, max_tokens: int = 200) -> str:
    """Generate text using HF Inference API."""
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set in environment.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens},
    }
    print(f"[HF] Text generation request to {model} ...")
    r = requests.post(url, headers=HEADERS, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Text generation failed: {r.status_code} - {r.text}")
    try:
        data = r.json()
        # output format varies by model; try common fields
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError("HF error: " + data["error"])
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # some models return plain string
        if isinstance(data, str):
            return data
        # sometimes API returns [{'generated_text': '...'}]
        if isinstance(data, list) and len(data) > 0:
            # combine text fields
            text = ""
            for item in data:
                if isinstance(item, dict):
                    text += item.get("generated_text", "") or item.get("text", "")
            if text:
                return text
        # fallback: try r.text
        return r.text
    except ValueError:
        # not json -> return raw text
        return r.text


def hf_image_generate(prompt: str, model: str = IMAGE_MODEL, out_path: Path = OUT_DIR / "visual.png") -> Path:
    """
    Generate image via huggingface inference API.
    The image endpoint returns binary image (png/jpg) when Accept header is image/*.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set in environment.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = dict(HEADERS)
    headers["Accept"] = "image/png"  # request image as PNG
    payload = {"inputs": prompt}
    print(f"[HF] Image generation request to {model} ... (this may take a while)")
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        # some error responses are JSON
        try:
            err = r.json()
            raise RuntimeError(f"Image generation failed: {r.status_code} - {err}")
        except ValueError:
            raise RuntimeError(f"Image generation failed: {r.status_code} - {r.text}")
    # response.content should be raw image bytes
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"[HF] Image saved to {out_path}")
    return out_path


def hf_tts_generate(text: str, model: str = TTS_MODEL, out_path: Path = OUT_DIR / "voice.wav") -> Path:
    """
    Generate speech audio (wav) from text.
    Some TTS models on HF return audio/wav when Accept header is audio/wav.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set in environment.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = dict(HEADERS)
    headers["Accept"] = "audio/wav"
    payload = {"inputs": text}
    print(f"[HF] TTS request to {model} ...")
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        # if model does not support direct TTS through inference endpoint, fallback to returning error
        try:
            err = r.json()
            raise RuntimeError(f"TTS failed: {r.status_code} - {err}")
        except ValueError:
            raise RuntimeError(f"TTS failed: {r.status_code} - {r.text}")
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"[HF] TTS audio saved to {out_path}")
    return out_path


def make_video(image_path: Path, audio_path: Path, out_path: Path = OUT_DIR / "output_video.mp4", fps: int = 24) -> Path:
    """
    Combine static image and audio into mp4.
    The image will be displayed for the duration of the audio.
    """
    print(f"[Video] Making video from {image_path} + {audio_path}")
    audio = AudioFileClip(str(audio_path))
    try:
        img_clip = ImageClip(str(image_path)).set_duration(audio.duration)
        img_clip = img_clip.set_audio(audio)
        img_clip.write_videofile(str(out_path), fps=fps, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    finally:
        # cleanup
        if 'audio' in locals():
            audio.close()
        if 'img_clip' in locals():
            img_clip.close()
    print(f"[Video] Saved video to {out_path}")
    return out_path


def upload_to_youtube(video_path: Path, title: str, description: str):
    """
    Upload to YouTube using google-api-client.
    Requires that CLIENT_SECRET_JSON and YOUTUBE_TOKEN_JSON exist in env.
    If google libs are missing or secrets not provided, this function will skip upload.
    """
    if Credentials is None or build is None:
        print("[YouTube] google libs not installed. Skipping upload.")
        return {"status": "skipped", "reason": "google libs not installed"}

    client_secret_json = os.getenv("CLIENT_SECRET_JSON")
    token_json = os.getenv("YOUTUBE_TOKEN_JSON")
    if not client_secret_json or not token_json:
        print("[YouTube] CLIENT_SECRET_JSON or YOUTUBE_TOKEN_JSON not provided. Skipping upload.")
        return {"status": "skipped", "reason": "missing credentials"}

    # load token and create Credentials object
    token_data = json.loads(token_json)
    creds = Credentials.from_authorized_user_info(token_data, scopes=["https://www.googleapis.com/auth/youtube.upload"])

    youtube = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {"title": title, "description": description},
        "status": {"privacyStatus": "public"},
    }
    media_body = {"mimeType": "video/*", "body": open(video_path, "rb")}
    # googleapiclient's MediaFileUpload is preferred, but to keep dependency list small we use media_body pattern below.
    # Use recommended upload if available:
    try:
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        print("[YouTube] Uploading video...")
        response = request.execute()
        print("[YouTube] Upload complete:", response.get("id"))
        return {"status": "uploaded", "id": response.get("id")}
    except Exception as e:
        print("[YouTube] Upload failed:", e)
        return {"status": "error", "error": str(e)}


def generate_script():
    print("Generating script...")
    # Prompt for the model: keep it short and structured
    prompt = (
        "Write a short (40-80 words) motivational YouTube video script in Hindi. "
        "The script should be punchy, in simple Hindi, include a one-line hook at the start and a call-to-action at the end."
    )
    text = hf_text_generate(prompt)
    # If the model returns long text, trim
    if isinstance(text, str):
        cleaned = text.strip()
        return cleaned
    return str(text)


def run():
    start = time.time()
    try:
        script = generate_script()
        print("Script:\n", script)

        # Use the first line or phrase of the script as image prompt
        image_prompt = script.split("\n")[0][:200] + " cinematic, high-quality, 16:9"
        image_path = hf_image_generate(image_prompt, model=IMAGE_MODEL, out_path=OUT_DIR / "visual.png")

        # TTS - convert whole script to speech
        audio_path = hf_tts_generate(script, model=TTS_MODEL, out_path=OUT_DIR / "voice.wav")

        # Create video
        video_path = make_video(image_path, audio_path, out_path=OUT_DIR / "video.mp4")

        # Upload if possible
        title = "AI Auto Video - " + (script.split("\n")[0][:50] if script else "Auto Video")
        desc = script + "\n\nGenerated automatically with Hugging Face"
        upload_result = upload_to_youtube(video_path, title, desc)
        print("Upload result:", upload_result)

    except Exception as e:
        print("Error:", e)
        raise
    finally:
        print("Total elapsed:", time.time() - start)


if __name__ == "__main__":
    run()
