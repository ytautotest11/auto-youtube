#!/usr/bin/env python3
"""
main.py
Full pipeline:
1) Generate/Evaluate script (Gemini/LLM)
2) Produce voiceovers (Gemini TTS or other TTS)
3) Build visuals (images + text slides)
4) Compose video in MoviePy, add background music and ducking
5) Auto-generate thumbnail (PIL)
6) Create description & tags (LLM)
7) Auto-detect Shorts (<=60s or vertical) and format
8) Upload to YouTube (two-channel support)
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import math
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List

import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, CompositeAudioClip, afx
)
from moviepy.audio.fx.all import audio_fadein, audio_fadeout

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration (via env)
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # required for LLM + TTS (or swap with your provider)
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "")  # optional
YOUTUBE_CLIENT_SECRETS = os.getenv("CLIENT_SECRET_JSON")  # path or raw JSON in env
YOUTUBE_TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON")  # path or raw token
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
TEMPLATE_IMAGES_DIR = Path(os.getenv("TEMPLATE_IMAGES_DIR", "assets/images"))
BACKGROUND_MUSIC_DIR = Path(os.getenv("BACKGROUND_MUSIC_DIR", "assets/music"))
FONT_PATH = os.getenv("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
# Two channels mapping. Provide CHANNEL_A_ID and/or CHANNEL_B_ID and their corresponding credentials if needed.
CHANNEL_MAPPING = {
    "channel_a": {
        "youtube_channel_id": os.getenv("YOUTUBE_CHANNEL_A_ID"),
        "credentials_secret": os.getenv("YOUTUBE_CREDENTIALS_A")  # optional: if separate
    },
    "channel_b": {
        "youtube_channel_id": os.getenv("YOUTUBE_CHANNEL_B_ID"),
        "credentials_secret": os.getenv("YOUTUBE_CREDENTIALS_B")
    }
}
# default voice settings
VOICE_MAP = {
    "hi": {"voice": "hi-in-1", "lang": "hi-IN"},
    "en": {"voice": "en-us-1", "lang": "en-US"}
}

# Create output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helper: LLM wrapper (script, description, tags)
# ----------------------------
def llm_generate(prompt: str, system: str = "", max_tokens: int = 400) -> str:
    """
    Minimal LLM call wrapper. Uses GEMINI_API_URL/GEMINI_API_KEY if provided.
    If you use a different LLM, replace inside.

    This function expects a JSON response with 'text' key — adjust as needed.
    """
    if not GEMINI_API_KEY or not GEMINI_API_URL:
        # Fallback: If user hasn't set LLM keys, raise informative error
        raise RuntimeError("GEMINI_API_KEY and GEMINI_API_URL must be set in environment to use LLM features.")

    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    # NOTE: adapt this to the exact Gemini endpoint you have access to.
    resp = requests.post(GEMINI_API_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # try to extract text in several common ways
    if "text" in data:
        return data["text"]
    if "output" in data and isinstance(data["output"], dict) and "text" in data["output"]:
        return data["output"]["text"]
    # last fallback
    return json.dumps(data)


# ----------------------------
# Helper: TTS wrapper (returns local file path)
# ----------------------------
def tts_generate(text: str, lang_code: str = "en", filename: str = None) -> Path:
    """
    Generate TTS audio from text. Uses Gemini TTS or alternative TTS endpoint.
    Returns path to a generated WAV/MP3 file.
    """
    if not GEMINI_API_KEY or not GEMINI_API_URL:
        raise RuntimeError("GEMINI_API_KEY and GEMINI_API_URL must be set for TTS.")

    if filename is None:
        filename = OUTPUT_DIR / f"tts_{lang_code}_{int(datetime.utcnow().timestamp())}.mp3"
    else:
        filename = Path(filename)
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "text": text,
        "voice": VOICE_MAP.get("hi" if lang_code.startswith("hi") else "en")["voice"],
        "language": lang_code
    }
    # NOTE: adapt to available TTS API — here we assume POST returns binary audio.
    tts_endpoint = os.getenv("GEMINI_TTS_URL", GEMINI_API_URL.rstrip("/") + "/tts")
    r = requests.post(tts_endpoint, headers=headers, json=body, stream=True, timeout=120)
    r.raise_for_status()
    # write binary
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info("TTS generated: %s", filename)
    return filename


# ----------------------------
# Build visuals: slides or image sequence
# ----------------------------
def build_slides_from_script(script_sentences: List[str],
                             lang: str,
                             slide_duration: float = 4.0,
                             resolution=(1280, 720),
                             vertical=False) -> List[ImageClip]:
    """
    Create a list of ImageClips (slides) from sentences using PIL, then convert to ImageClip.
    """
    clips = []
    w, h = resolution if not vertical else (720, 1280)
    font = ImageFont.truetype(FONT_PATH, size=44 if not vertical else 56)
    logger.info("Building %d slides (vertical=%s)", len(script_sentences), vertical)

    for i, sentence in enumerate(script_sentences):
        img = Image.new("RGB", (w, h), color=(12, 12, 12))
        draw = ImageDraw.Draw(img)
        margin = 40
        # wrap text roughly
        import textwrap
        max_chars = 28 if vertical else 40
        lines = textwrap.wrap(sentence, width=max_chars)
        y = margin
        for line in lines:
            tw, th = draw.textsize(line, font=font)
            x = (w - tw) // 2
            draw.text((x, y), line, font=font, fill=(255, 255, 255))
            y += th + 8
        # optional: overlay a template image if available
        template_path = next(TEMPLATE_IMAGES_DIR.glob("*.png"), None) if TEMPLATE_IMAGES_DIR.exists() else None
        if template_path:
            tpl = Image.open(template_path).convert("RGBA").resize((w, h))
            img = Image.alpha_composite(img.convert("RGBA"), tpl).convert("RGB")
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(tmpfile.name, format="PNG")
        clip = ImageClip(tmpfile.name).set_duration(slide_duration)
        if vertical:
            clip = clip.resize(height=1280)
        clips.append(clip)
    return clips


# ----------------------------
# Combine audio (voice + bg music with ducking)
# ----------------------------
def combine_audio_with_music(voice_path: str, music_path: str, output_path: str,
                             target_voice_db: float = -18.0) -> str:
    """
    Creates a mixed audio: background music looped and ducked under voice.
    Returns path to final audio file.
    """
    voice = AudioFileClip(voice_path)
    music = AudioFileClip(music_path).volumex(0.6)

    # loop music to cover voice duration + buffer
    music_duration = max(voice.duration + 2, music.duration)
    music = afx.audio_loop(music, duration=music_duration)

    # Simple ducking: reduce music volume during voice
    # Implementation: create music_clip where during voice segments volume lowered
    # MoviePy doesn't have direct per-sample envelopes easily; using CompositeAudioClip with volumes.
    music = music.audio_fadein(0.5)
    # Lower music volume when voice is playing by overlaying a muted segment? We'll approximate:
    # Implementation: mix voice + music with music at lower volume overall.
    final_music = music.volumex(0.25)
    mixed = CompositeAudioClip([final_music, voice])
    out = Path(output_path)
    mixed.write_audiofile(str(out), fps=44100, bitrate="192k")
    logger.info("Combined audio written to %s", out)
    return str(out)


# ----------------------------
# Build video (slides + voice)
# ----------------------------
def build_video_from_slides_and_audio(slides: List[ImageClip], audio_path: str,
                                      resolution=(1280, 720), is_shorts=False) -> Path:
    """
    Assemble slides into a video, set audio, and export MP4.
    If is_shorts=True, ensure vertical format and <=60s; otherwise normal.
    """
    # calculate durations and set fps
    video_duration = sum(clip.duration for clip in slides)
    # Concatenate
    concatenated = concatenate_videoclips(slides, method="compose")
    audio = AudioFileClip(audio_path)
    # If audio and slides duration mismatch, adjust slides durations to fit audio
    if abs(video_duration - audio.duration) > 0.5:
        # proportionally scale durations
        scale = audio.duration / video_duration
        logger.info("Scaling slide durations by %.3f to match audio duration", scale)
        for clip in slides:
            clip._duration = clip.duration * scale
        concatenated = concatenate_videoclips(slides, method="compose")

    final = concatenated.set_audio(audio)
    # Adjust size for shorts
    if is_shorts:
        final = final.resize(height=1280)  # will keep aspect
        # Optional: crop to 9:16
        final = final.crop(x_center=final.w // 2, y_center=final.h // 2, width=720, height=1280)
    else:
        final = final.resize(height=720)

    out_path = OUTPUT_DIR / f"final_{'short' if is_shorts else 'long'}_{int(datetime.utcnow().timestamp())}.mp4"
    final.write_videofile(str(out_path), codec="libx264", audio_codec="aac", threads=4,
                          preset="medium", fps=24, verbose=False, logger=None)
    logger.info("Video exported to %s", out_path)
    return out_path


# ----------------------------
# Thumbnail generation
# ----------------------------
def generate_thumbnail(title_hi: str, title_en: str, output_file: Path, resolution=(1280, 720)):
    """
    Generate a thumbnail with both Hindi and English titles using PIL.
    """
    w, h = resolution
    img = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    font_large = ImageFont.truetype(FONT_PATH, 56)
    font_small = ImageFont.truetype(FONT_PATH, 36)

    # Title EN top, Title HI bottom
    en_text = title_en[:60]
    hi_text = title_hi[:60]

    tw, th = draw.textsize(en_text, font=font_large)
    draw.text(((w - tw) / 2, h * 0.15), en_text, font=font_large, fill=(255, 255, 255))

    tw, th = draw.textsize(hi_text, font=font_small)
    draw.text(((w - tw) / 2, h * 0.7), hi_text, font=font_small, fill=(255, 255, 0))

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_file)
    logger.info("Thumbnail generated at %s", output_file)
    return output_file


# ----------------------------
# Auto description and tags
# ----------------------------
def generate_description_and_tags(title: str, script: str, language: str = "en") -> Tuple[str, List[str]]:
    """
    Use LLM to generate a YouTube description and tags for given title+script.
    Returns (description, tags).
    """
    prompt = (f"Generate a YouTube video description and 12 comma-separated tags for a video titled: '{title}' "
              f"in {language}. Include a short 3-line description, a 1-line hashtags list and 12 tags.")
    out = llm_generate(prompt, max_tokens=300)
    # naive parse: expect 'DESCRIPTION: ... TAGS: a,b,c'
    # We'll return raw text and tags list if detected; caller may adjust.
    # Try to find 'TAGS:' substring
    tags = []
    if "TAGS" in out.upper():
        try:
            # crude split
            tag_part = out.split("TAGS", 1)[1]
            # remove punctuation, newlines
            tag_part = tag_part.replace(":", "").replace("\n", " ")
            # extract words/phrases separated by commas
            tags = [t.strip() for t in tag_part.split(",") if t.strip()]
        except Exception:
            tags = []
    # fallback: pick last line commas
    if not tags:
        lines = out.strip().splitlines()
        if len(lines) > 0:
            candidate = lines[-1]
            if "," in candidate:
                tags = [t.strip() for t in candidate.split(",") if t.strip()]
    # return
    return out.strip(), tags


# ----------------------------
# Upload to YouTube (placeholder)
# ----------------------------
def upload_to_youtube(video_file: Path, title: str, description: str, tags: List[str],
                      thumbnail: Path, channel_key: Dict[str, Any], is_shorts: bool):
    """
    Upload video to YouTube using Google API client. Here we include a placeholder
    implementation outline. Replace with googleapiclient code and credentials logic.

    channel_key: dict containing 'youtube_channel_id' and optionally 'credentials_secret'
    """
    # NOTE: Implement OAuth2 authentication using CLIENT_SECRET_JSON and token storage.
    # For security, pass credentials via environment secrets and store token file path in YOUTUBE_TOKEN_JSON.
    #
    # Minimal outline:
    #   from googleapiclient.discovery import build
    #   from googleapiclient.http import MediaFileUpload
    #   credentials = load_credentials_from_env(...)  # implement
    #   youtube = build('youtube', 'v3', credentials=credentials)
    #
    #   request = youtube.videos().insert(
    #       part="snippet,status",
    #       body={
    #           "snippet": {"title": title, "description": description, "tags": tags, "categoryId": "22"},
    #           "status": {"privacyStatus": "public"}
    #       },
    #       media_body=MediaFileUpload(str(video_file), chunksize=-1, resumable=True)
    #   )
    #   response = request.execute()
    #
    #   youtube.thumbnails().set(videoId=response["id"], media_body=MediaFileUpload(thumbnail)).execute()
    #
    logger.info("Uploading to YouTube: %s (channel: %s) SHORTS=%s", video_file, channel_key.get("youtube_channel_id"), is_shorts)
    # Raise NotImplementedError to remind you to add real upload code.
    # If you prefer, I can supply a ready-to-use googleapiclient implementation in a follow up.
    raise NotImplementedError("YouTube upload not implemented in this script. See comments in function for implementation steps.")


# ----------------------------
# Utilities
# ----------------------------
def split_script_into_sentences(script: str) -> List[str]:
    # Very simple sentence split; for production use a sentence tokenizer
    return [s.strip() for s in script.replace("\n", " ").split(".") if s.strip()]


def detect_shorts(video_duration: float, target_vertical: bool = False) -> bool:
    """
    Shorts criteria: <= 60 seconds OR vertical flag set.
    """
    if target_vertical:
        return True
    return video_duration <= 60.0


# ----------------------------
# Main pipeline
# ----------------------------
def pipeline(topic_prompt: str, channel_key_name: str = "channel_a",
             prefer_vertical: bool = False):
    """
    Full run:
    - get script in English and Hindi
    - produce TTS
    - build slides
    - combine audio+music
    - build video (two language outputs)
    - thumbnail, description, tags
    - detect shorts
    - upload
    """
    logger.info("Pipeline start for topic: %s", topic_prompt)
    nowstr = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    working_dir = OUTPUT_DIR / f"job_{nowstr}"
    working_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate script EN and HI
    prompt_en = f"Write a concise, engaging YouTube video script about: {topic_prompt} in English. Keep it 180-300 words."
    prompt_hi = f"Write a concise, engaging YouTube video script about: {topic_prompt} in Hindi. Keep it 180-300 words in Devanagari."

    logger.info("Generating English script...")
    script_en = llm_generate(prompt_en, max_tokens=600)
    logger.info("Generating Hindi script...")
    script_hi = llm_generate(prompt_hi, max_tokens=600)

    # Optionally shorten or split into sentences
    sentences_en = split_script_into_sentences(script_en)
    sentences_hi = split_script_into_sentences(script_hi)

    # 2) Get TTS for both languages
    logger.info("Generating TTS English...")
    tts_en = tts_generate(script_en, lang_code="en-US", filename=working_dir / "voice_en.mp3")
    logger.info("Generating TTS Hindi...")
    tts_hi = tts_generate(script_hi, lang_code="hi-IN", filename=working_dir / "voice_hi.mp3")

    # 3) Choose background music (first available)
    bg_music = None
    if BACKGROUND_MUSIC_DIR.exists():
        candidates = list(BACKGROUND_MUSIC_DIR.glob("*.mp3"))
        if candidates:
            bg_music = str(candidates[0])
    if not bg_music:
        raise FileNotFoundError("No background music found in BACKGROUND_MUSIC_DIR. Place an mp3 file there.")

    # 4) Combine audio (voice + music) with ducking
    combined_en = combine_audio_with_music(tts_en, bg_music, str(working_dir / "mix_en.mp3"))
    combined_hi = combine_audio_with_music(tts_hi, bg_music, str(working_dir / "mix_hi.mp3"))

    # 5) Build slides and video for both languages
    is_shorts = detect_shorts(audio_path_to_duration(combined_en), target_vertical=prefer_vertical)

    slides_en = build_slides_from_script(sentences_en, lang="en", slide_duration=4.0,
                                         resolution=(720, 1280) if prefer_vertical else (1280, 720),
                                         vertical=prefer_vertical)
    slides_hi = build_slides_from_script(sentences_hi, lang="hi", slide_duration=4.0,
                                         resolution=(720, 1280) if prefer_vertical else (1280, 720),
                                         vertical=prefer_vertical)

    video_en = build_video_from_slides_and_audio(slides_en, combined_en,
                                                 resolution=(720, 1280) if prefer_vertical else (1280, 720),
                                                 is_shorts=is_shorts)
    video_hi = build_video_from_slides_and_audio(slides_hi, combined_hi,
                                                 resolution=(720, 1280) if prefer_vertical else (1280, 720),
                                                 is_shorts=is_shorts)

    # 6) Thumbnail
    thumbnail_path = working_dir / "thumbnail.jpg"
    generate_thumbnail(title_hi=script_hi.splitlines()[0] if script_hi else topic_prompt,
                       title_en=script_en.splitlines()[0] if script_en else topic_prompt,
                       output_file=thumbnail_path)

    # 7) Description & tags
    desc_en, tags_en = generate_description_and_tags(title=script_en.splitlines()[0] if script_en else topic_prompt,
                                                     script=script_en, language="English")
    desc_hi, tags_hi = generate_description_and_tags(title=script_hi.splitlines()[0] if script_hi else topic_prompt,
                                                     script=script_hi, language="Hindi")

    # 8) Upload to YouTube (two channels support)
    channel_key = CHANNEL_MAPPING.get(channel_key_name)
    if not channel_key:
        raise RuntimeError(f"Channel mapping '{channel_key_name}' not found in CHANNEL_MAPPING.")

    # For each language, prepare title and description lines
    title_en = (script_en.splitlines()[0] if script_en else topic_prompt)[:100]
    title_hi = (script_hi.splitlines()[0] if script_hi else topic_prompt)[:100]

    try:
        upload_to_youtube(video_en, title_en, desc_en, tags_en, thumbnail_path, channel_key, is_shorts)
    except NotImplementedError:
        logger.warn
