import os
import json
import requests
from moviepy.editor import *
from gtts import gTTS

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def gemini(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText?key=" + GEMINI_KEY
    
    body = {
        "prompt": {
            "text": prompt
        }
    }
    
    r = requests.post(url, json=body, timeout=40)
    data = r.json()

    try:
        return data["candidates"][0]["output"]
    except:
        return "Gemini failed. Default script."

def generate_script():
    prompt = """
    Write a 50-second YouTube Hindi short script.
    Topic: One amazing fact that shocks people.
    Make it engaging, simple, and natural.
    """
    return gemini(prompt)

def generate_image_prompt(script):
    prompt = f"""
    Based on this script:
    {script}
    write a single-sentence image description for an AI photo.
    """
    return gemini(prompt)

def text_to_speech(text, out_file):
    tts = gTTS(text, lang="hi")
    tts.save(out_file)

def create_video(script, image_prompt):
    os.makedirs("output", exist_ok=True)

    # Generate voice
    audio_path = "output/voice.mp3"
    text_to_speech(script, audio_path)

    # Dummy background (black)
    clip = ColorClip(size=(1080,1920), color=(0,0,0), duration=50)

    # Add text overlay
    txt = TextClip(script, fontsize=50, color="white", size=(1000, None), method="caption")
    txt = txt.set_position("center").set_duration(50)

    # Add audio
    audio = AudioFileClip(audio_path)
    final = clip.set_audio(audio)

    # Add text on top
    final = CompositeVideoClip([final, txt])

    out = "output/final.mp4"
    final.write_videofile(out, fps=24, codec="libx264", audio_codec="aac")

    return out

def run():
    print("Generating script...")
    script = generate_script()

    print("Generating image prompt...")
    img_prompt = generate_image_prompt(script)

    print("Creating video...")
    video = create_video(script, img_prompt)

    print("DONE:", video)

if __name__ == "__main__":
    run()
