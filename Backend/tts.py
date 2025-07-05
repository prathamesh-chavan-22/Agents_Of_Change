import os
import uuid
import asyncio
from sarvamai import SarvamAI
from sarvamai.play import save
import edge_tts
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")

# Initialize SarvamAI client once
sarvam_client = SarvamAI(api_subscription_key=API_KEY)

# Async TTS function
async def tts_async(text: str, lang: str = "en") -> str:
    os.makedirs("tts_outputs", exist_ok=True)
    tmp_filename = f"{uuid.uuid4()}"
    
    if lang == "or":  # Odia
        # Use SarvamAI Swayam model for Odia
        audio = sarvam_client.text_to_speech.convert(
            target_language_code="od-IN",
            text=text,
            model="bulbul:v2",
            speaker="manisha"
        )
        output_path = f"tts_outputs/{tmp_filename}.wav"
        save(audio, output_path)
        return output_path

    else:
        # Use Edge TTS for all other languages
        voices = {
            "en": "en-US-AriaNeural",
            "hi": "hi-IN-SwaraNeural",
            "mr": "mr-IN-AarohiNeural",
            "ta": "ta-IN-PallaviNeural",
            "bn": "bn-IN-TanishaaNeural"
        }
        voice = voices.get(lang, "en-US-AriaNeural")
        output_path = f"tts_outputs/{tmp_filename}.mp3"

        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)

        return output_path
