import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from dotenv import load_dotenv
from groq import Groq
from tts import tts_async
from chatfinal import load_graph_resources
from fastapi.responses import StreamingResponse
import aiofiles
import shutil
import traceback

# ---------------------- Init ---------------------- #
load_dotenv()
SAMPLE_RATE = 44100

app = FastAPI(title="Visitor Guide Bot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix asyncio loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load LangGraph resources
graph = load_graph_resources()

class LanguageRequest(BaseModel):
    language: str  # e.g., 'en', 'hi', 'mr', 'ta'

current_language = {"lang": "en"}  # Default to English

# ---------------------- Helpers ---------------------- #
def record_audio(duration=5):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio

def save_temp_m4a(audio_data, sample_rate):
    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
    noise_profile = audio_float[:min(len(audio_float), int(sample_rate * 0.5))]
    reduced_noise = nr.reduce_noise(y=audio_float, sr=sample_rate, y_noise=noise_profile)
    reduced_noise /= np.max(np.abs(reduced_noise)) + 1e-8
    reduced_int16 = (reduced_noise * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name
        AudioSegment(
            reduced_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        ).export(wav_path, format="wav")

    m4a_path = wav_path.replace(".wav", ".m4a")
    AudioSegment.from_wav(wav_path).export(m4a_path, format="ipod")
    os.remove(wav_path)
    return m4a_path

def transcribe_audio(m4a_path):
    client = Groq()
    with open(m4a_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
            response_format="text",
            language=current_language["lang"]
        )
    os.remove(m4a_path)
    return transcription.strip()

# ---------------------- Request Models ---------------------- #
class TextRequest(BaseModel):
    question: str

# ---------------------- API Routes ---------------------- #
@app.post("/ask")
async def ask_question(request: TextRequest):
    try:
        result = graph.invoke({"question": request.question, "context_chunks": [], "answer": "", "language": current_language["lang"]})
        answer = result["answer"]
        audio_path = await tts_async(answer,lang=current_language["lang"])

        audio_filename = os.path.basename(audio_path)
        audio_url = f"http://localhost:8000/audio/{audio_filename}"
 
        return {
            "question": request.question,
            "answer": answer,
            "audio_url": audio_url
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/transcribe")
async def transcribe_voice():
    try:
        audio_data = record_audio(duration=5)
        m4a_path = save_temp_m4a(audio_data, SAMPLE_RATE)
        transcription = transcribe_audio(m4a_path)

        return {
            "transcription": transcription
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# In-memory storage for user language (can be extended per session/user)
current_language = {"lang": "en"}  # Default to English

@app.post("/set_language")
async def set_language(request: LanguageRequest):
    supported_languages = ["en", "hi", "mr", "ta", "bn", "or"]

    if request.language not in supported_languages:
        return JSONResponse(
            content={"error": f"Unsupported language '{request.language}'"},
            status_code=400
        )

    current_language["lang"] = request.language
    return {"message": f"Language set to {request.language}"}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join("tts_outputs", filename)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    async def file_streamer():
        async with aiofiles.open(path, mode='rb') as f:
            yield await f.read()

    return StreamingResponse(file_streamer(), media_type="audio/mpeg")

