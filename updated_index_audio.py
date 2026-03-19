import google.generativeai as genai
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def ensure_wav_16k(in_path, out_path):
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path


def gemini_transcribe(audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(
        {
            "parts": [{
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_bytes
                }
            }]
        },
        generation_config={"response_mime_type": "text/plain"}
    )
    return resp.text


def index_audio_file(audio_path):
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    safe_audio = os.path.join(DATA_DIR, f"{filename}.wav")

    print(f"🎤 Converting {audio_path} → {safe_audio}")
    safe_audio = ensure_wav_16k(audio_path, safe_audio)

    print("📝 Transcribing...")
    transcript = gemini_transcribe(safe_audio)

    audio = AudioSegment.from_wav(safe_audio)
    duration_ms = len(audio)

    words = transcript.split()
    chunks, timestamps = [], []

    chunk_size = 350
    current_text = ""
    current_start = 0
    cursor = 0
    total_len = len(transcript)

    for w in words:
        if len(current_text) + len(w) + 1 < chunk_size:
            current_text += " " + w
        else:
            end = int((cursor / total_len) * duration_ms)
            chunks.append(current_text.strip())
            timestamps.append((current_start, end))
            current_text = w
            current_start = end
        cursor += len(w) + 1

    chunks.append(current_text.strip())
    timestamps.append((current_start, duration_ms))

    vectors = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, os.path.join(DATA_DIR, f"{filename}.index"))

    pickle.dump(
        {"chunks": chunks, "timestamps": timestamps, "audio_file": f"{filename}.wav"},
        open(os.path.join(DATA_DIR, f"{filename}.pkl"), "wb")
    )

    print(f"✅ Indexed {filename}")
