import os
import uuid
import pickle
import faiss
import numpy as np
import pyttsx3
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def ensure_wav_16k(in_path, out_path="temp_q.wav"):
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path


def gemini_transcribe(audio_path):
    safe = ensure_wav_16k(audio_path)
    with open(safe, "rb") as f:
        audio_bytes = f.read()
    resp = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(
        {"parts": [{
            "inline_data": {"mime_type": "audio/wav", "data": audio_bytes}
        }]},
        generation_config={"response_mime_type": "text/plain"}
    )
    return resp.text


def embed(text):
    return embedder.encode([text]).astype("float32")


def ms_to_timestamp(ms):
    m = int(ms / 60000)
    s = (ms % 60000) / 1000
    return f"{m:02d}:{s:04.1f}"


def generate_audio(text):
    out = f"answer_{uuid.uuid4().hex}.wav"
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(text, out)
    engine.runAndWait()
    return out


def load_all_knowledge():
    dbs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".pkl"):
            base = f.replace(".pkl", "")
            index = faiss.read_index(os.path.join(DATA_DIR, f"{base}.index"))
            meta = pickle.load(open(os.path.join(DATA_DIR, f"{base}.pkl"), "rb"))
            dbs.append((base, index, meta))
    return dbs


def run_audio_rag_agent(user_input, is_text=False):
    # If text question → directly use it
    if is_text:
        query = user_input
    else:
        query = gemini_transcribe(user_input)

    qvec = embed(query)

    dbs = load_all_knowledge()
    all_hits = []

    for base, index, meta in dbs:
        D, I = index.search(qvec, k=3)

        for dist, idx in zip(D[0], I[0]):
            all_hits.append({
                "chunk": meta["chunks"][idx],
                "start": meta["timestamps"][idx][0],
                "end": meta["timestamps"][idx][1],
                "audio_file": meta["audio_file"],
                "distance": float(dist)
            })

    all_hits.sort(key=lambda x: x["distance"])
    best = all_hits[:3]

    context = "\n".join([b["chunk"] for b in best])

    prompt = f"""
Use the following context to answer the user's question.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    answer = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(prompt).text

    for b in best:
        b["start_ts"] = ms_to_timestamp(b["start"])
        b["end_ts"] = ms_to_timestamp(b["end"])

    audio_out = generate_audio(answer)

    return answer, audio_out, best
