import os
import google.generativeai as genai
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import pyttsx3


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load local embedding model (FREE)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------------------------------------
# FREE OFFLINE TTS (pyttsx3)
# -------------------------------------------------------------
engine = pyttsx3.init()

# Try to set female voice (Windows: "Zira")
voices = engine.getProperty("voices")
for v in voices:
    if "female" in v.name.lower() or "zira" in v.id.lower():
        engine.setProperty("voice", v.id)
        break

engine.setProperty("rate", 150)  # speaking speed


def generate_audio(text, out_path="agent_response.wav"):
    """Convert text to speech locally using pyttsx3 (offline + free)."""
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path


# -------------------------------------------------------------
# 1️⃣ Gemini Transcription (Audio → Text)
# -------------------------------------------------------------
def gemini_transcribe(audio_path):
    """Transcribe audio using Gemini 1.5 Flash."""
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(
        {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_bytes
                    }
                }
            ]
        },
        generation_config={"response_mime_type": "text/plain"}
    )

    return response.text


# -------------------------------------------------------------
# 2️⃣ Embedding (SentenceTransformer)
# -------------------------------------------------------------
def embed(text: str):
    """Create embedding vector for text."""
    return embedder.encode([text]).astype("float32")


# -------------------------------------------------------------
# 3️⃣ MAIN AGENT PIPELINE
# -------------------------------------------------------------
def run_audio_rag_agent(user_audio_path):
    """
    Entire pipeline:
    - Transcribe user audio
    - Embed query
    - Load FAISS DB
    - Retrieve relevant text chunks
    - Ask Gemini LLM
    - Convert answer to local offline audio
    """

    # Load FAISS vector index
    index = faiss.read_index("audio_rag_db.index")

    # Load stored transcript chunks
    with open("audio_rag_db_texts.pkl", "rb") as f:
        chunks = pickle.load(f)

    print("📚 Loaded FAISS database")

    # 1) STT: Transcribe audio
    print("🎤 Transcribing user question...")
    query = gemini_transcribe(user_audio_path)
    print("User asked:", query)

    # 2) Embed the query
    q_vec = embed(query)

    # 3) Retrieve top-k relevant chunks
    print("🔍 Retrieving relevant information...")
    D, I = index.search(q_vec, k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(retrieved_chunks)

    # 4) Ask Gemini for final answer
    print("🤖 Generating final answer with Gemini...")

    prompt = f"""
Use the following context extracted from my audio database to answer the user's question.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (clear, short, and helpful):
"""

    answer = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(prompt).text

    print("💬 Agent Answer:", answer)

    # 5) Convert answer to offline audio
    print("🔊 Generating speech output (offline)...")
    audio_file = generate_audio(answer)

    print("✅ Audio response saved as:", audio_file)

    return answer, audio_file


# -------------------------------------------------------------
# Optional quick test
# -------------------------------------------------------------
if __name__ == "__main__":
    ans, audio = run_audio_rag_agent("user_question.wav")
    print("Final Answer:", ans)
    print("Audio File:", audio)

