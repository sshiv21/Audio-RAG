import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def gemini_transcribe(audio_path):
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


def build_audio_rag_db(long_audio_path):

    # 1) Transcribe audio
    transcript = gemini_transcribe(long_audio_path)

    # 2) Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(transcript)

    print(f"Total chunks: {len(chunks)}")

    # 3) Create embeddings
    vectors = embedder.encode(chunks).astype("float32")

    # 4) Build FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # 5) Save FAISS index + chunks
    faiss.write_index(index, "audio_rag_db.index")

    with open("audio_rag_db_texts.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("📚 Saved FAISS index + text chunks")


if __name__ == "__main__":
    build_audio_rag_db("long_audio.wav")

