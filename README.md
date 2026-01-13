🎧 Audio RAG – Multi-Knowledge Audio Question Answering System

This project implements an **Audio-based Retrieval-Augmented Generation (RAG)** system that allows users to upload **audio knowledge files** and ask questions using **text or voice**. The system transcribes audio, converts content into embeddings, retrieves relevant context, and generates accurate answers using an LLM.
## 🚀 Features

* 🎙️ Upload **multiple audio knowledge files**
* 🔊 Supports **voice-based questions**
* 📝 Automatic **audio transcription**
* ✂️ Intelligent **text chunking**
* 🧠 Semantic search using **vector embeddings**
* 🤖 LLM-powered answer generation
* 📦 Modular & scalable architecture

---

## 🧩 Architecture Overview

**High-level workflow:**

1. User uploads an **audio knowledge file**
2. Audio is converted to **16 kHz WAV**
3. Audio is transcribed into text
4. Text is **chunked**
5. Chunks are converted into **vector embeddings**
6. Embeddings are stored in a **vector database**
7. User asks a question (voice/text)
8. Relevant chunks are retrieved
9. LLM generates a **context-aware answer**

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Audio Processing:** Pydub
* **Speech-to-Text:** Google / Gemini Transcription
* **Embeddings:** Sentence Transformers
* **Vector Store:** FAISS / ChromaDB
* **LLM:** Gemini / OpenAI (configurable)
* **Language:** Python

---

## 📂 Project Structure

```bash
audio-rag/
│
├── app.py                     # Streamlit app entry point
├── updated_index_audio.py     # Audio indexing & embedding pipeline
├── updated_llm_agent.py       # RAG-based LLM response logic
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── data/
│   ├── audio/                 # Uploaded audio files
│   └── vectors/               # Stored embeddings
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/audio-rag.git
cd audio-rag
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧠 Why 16 kHz WAV?

Speech-to-text models are optimized for:

* **16,000 Hz sample rate**
* **Mono-channel audio**
* **Uncompressed WAV format**

This ensures:

* Better transcription accuracy
* Lower noise distortion
* Model compatibility

---

## 📌 Use Cases

* 📚 Audio-based knowledge assistants
* 🎓 Lecture & meeting Q&A systems
* 📞 Call center intelligence
* 🎧 Podcast content search
* 🏥 Medical or legal audio analysis

---

## 🔮 Future Improvements

* Speaker diarization
* Multilingual support
* Streaming audio ingestion
* Metadata-based retrieval
* Cloud vector storage

---

## 👤 Author

**Shivam Sharma**
