import os
from flask import Flask, request, jsonify, send_file
from llm_agent import run_audio_rag_agent

app = Flask(__name__)

# Load Gemini key from Railway environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


@app.route("/")
def home():
    return {"message": "Audio RAG Agent Running on Railway!"}


@app.route("/process_audio", methods=["POST"])
def process_audio():
    """
    Upload user audio -> process -> return text + audio file link
    """

    if "audio" not in request.files:
        return {"error": "Please upload an audio file."}, 400

    file = request.files["audio"]
    file_path = "user_query.wav"
    file.save(file_path)

    # Run your agent
    text_answer, audio_answer_path = run_audio_rag_agent(file_path)

    return send_file(audio_answer_path, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
