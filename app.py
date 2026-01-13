import streamlit as st
from streamlit_mic_recorder import mic_recorder
from updated_index_audio import index_audio_file
from updated_llm_agent import run_audio_rag_agent
from pydub import AudioSegment
import uuid
import os

st.set_page_config(page_title="🎧 Audio RAG", layout="wide")

# --------------------------------------------
# SIDEBAR = Knowledge Upload (Fixed)
# --------------------------------------------
with st.sidebar:
    st.header("📚 Knowledge Upload")

    files = st.file_uploader("Upload knowledge audio", type=["mp3","wav","m4a"], accept_multiple_files=True)
    if files:
        for f in files:
            name = f"up_{uuid.uuid4().hex}_{f.name}"
            with open(name,"wb") as out:
                out.write(f.read())
            index_audio_file(name)
            st.success(f"Indexed {name}")

    st.markdown("---")

    st.subheader("🎧 Upload Question Audio")
    q = st.file_uploader("Upload question audio", type=["mp3","wav","m4a"])
    if q:
        with open("q_user.wav","wb") as o:
            o.write(q.read())
        st.session_state["question_audio_path"] = "q_user.wav"
        st.audio("q_user.wav")
        st.success("Audio question ready!")


# --------------------------------------------
# CHAT STATE
# --------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "question_audio_path" not in st.session_state:
    st.session_state.question_audio_path = None


# --------------------------------------------
# MAIN CHAT PAGE (FULL WIDTH)
# --------------------------------------------

st.title("💬 AI Chat")

# ---- CSS FOR CHATGPT UI ----
st.markdown("""
<style>

.chat-box {
    height: 70vh;
    padding: 15px;
    overflow-y: auto;
    background: #1f1f1f;
    border-radius: 12px;
    border: 1px solid #333;
}

.user-bubble {
    text-align: right;
    margin: 8px;
}
.user-bubble span {
    background: #4c6ef5;
    color: white;
    padding: 10px 15px;
    border-radius: 12px;
}

.ai-bubble {
    text-align: left;
    margin: 8px;
}
.ai-bubble span {
    background: #2e2e2e;
    color: white;
    padding: 10px 15px;
    border-radius: 12px;
}

.input-bar {
    position: fixed;
    bottom: 15px;
    right: 25px;
    width: 60%;
    padding: 15px;
    background: #0f0f0f;
    border: 2px solid #333;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---- CHAT WINDOW ----
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

for m in st.session_state.chat:

    if m["role"] == "user":
        st.markdown(f"<div class='user-bubble'><span>{m['text']}</span></div>", unsafe_allow_html=True)

    else:
        st.markdown(f"<div class='ai-bubble'><span>{m['text']}</span></div>", unsafe_allow_html=True)

        if m.get("audio"):
            st.audio(m["audio"])

        if m.get("timestamps"):
            st.markdown("**⏱ Relevant Segments:**")
            for ts in m["timestamps"]:
                st.write(f"{ts['audio_file']} — {ts['start_ts']} → {ts['end_ts']}")

st.markdown("</div>", unsafe_allow_html=True)


# ---- FIXED INPUT BAR (Bottom-right) ----
st.markdown("<div class='input-bar'>", unsafe_allow_html=True)

text_q = st.text_input("Your question:", key="text_input")

rec = mic_recorder(
    start_prompt="🎤 Record",
    stop_prompt="⏹ Stop",
    format="wav",
    key="rec"
)

if rec:
    with open("recorded_q.wav","wb") as f:
        f.write(rec["bytes"])
    st.session_state.question_audio_path = "recorded_q.wav"
    st.audio("recorded_q.wav")

send = st.button("🚀 Send")

if send:
    if text_q.strip():
        st.session_state.chat.append({"role":"user","text":text_q})
        ans, ans_aud, ts = run_audio_rag_agent(text_q, is_text=True)
        st.session_state.chat.append({"role":"ai","text":ans,"audio":ans_aud,"timestamps":ts})

    elif st.session_state.question_audio_path:
        st.session_state.chat.append({"role":"user","text":"🎤 (audio message)"})
        ans, ans_aud, ts = run_audio_rag_agent(st.session_state.question_audio_path, is_text=False)
        st.session_state.chat.append({"role":"ai","text":ans,"audio":ans_aud,"timestamps":ts})

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
