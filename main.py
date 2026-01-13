from updated_llm_agent import run_audio_rag_agent

if __name__ == "__main__":
    text_answer, audio_answer, time_stamp = run_audio_rag_agent("user_question.wav")

    print("\n==============================")
    print("FINAL TEXT ANSWER:")
    print(text_answer)
    print("==============================")

    print("Audio saved as:", audio_answer)

    print("------------------------------")
    print("Relevant time stamps:")
    print(time_stamp)
