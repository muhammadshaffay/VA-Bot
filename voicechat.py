# imports
from chat import Bot
import os
import streamlit as st
from audiorecorder import audiorecorder
from speech_to_text import stt_model, transcribe
from text_to_speech import tts_model, speak
from IPython.display import Audio


#### #### #### Page Title & Configuration #### #### ####

# Configuration
st.set_page_config(page_title="Tech Tutor LLM")

# Title + Description
st.title("FineTuned LLM")
with st.expander("Disclaimer"):
  st.write("Powered by fine-tuned version of Llama 2 - 7B Chat HF. This bot is exclusively for computational questions (PF, OOP, DSA) with limited coding. Only English Language is supported for now. Responses may be delayed due to computing constraints.")


#### #### #### Side Bar #### #### ####

# Heading + About Section
with st.sidebar:
  st.markdown("<h1 style='text-align: center;'>🎓Tech Tutor</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center;'>F23-058-D-eLearnPlatform</h3>", unsafe_allow_html=True)
  st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); 
  st.markdown("<h3 style='text-align: center;'>About Us</h3>", unsafe_allow_html=True)
  st.markdown("<p style='text-align: center;'>At the forefront of educational innovation, our project revolutionizes online learning for those passionate about Programming and Computing. Through advanced AI agents, interactive quizzes, and a virtual whiteboard, we provide personalized, real-time support. Our mission is to make the complex world of computing accessible, engaging, and interactive for learners, empowering them to master key concepts with confidence. Welcome to a new era of online education.</p>", unsafe_allow_html=True)
  st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); 

# Clear Chat Button
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", 
                                "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History and Reset All Settings', on_click=clear_chat_history)


#### #### #### Load Essentials #### #### ####

# A. Speech-To-Text 
if "stt" not in st.session_state:
    st.session_state.stt = stt_model()
    print("\n\n\t\tNOTIFICATION: STT Loaded.\n\n")

# B. Large-Language-Model
if "bot" not in st.session_state:
    bot = Bot()
    st.session_state.bot = bot.build_chain(use_api=True)
    print("\n\n\t\tNOTIFICATION: LLM Loaded.\n\n")

# C. Text-To-Speech
if "tts" not in st.session_state:
    model, embedding = tts_model()
    st.session_state.tts = model
    st.session_state.embedding = embedding
    print("\n\n\t\tNOTIFICATION: TTS Loaded.\n\n")

# D. Chat Storage
if "messages" not in st.session_state.keys():
  st.session_state.messages = [{"role" : "assistant", "content" : "How may I assist you today?"}]


#### #### #### User Input (Voice Recording + Transcription) #### #### ####

# Input
prompt = None
if not os.path.exists("audio.mp3"):
    # 1. Record
    audio = audiorecorder("Start Recording", "Stop Recording")
    if len(audio) > 0:
      # 2. Save
      audio.export("audio.mp3", format="mp3")
      # 3. Transcribe
      prompt = transcribe(st.session_state.stt)


#### #### #### Display Entire Chat #### #### ####
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Add & Display Prompt
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


#### #### #### Generated Output (LLM Text + Speech Generation) #### #### ####
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Generation          
            try:
              # 1. Text Generation
              bot = st.session_state.bot
              response = bot.predict(user_input=prompt)
              # 2. Speech Generation
              speech = speak(st.session_state.tts, st.session_state.embedding, response[:600])
              st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
            except:
              response = "Error : Unable to generate response!"

            # Dynamic Answer Display
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

    os.remove("audio.mp3")