# imports
from chat import Bot
import streamlit as st

# Page Title
st.set_page_config(page_title="Tech Tutor LLM")

st.title("FineTuned LLM")
with st.expander("Disclaimer"):
  st.write("Powered by fine-tuned version of Llama 2 - 7B Chat HF. This bot is exclusively for computational questions (PF, OOP, DSA) with limited coding. Only English Language is supported for now. Responses may be delayed due to computing constraints.")

# Load LLM
if "bot" not in st.session_state:
    bot = Bot()
    st.session_state.bot = bot.build_chain(use_api=True)

# Side Bar
with st.sidebar:
  st.markdown("<h1 style='text-align: center;'>🎓Tech Tutor</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center;'>F23-058-D-eLearnPlatform</h3>", unsafe_allow_html=True)
  st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); 
  st.markdown("<h3 style='text-align: center;'>About Us</h3>", unsafe_allow_html=True)
  st.markdown("<p style='text-align: center;'>At the forefront of educational innovation, our project revolutionizes online learning for those passionate about Programming and Computing. Through advanced AI agents, interactive quizzes, and a virtual whiteboard, we provide personalized, real-time support. Our mission is to make the complex world of computing accessible, engaging, and interactive for learners, empowering them to master key concepts with confidence. Welcome to a new era of online education.</p>", unsafe_allow_html=True)
  st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); st.text(""); 

# Store LLM Generated Responses
if "messages" not in st.session_state.keys():
  st.session_state.messages = [{"role" : "assistant",
                                "content" : "How may I assist you today?"}]

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", 
                                "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History and Reset All Settings', on_click=clear_chat_history)

# User-provided prompt
prompt = st.chat_input("Ask Anything!")
if prompt:
    st.session_state.messages.append({"role": "user", 
                                      "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
          
            try:
              bot = st.session_state.bot
              response = bot.predict(user_input=prompt)
            except:
              response = "Error : Unable to generate response!"
            placeholder = st.empty()

            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

    message = {"role": "assistant", 
              "content": full_response}
    st.session_state.messages.append(message)