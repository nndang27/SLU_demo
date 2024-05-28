import streamlit as st
from whisper_stt import whisper_stt

text = whisper_stt()  
# If you don't pass an API key, the function will attempt to retrieve it as an environment variable : 'OPENAI_API_KEY'.
if text:
    st.write(text)