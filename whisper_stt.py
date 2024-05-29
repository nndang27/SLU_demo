# whisper_stt.py

from streamlit_mic_recorder import mic_recorder
import streamlit as st
import io
# from openai import OpenAI
from huggingface_hub import hf_hub_download
import os
from infer_Whisper import infer_PhoWhisper
from NLU.predict import predict_nlu

def whisper_stt( start_prompt="Start recording", stop_prompt="Stop recording", just_once=False,
               use_container_width=False, callback=None, args=(), kwargs=None, key=None):
    # if not 'openai_client' in st.session_state:
    #     st.session_state.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
    if not '_last_speech_to_text_transcript_id' in st.session_state:
        st.session_state._last_speech_to_text_transcript_id = 0
    if not '_last_speech_to_text_transcript' in st.session_state:
        st.session_state._last_speech_to_text_transcript = None
    if key and not key + '_output' in st.session_state:
        st.session_state[key + '_output'] = None
    audio = mic_recorder(start_prompt=start_prompt, stop_prompt=stop_prompt, just_once=just_once,
                         use_container_width=use_container_width,format="wav", key=key)
    new_output = False
    if audio is None:
        output = None
    else:
        id = audio['id']
        new_output = (id > st.session_state._last_speech_to_text_transcript_id)
        if new_output:
            output = None
            st.session_state._last_speech_to_text_transcript_id = id
            audio_bio = io.BytesIO(audio['bytes'])
            audio_bio.name = 'audio.wav'
            success = False
            err = 0

            while not success and err < 3:  # Retry up to 3 times in case of OpenAI server error.
                try:
                    transcript = infer_PhoWhisper(audio_bio)
                    slot_path = "./NLU/slot_label.txt"
                    intent_path = "./NLU/intent_label.txt"
                    # "E:/Data_SLU_journal/NLU_MODEL_V100/NLU_model/JointIDSF/checkpoint_NLU_slotfilling_250/JointIDSF_PhoBERTencoder/4e-5/0.15/100"
                    model_dir = "nndang/NLU_demo_checkpoint"
                    filename = "training_args.bin"
                    file_path = hf_hub_download(repo_id=model_dir, filename=filename)
                    
                    transcript = predict_nlu(transcript,file_path, model_dir, slot_path, intent_path)
                    # st.session_state.openai_client.audio.transcriptions.create(
                    #     model="whisper-1",
                    #     file=audio_bio,
                    #     language=language
                    # )
                except Exception as e:
                    print(str(e))  # log the exception in the terminal
                    err += 1
                else:
                    success = True
                    output = transcript
                    # transcript.text
                    st.session_state._last_speech_to_text_transcript = output
        elif not just_once:
            output = st.session_state._last_speech_to_text_transcript
        else:
            output = None

    if key:
        st.session_state[key + '_output'] = output
    if new_output and callback:
        callback(*args, **(kwargs or {}))
    return output