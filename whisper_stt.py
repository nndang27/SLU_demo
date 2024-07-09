# whisper_stt.py

from streamlit_mic_recorder import mic_recorder
import streamlit as st
import io
# from openai import OpenAI
import hydralit_components as hc
import os

import requests

API_asr_dict = {"PhoWhisper(small)": "asr/phowhisper/small",
            "PhoWhisper(base)": "asr/phowhisper/base",
            "Whisper": "asr/whisper/small",
            "Wav2Vec": "asr/wav2vec"}

API_nlu_dict = {"JointIDSF_PhoBert": "nlu/jointidsf/phobert",
            "JointIDSF_XLMR": "nlu/jointidsf/xlm",
            "JointBERT_PhoBert": "nlu/jointbert/phobert",
            "JointBERT_XLMR": "nlu/jointbert/xlm",
            "DIET_PhoBert": "nlu/diet/phobert",
            "DIET_XLMR": "nlu/diet/xlm"}

def whisper_stt(ASR_name, NLU_name, start_prompt="Start recording", stop_prompt="Stop recording", just_once=False,
               use_container_width=False, callback=None, args=(), kwargs=None, key=None):
    # if not 'openai_client' in st.session_state:
    #     st.session_state.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
    print("ASR: ", ASR_name)
    print("NLU: ", NLU_name)
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
            audio_bio = audio['bytes']
            # io.BytesIO(audio['bytes'])
            # audio_bio.name = 'audio.wav'
            success = False
            err = 0

            while not success and err < 3:  # Retry up to 3 times in case of OpenAI server error.
                try: # http://localhost:8002
                    # API PHOWHISPER  
                    with hc.HyLoader('Loading ...',hc.Loaders.standard_loaders, height=10):
                        # time.sleep(5)  http://127.0.0.1:8000
                        res = requests.post(url = f"https://6292-43-239-223-87.ngrok-free.app/{API_asr_dict[ASR_name]}", data = audio_bio)
                    
                        print("TEXT: ",  res.text)
                        transcript = requests.post(url = f"https://6292-43-239-223-87.ngrok-free.app/{API_nlu_dict[NLU_name]}", params ={"transcript": res.text} )
                        print("transcript: ", transcript.text)

                except Exception as e:
                    print(str(e))  # log the exception in the terminal
                    err += 1
                else:
                    success = True
                    output = transcript.text
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