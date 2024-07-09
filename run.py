import streamlit as st
from whisper_stt import whisper_stt
from streamlit_option_menu import option_menu

with st.sidebar:
    ASR_options = st.selectbox(
   "Select ASR models",
   ("PhoWhisper(small)", "PhoWhisper(base)", "Whisper", "Wav2Vec"),
    )

    NLU_options = st.selectbox(
   "Select NLU models",
   ("JointIDSF_PhoBert", "JointIDSF_XLMR", "JointBERT_PhoBert", "JointBERT_XLMR", "DIET_PhoBert", "DIET_XLMR"),
    )
text = whisper_stt(ASR_name=ASR_options, NLU_name=NLU_options)  
# If you don't pass an API key, the function will attempt to retrieve it as an environment variable : 'OPENAI_API_KEY'.\
print("texttt: ", text)
if text:
    # text = "Đây là một đoạn văn bản cần in ra với cỡ chữ tùy chỉnh."
    text = text.replace('"', '')
    text = text.replace("\\", '')
    intent = text.split('==>')[0]
    intent = intent.replace("<", "&lt;").replace(">", "&gt;")
    sentence_annotation = text.split('==>')[1]
    # html_content = f"""
    # <div style="font-size:15px; color:black; width:1200px; text-align:center;">
    #     {text}
    # </div>
    # """
    html_content = f"""
        <div style="font-size:15px; color:black; width:800px; text-align:center;">
            <table style="width:100%; border-collapse: collapse; margin: 0 auto;">
                <tr>
                    <th style="border: 1px solid black; padding: 8px;">Intent</th>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">{intent}</td>
                </tr>
                <tr>
                    <th style="border: 1px solid black; padding: 8px;">Annotation</th>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">{sentence_annotation}</td>
                </tr>
            </table>
        </div>
        """
    # Display the HTML content in Streamlit
    st.markdown(html_content, unsafe_allow_html=True)
    # st.write(text)