from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
# import soundfile as sf
# file_path = "E:/Data_SLU_journal/Audio_synthetic_FreeVC/vi_freevc/0.wav"
def infer_PhoWhisper(file_path):
    waveform, sampling_rate = librosa.load(file_path, sr=16000)
    # librosa.load(file_path, sr=None)

    # Load the Whisper model in Hugging Face format:
    processor = WhisperProcessor.from_pretrained("nndang/PhoWhisper_streamlit_demo")
    model = WhisperForConditionalGeneration.from_pretrained("nndang/PhoWhisper_streamlit_demo")

    # Use the model and processor to transcribe the audio:
    input_features = processor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    # Generate token ids
    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]
# print(transcription[0])