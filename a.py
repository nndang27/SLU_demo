import numpy as np #  A library for numerical computing in Python.
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # ASGI server to run the FastAPI application.
import uvicorn
from io import BytesIO # A class for working with binary data in memory.
from PIL import Image # A library for image processing.
from typing import Tuple # A library for type hints.
import tensorflow as tf # A library for machine learning.
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from infer_Whisper import infer_PhoWhisper
from NLU.predict import predict_nlu
from pyngrok import ngrok, conf
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import os
app = FastAPI() # Create the FastAPI app

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def main():
    return {'message': 'Welcome to GeeksforGeeks!'}

@app.post("/asr/phowhisper/small") # A decorator to create a route for the predict endpoint
async def predict(request: Request): # The function that will be executed when the endpoint is called
    data: bytes = await request.body()

    try: # A try block to handle any errors that may occur
        # audio_bytes = await audio_bio.read()
        transcript = infer_PhoWhisper(BytesIO(data))
 
        return transcript
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e)) # Raise an HTTPException with the error message

@app.post("/nlu/jointidsf/phobert") # A decorator to create a route for the predict endpoint
async def predict(transcript: str): # The function that will be executed when the endpoint is called
    # data_bytes = await request.body()
    # data = data_bytes.decode('utf-8')
    try: # A try block to handle any errors that may occur
        # audio_bytes = await audio_bio.read()
        slot_path = "./NLU/slot_label.txt"
        intent_path = "./NLU/intent_label.txt"
        model_dir = "nndang/NLU_demo_checkpoint"
        filename = "training_args.bin"
        file_path = hf_hub_download(repo_id=model_dir, filename=filename)
        transcript = predict_nlu(transcript, file_path, model_dir, slot_path, intent_path)
        return str(transcript)
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e)) # Raise an HTTPException with the error message
    
if __name__ == "__main__": # If the script is run directly
    conf.get_default().config_path = "E:/Data_SLU_journal/STREAMLIT_SLU/streamlit_app/ngrok.yml"

    ngrok.set_auth_token("2hBxCsADND4DRjAU8aqXFxIgLef_2i6DAieR4TGSBTjXqhuVZ")
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

    # uvicorn.run(app, host="localhost", port=8002) 