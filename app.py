from fastapi import FastAPI
from transformers import pipeline
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import generator_text
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import mlflow
import time
import os

app = FastAPI()

origins = ["*"]

os.environ['MLFLOW_TRACKING_URI'] = '/root/code/mlflow'
experiment_id = mlflow.set_experiment('STORY')


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class StoryRequest(BaseModel):
    theme: str
    max_length: int
    num_stories: int

class StoryResponse(BaseModel):
    histoire: List[Dict[str, str]]


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse('static/index.html')



@app.post("/api", response_model=StoryResponse)
async def predict(request: StoryRequest):


    with mlflow.start_run():

        mlflow.log_param("theme", request.theme)
        mlflow.log_param("max_length", request.max_length)
        mlflow.log_param("num_stories", request.num_stories)

        start_time = time.perf_counter()
        prediction = generator_text(request.theme, request.max_length, request.num_stories)
        end_time = time.perf_counter()
        generation_time = end_time - start_time
        mlflow.log_metric("generation_time", generation_time)

    mlflow.end_run()

    json_prediction = jsonable_encoder(prediction)
    return {"histoire": json_prediction}
