from typing import Optional

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from utils import transform_dict_to_pandas
from model_tree import DecisionTreeModel

app = FastAPI()

class Music(BaseModel):
    loudness: float
    key: int
    explicit: int
    acousticness: float
    danceability: float
    energy:float
    valence:float
    name: str
    popularity: Optional[float]
    mode: Optional[int]


@app.get("/")
async def hello_word():
    return {"message": "Hello Galera"}


@app.post("/predict/")
async def predict_pipe(music:Music):
    music_dict = music.dict()
    df = transform_dict_to_pandas(music_dict, ['acousticness',	'danceability',	'energy',	'valence', 'key',	'loudness',	'explicit'])
    dt = DecisionTreeModel()
    predict_value = dt.predict(df)[0]
    response_body = {}
    response_body["name"] = music.name
    response_body["predict"] = predict_value
    response_body["received_values"] = music_dict
    return response_body

