from typing import Union
from fastapi import FastAPI
from joblib import load
import pandas as pd
from starlette.responses import JSONResponse

app = FastAPI()

nn_model = load("/code/models/nn_pipe.joblib")

@app.get("/")
def read_root():
    return {"This project aims to accurately predict the type of beer based on different inputs. "
    "Endpoints: /health, /beers/type, /model/architecture. "
    "Github link: https://github.com/FedGlz/adv_dsi_lab_5 "}

@app.get("/health", status_code=200)
def healthcheck():
    return {"Welcome to the Project"}

@app.get("/model/architecture")
def healthcheck():
    return {"the neural network model consists of x variables, 60 neurons and 105 outputs"}

def format_features(brewery_name:int, review_aroma:float, review_appearance:float, review_palate:float, review_taste:float):
    return{
        'Brewery Name': [brewery_name],
        'Review Aroma': [review_aroma],  
        'Review Appearance': [review_appearance], 
        'Review Palate': [review_palate], 
        'Review Taste': [review_taste]
    }

@app.get('/beers/type')
def predict (brewery_name:int, review_aroma:float, review_appearance:float, review_palate:float, review_taste:float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    pred = nn_model.predict(obs)
    return JSONResponse(pred.tolist())
    

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}