from fastapi import FastAPI
from starlette.responses import JSONresponse
from joblib import load
import pandas as pd

app = FastAPI()

nn_pipe = load('../models/nn_pipe.joblib')

@app.get("/")
def read_root():
    return("Hello")

@app.get('/health', status_code=200)
def healthcheck():
    return("Welcome to the project")

def format_features(brewery_name:str, review_aroma:float, review_appearance:float, review_palate:float, review_taste:float):
    return{
        'Brewery Name': [brewery_name],
        'Review Aroma': [review_aroma],  
        'Review Appearance': [review_appearance], 
        'Review Palate': [review_palate], 
        'Review Taste': [review_taste]
    }

@app.get('/beers/type')
def predict (brewery_name:str, review_aroma:float, review_appearance:float, review_palate:float, review_taste:float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    pred = nn_pipe.precit(obs)
    return JSONresponse(pred.tolist())


