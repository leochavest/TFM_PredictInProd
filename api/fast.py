from tkinter import Y
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(
    pickup_datetime: object,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int,
):

    key = "2013-07-06 17:18:00.000000119"
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    X_pred = pd.DataFrame(
        {
            "key": [key],
            "pickup_datetime": [utc_pickup_datetime],
            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "passenger_count": [passenger_count],
        }
    )
    model = joblib.load("model.joblib")
    y_pred = model.predict(X_pred)

    return {"test":[y_pred[0]]}
