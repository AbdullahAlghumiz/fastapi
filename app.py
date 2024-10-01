# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the pre-trained model
model = joblib.load("/Users/abdullahg/Downloads/Tutorial/trained_pipeline.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Extract data into the right format
    features = [[data.feature1, data.feature2, data.feature3, data.feature4]]
    
    # Make prediction using the model
    prediction = model.predict(features)
    
    return {"prediction": int(prediction[0])}
