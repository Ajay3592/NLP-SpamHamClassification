import joblib
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import re

# Load the model
#with open("model/spam_detection-0.5.pkl", "rb") as f:
#    model = pickle.load(f)

model = load('model/model1.joblib')

#with open("model/vectorizer-0.5.pkl", "rb") as f1:
#    vectors = pickle.load(f1)

vectors = load('model/vectorizer1.joblib')

#print("Model load completed")

classes = {0: "Not Spam", 1: "Spam"}


def predict_pipeline(text):
     text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
     text = re.sub(r"[[]]", " ", text)
     text = text.lower()
     text_vector = vectors.transform([text])
     text_vector_dence = text_vector.toarray()
     pred = model.predict(text_vector_dence)
     return classes[pred[0]]

app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    EmailDetectedAs: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
     language = predict_pipeline(payload.text)
     return {"EmailDetectedAs": language}