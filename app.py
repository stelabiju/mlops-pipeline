from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Point directly to the model folder inside mlruns
# Replace <experiment_id> and <run_id> with your actual directory names inside mlruns/
MODEL_URI = "./mlruns/1/models/m-4e19b644d6264788ab257c4cd7500170/artifacts"
model = mlflow.sklearn.load_model(MODEL_URI)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisInput):
    data_df = pd.DataFrame([features.dict().values()], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])
    prediction = model.predict(data_df)
    return {"class_id": int(prediction[0])}