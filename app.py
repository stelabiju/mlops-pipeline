import os
import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

# Target root directory containing mlflow.db
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

MODEL_NAME = "IrisRandomForestModel"
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

def load_latest_model():
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    client = MlflowClient()
    
    try:
        # 1. Try loading by assigned @champion alias
        model_uri = f"models:/{MODEL_NAME}@champion"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # 2. Fallback: try loading highest numerical version
        try:
            versions = client.get_latest_versions(MODEL_NAME)
            if versions:
                latest_ver = versions[-1].version
                return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest_ver}")
        except Exception as err:
            print(f"Model registry fetch notice: {err}")
            
    return None

# Initial model load
model = load_latest_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/reload")
def reload_model():
    global model
    model = load_latest_model()
    if model is None:
        raise HTTPException(status_code=404, detail="No valid model version found in registry.")
    return {"status": "success", "message": "Model updated to champion version"}

@app.post("/predict")
def predict(data: IrisInput):
    global model
    if model is None:
        model = load_latest_model()
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail=f"Model artifact unavailable at {DB_PATH}. Run src/retrain_pipeline.py first."
            )
            
    features_df = pd.DataFrame(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
        columns=FEATURE_NAMES
    )
    prediction = model.predict(features_df)
    return {"prediction": int(prediction[0])}