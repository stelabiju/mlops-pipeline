import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.monitor_drift import check_data_drift

# Set path to root directory containing mlflow.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"

def train_and_register_model():
    print("Initiating retraining process...")
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 5
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        print(f"Candidate Model Trained. Accuracy: {acc:.4f}")

        if acc >= 0.90:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            
            # Set alias to 'champion'
            client = MlflowClient()
            model_version = model_info.registered_model_version
            client.set_registered_model_alias(MODEL_NAME, "champion", model_version)
            
            print(f"Model version {model_version} successfully registered with alias 'champion'")
        else:
            print("Candidate model failed accuracy threshold (0.90). Promotion skipped.")

def run_pipeline():
    print("Checking production data for drift...")
    drift_detected = check_data_drift()

    if drift_detected:
        print("Data drift confirmed! Triggering retraining pipeline...")
        train_and_register_model()
    else:
        print("No significant drift detected. Model operating within normal bounds.")

if __name__ == "__main__":
    run_pipeline()