# End-to-End MLOps Pipeline — Iris Classification

An end-to-end MLOps project for Iris classification using **Random Forest**, **MLflow**, **FastAPI**, **Docker**, **Jenkins**, **Kubernetes**, and **Argo CD**.

The project demonstrates the complete machine learning lifecycle — from model training and experiment tracking to containerization, CI/CD, GitOps deployment, data drift detection, and automated retraining.

## 🚀 Project Overview

This project uses the classic **Iris dataset** to demonstrate how a machine learning model can be developed, tracked, deployed, monitored, and retrained using MLOps practices.

### Main workflow

```text
Iris Dataset
     ↓
Model Training
     ↓
MLflow Experiment Tracking
     ↓
MLflow Model Registry
     ↓
FastAPI Prediction API
     ↓
Docker Container
     ↓
Jenkins CI/CD
     ↓
Docker Hub
     ↓
GitHub
     ↓
Argo CD
     ↓
Kubernetes / Minikube
     ↓
Data Drift Monitoring
     ↓
Automated Retraining
     ↓
MLflow Model Registry
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application and ML development |
| Scikit-learn | Iris dataset and Random Forest model |
| MLflow | Experiment tracking and model registry |
| FastAPI | Model serving API |
| Docker | Application containerization |
| Jenkins | CI/CD pipeline |
| SonarQube | Code quality analysis |
| Docker Hub | Container image registry |
| Kubernetes | Application deployment |
| Minikube | Local Kubernetes cluster |
| Argo CD | GitOps continuous deployment |
| Evidently | Data drift report generation |
| SciPy | Statistical drift detection |
| GitHub | Source code and deployment manifest management |

## 📁 Project Structure

```text
mlops-pipeline/
│
├── .github/
│   └── workflow/
│       └── deploy.yml
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── mlruns/
│   └── MLflow tracking data
│
├── src/
│   ├── monitor_drift.py
│   └── retrain_pipeline.py
│
├── app.py
├── train.py
├── Dockerfile
├── JenkinsFile
├── requirements.txt
├── mlflow.db
└── drift_report.html
```

The current repository contains Kubernetes manifests under `k8s/`, the monitoring and retraining scripts under `src/`, and the generated Evidently report at the project root.

## 1. Model Training

The initial model is trained using the Iris dataset and a **Random Forest Classifier**.

The training script:

- Loads the Iris dataset using Scikit-learn
- Splits the data into training and testing sets
- Trains a Random Forest model
- Calculates accuracy
- Logs parameters and metrics to MLflow
- Registers the trained model as `IrisRandomForest`

The current training configuration uses:

```text
n_estimators = 50
max_depth = 3
random_state = 42
```

The experiment is stored under:

```text
iris_classification
```

The model is registered in MLflow as:

```text
IrisRandomForest
```



### Run training

```bash
python train.py
```

## 2. MLflow Experiment Tracking

MLflow is used to track model training experiments and manage model versions.

The pipeline records:

- Model parameters
- Accuracy
- Training runs
- Model artifacts
- Registered model versions

Start the MLflow UI with:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

The project also contains MLflow tracking data in the `mlruns/` directory and an `mlflow.db` database.

## 3. FastAPI Model Serving

The trained model is served through a **FastAPI** application.

The API provides a prediction endpoint for Iris classification.

Run the API locally with:

```bash
uvicorn app:app --reload
```

The interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## 4. Dockerization

The FastAPI application is containerized using Docker.

Build the image:

```bash
docker build -t stelabiju/iris-mlops-api .
```

Run the container:

```bash
docker run -p 8000:8000 stelabiju/iris-mlops-api
```

The application can then be accessed through:

```text
http://localhost:8000
```

## 5. CI/CD with Jenkins

Jenkins automates the application build and deployment workflow.

The Jenkins pipeline performs:

```text
Checkout
   ↓
Build & Test
   ↓
SonarQube Analysis
   ↓
Docker Build
   ↓
Docker Push
   ↓
Update Kubernetes Manifest
   ↓
Git Push
```

The pipeline uses the Docker image:

```text
stelabiju/iris-mlops-api:<BUILD_NUMBER>
```

After a successful Docker build, Jenkins pushes the image to Docker Hub.

Jenkins then updates:

```text
k8s/deployment.yaml
```

with the new Docker image tag and pushes the updated Kubernetes manifest back to GitHub.

## 6. GitOps Deployment with Argo CD

Argo CD monitors the Kubernetes configuration stored in GitHub.

The deployment flow is:

```text
Jenkins
   ↓
Docker Hub
   ↓
Update deployment.yaml
   ↓
GitHub
   ↓
Argo CD
   ↓
Minikube
```

When Jenkins updates the image tag in `k8s/deployment.yaml`, Argo CD detects the Git change and synchronizes the Kubernetes deployment.

This separates:

- **CI** — building and testing the application
- **CD/GitOps** — deploying the desired Kubernetes state from Git

## 7. Kubernetes Deployment

The application is deployed to Kubernetes using:

```text
k8s/deployment.yaml
k8s/service.yaml
```

Check the deployment:

```bash
kubectl get deployments
```

Check the pods:

```bash
kubectl get pods
```

Check the services:

```bash
kubectl get services
```

The application is deployed to a local **Minikube** Kubernetes cluster.

## 8. Data Drift Monitoring

The project includes a data drift monitoring component using:

- Evidently
- SciPy Kolmogorov-Smirnov test
- Scikit-learn Iris dataset

The monitoring script is:

```text
src/monitor_drift.py
```

The script creates:

1. A reference dataset
2. A simulated current production dataset
3. Artificial distribution drift
4. An Evidently HTML report
5. A statistical KS test for each feature

### Drift simulation

The current production data intentionally introduces drift by modifying:

```python
current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5
```

The KS test uses:

```text
p-value < 0.05
```

as the threshold for an individual feature.

Dataset-level drift is declared when **at least 50% of the feature columns are detected as drifted**.

### Run drift monitoring

```bash
python src/monitor_drift.py
```

The script generates:

```text
drift_report.html
```

The generated report is included in the repository.

## 9. Automated Retraining

When data drift is detected, the retraining pipeline automatically trains a new Random Forest model.

The script is:

```text
src/retrain_pipeline.py
```

The workflow is:

```text
Check Drift
    ↓
Drift Detected?
   ↙       ↘
 No        Yes
 ↓          ↓
Stop     Retrain Model
            ↓
       Evaluate Accuracy
            ↓
       Log to MLflow
            ↓
       Register Model
```

The retraining process:

- Checks for data drift
- Loads the Iris dataset
- Splits the data
- Trains a Random Forest classifier
- Calculates accuracy
- Logs the drift trigger to MLflow
- Logs the accuracy
- Registers the new model version as `IrisRandomForest`

The retraining configuration currently uses:

```text
n_estimators = 100
max_depth = 5
random_state = 42
```



### Run the retraining pipeline

```bash
python src/retrain_pipeline.py
```

If drift is detected:

```text
Drift threshold breached!
Triggering automated retraining...
Retraining completed.
Updated version registered to MLflow Model Registry.
```

If no drift is detected:

```text
No significant drift detected.
Retraining skipped.
```

## 10. Dependencies

The project dependencies include:

```text
fastapi
uvicorn
scikit-learn
pandas
mlflow
pydantic
evidently
requests
```



Install them with:

```bash
pip install -r requirements.txt
```

## 🔄 Complete MLOps Workflow

The complete system can be summarized as:

```text
                ┌─────────────────┐
                │   Iris Dataset  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │  Model Training │
                │ Random Forest   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     MLflow      │
                │ Tracking +      │
                │ Model Registry  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     FastAPI     │
                │ Prediction API  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     Docker      │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     Jenkins     │
                │ CI/CD Pipeline  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │    Docker Hub   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     GitHub      │
                │ deployment.yaml │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     Argo CD     │
                │     GitOps      │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │    Kubernetes   │
                │    Minikube     │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │  Drift Monitor  │
                │ KS + Evidently  │
                └────────┬────────┘
                         ↓
                    Drift Found?
                     ↙       ↘
                   No         Yes
                   ↓           ↓
                 Continue   Retrain
                               ↓
                         MLflow Registry
```

## 🎯 Key MLOps Concepts Demonstrated

- **Experiment Tracking** — MLflow
- **Model Versioning** — MLflow Model Registry
- **Model Serving** — FastAPI
- **Containerization** — Docker
- **Continuous Integration** — Jenkins
- **Code Quality** — SonarQube
- **Container Registry** — Docker Hub
- **Kubernetes Deployment** — Minikube
- **GitOps** — Argo CD
- **Data Drift Detection** — KS test + Evidently
- **Continuous Training** — Automated retraining after drift detection

## 📌 Project Status

### Completed

- Iris Random Forest model training
- MLflow experiment tracking
- MLflow model registration
- FastAPI prediction API
- Docker containerization
- Jenkins CI/CD pipeline
- SonarQube analysis
- Docker Hub image publishing
- Kubernetes deployment
- Argo CD GitOps synchronization
- Data drift detection
- Evidently HTML drift report
- Automated retraining pipeline

## 👩‍💻 Author

**Stella Biju**

GitHub: [@stelabiju](https://github.com/stelabiju)

## 📄 License

This project is intended as an MLOps learning and demonstration project.