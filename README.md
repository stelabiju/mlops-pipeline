# End-to-End MLOps Pipeline — Iris Classification

An end-to-end MLOps project for Iris classification using **Random Forest, MLflow, FastAPI, Docker, GitHub Actions, Jenkins, Kubernetes, Argo CD, and Evidently**.

The project demonstrates the machine learning lifecycle from model training and experiment tracking to model serving, containerization, CI/CD, deployment, data drift detection, and automated retraining.

---

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
GitHub Actions
     ↓
Docker Hub
     ↓
EC2 Deployment
     ↓
Data Drift Monitoring
     ↓
Automated Retraining
     ↓
MLflow Model Registry
```

The project also contains a separate Jenkins + Kubernetes + Argo CD GitOps deployment setup.

---

## 🛠️ Technologies Used

| Technology     | Purpose                                |
| -------------- | -------------------------------------- |
| Python         | Application and ML development         |
| Scikit-learn   | Iris dataset and Random Forest model   |
| MLflow         | Experiment tracking and model registry |
| FastAPI        | Model serving API                      |
| Docker         | Application containerization           |
| GitHub Actions | Automated CI/CD                        |
| Docker Hub     | Container image registry               |
| AWS EC2        | Remote Docker deployment               |
| Jenkins        | CI/CD and GitOps pipeline              |
| SonarQube      | Code quality analysis                  |
| Kubernetes     | Application deployment                 |
| Minikube       | Local Kubernetes cluster               |
| Argo CD        | GitOps continuous deployment           |
| Evidently      | Data drift report generation           |
| SciPy          | Statistical drift detection            |
| GitHub         | Source code and deployment management  |

---

## 📁 Project Structure

```text
mlops-pipeline/
│
├── .github/
│   └── workflows/
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
├── tests/
│   ├── test_app.py
│   └── test_retrain.py
│
├── app.py
├── train.py
├── Dockerfile
├── JenkinsFile
├── requirements.txt
├── mlflow.db
└── drift_report.html
```

The repository contains the FastAPI application, MLflow tracking data, monitoring and retraining scripts, tests, Docker configuration, CI/CD workflows, and Kubernetes manifests.

---

# 1. Model Training

The initial model is trained using the Iris dataset and a **Random Forest Classifier**.

The training script:

* Loads the Iris dataset using Scikit-learn
* Splits the data into training and testing sets
* Trains a Random Forest model
* Calculates accuracy
* Logs parameters and metrics to MLflow
* Registers the trained model with MLflow

### Training configuration

```text
n_estimators = 50
max_depth = 3
random_state = 42
```

The experiment is stored under:

```text
iris_classification
```

### Run training

```bash
python train.py
```

---

# 2. MLflow Experiment Tracking

MLflow is used to track model training experiments and manage model versions.

The pipeline records:

* Model parameters
* Accuracy
* Training runs
* Model artifacts
* Registered model versions

Start the MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

The project also contains MLflow tracking data in:

```text
mlruns/
```

and the MLflow database:

```text
mlflow.db
```

---

# 3. FastAPI Model Serving

The trained model is served through a **FastAPI** application.

The API provides prediction and health-check endpoints.

### Run the API locally

```bash
uvicorn app:app --reload
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Health endpoint

```text
GET /health
```

### Prediction endpoint

```text
POST /predict
```

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": 0
}
```

---

# 4. Dockerization

The FastAPI application is containerized using Docker.

### Build the image

```bash
docker build -t stelabiju/iris-mlops-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name iris-api-test stelabiju/iris-mlops-api
```

The application is then available at:

```text
http://localhost:8000
```

The Docker image contains the FastAPI application together with the MLflow database and model tracking artifacts required by the application.

---

# 5. Testing

The project contains automated tests under:

```text
tests/
```

Run the tests locally:

```bash
python -m pytest tests/
```

The test suite covers the FastAPI application and retraining functionality.

---

# 6. CI/CD with GitHub Actions

GitHub Actions is used to automate testing, Docker image creation, publishing, and deployment.

The workflow is located under:

```text
.github/workflows/deploy.yml
```

### GitHub Actions workflow

```text
GitHub Push
     ↓
Set up Python
     ↓
Install Dependencies
     ↓
Run Tests
     ↓
Docker Login
     ↓
Build Docker Image
     ↓
Push Image to Docker Hub
     ↓
SSH into AWS EC2
     ↓
Pull Docker Image
     ↓
Stop Existing Container
     ↓
Remove Existing Container
     ↓
Run New Container
```

The workflow runs automatically when changes are pushed to the `main` branch.

---

## 7. Docker Hub

After successful testing, GitHub Actions builds the Docker image and pushes it to Docker Hub.

Image:

```text
stelabiju/iris-mlops-api:latest
```

The Docker Hub credentials are stored securely in GitHub Actions repository secrets.

Required secrets:

```text
DOCKER_USERNAME
DOCKER_PASSWORD
```

`DOCKER_PASSWORD` contains the Docker Hub Personal Access Token.

Credentials are not stored directly in the workflow file.

---

# 8. AWS EC2 Deployment

The GitHub Actions workflow deploys the Docker container to a remote **AWS EC2 Linux instance** using SSH.

The EC2 instance requires:

* Running EC2 instance
* Public IPv4 address
* SSH access on port 22
* Docker installed and running
* User configured to run Docker commands

The workflow connects to the EC2 instance and executes:

```bash
docker pull stelabiju/iris-mlops-api:latest
docker stop iris-api || true
docker rm iris-api || true
docker run -d -p 8000:8000 --name iris-api stelabiju/iris-mlops-api:latest
```

### GitHub Actions EC2 secrets

```text
VM_HOST
VM_USER
VM_SSH_KEY
```

Where:

* `VM_HOST` = EC2 public IPv4 address
* `VM_USER` = Linux SSH username
* `VM_SSH_KEY` = private SSH key used to connect to EC2

The private key is stored securely as a GitHub Actions secret.

### Deployment flow

```text
GitHub
   ↓
GitHub Actions
   ↓
Docker Hub
   ↓
SSH
   ↓
AWS EC2
   ↓
Docker Container
   ↓
FastAPI
```

---

# 9. CI/CD with Jenkins

The project also contains a Jenkins-based CI/CD and GitOps deployment setup.

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

The Jenkins pipeline uses:

```text
stelabiju/iris-mlops-api:<BUILD_NUMBER>
```

as the Docker image tag.

After a successful Docker build, Jenkins pushes the image to Docker Hub.

Jenkins then updates:

```text
k8s/deployment.yaml
```

with the new image tag and pushes the updated Kubernetes manifest to GitHub.

---

# 10. GitOps Deployment with Argo CD

Argo CD monitors the Kubernetes configuration stored in GitHub.

The Jenkins-based GitOps deployment flow is:

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
Kubernetes / Minikube
```

When Jenkins updates the image tag in:

```text
k8s/deployment.yaml
```

Argo CD detects the Git change and synchronizes the Kubernetes deployment.

This separates:

* **CI** — building and testing the application
* **CD/GitOps** — deploying the desired Kubernetes state from Git

---

# 11. Kubernetes Deployment

The application can be deployed to Kubernetes using:

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

The Kubernetes setup uses a local **Minikube** cluster.

---

# 12. Data Drift Monitoring

The project includes a data drift monitoring component using:

* Evidently
* SciPy Kolmogorov-Smirnov test
* Scikit-learn Iris dataset

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

Dataset-level drift is declared when at least **50% of the feature columns** are detected as drifted.

### Run drift monitoring

```bash
python src/monitor_drift.py
```

This generates:

```text
drift_report.html
```

---

# 13. Automated Retraining

When data drift is detected, the retraining pipeline trains a new Random Forest model.

The script is:

```text
src/retrain_pipeline.py
```

### Workflow

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

* Checks for data drift
* Loads the Iris dataset
* Splits the data
* Trains a Random Forest classifier
* Calculates accuracy
* Logs the drift trigger to MLflow
* Logs the accuracy
* Registers the new model version

### Retraining configuration

```text
n_estimators = 100
max_depth = 5
random_state = 42
```

### Run retraining

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

---

# 14. Dependencies

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
pytest
```

Install them with:

```bash
pip install -r requirements.txt
```

---

# 🔄 Complete MLOps Workflow

The current automated deployment workflow is:

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
                 │ GitHub Actions  │
                 │ CI/CD Pipeline  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │    Docker Hub   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │    AWS EC2      │
                 │ Docker Deploy   │
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
                 Continue    Retrain
                                ↓
                         MLflow Registry
```

The repository also contains a separate Jenkins + Kubernetes + Argo CD GitOps implementation:

```text
Jenkins
   ↓
Docker Hub
   ↓
GitHub
   ↓
Argo CD
   ↓
Kubernetes / Minikube
```

---

# 🎯 Key MLOps Concepts Demonstrated

* **Model Training** — Scikit-learn Random Forest
* **Experiment Tracking** — MLflow
* **Model Versioning** — MLflow Model Registry
* **Model Serving** — FastAPI
* **Containerization** — Docker
* **Continuous Integration** — GitHub Actions / Jenkins
* **Code Quality** — SonarQube
* **Container Registry** — Docker Hub
* **Cloud Deployment** — AWS EC2
* **Kubernetes Deployment** — Minikube
* **GitOps** — Argo CD
* **Data Drift Detection** — KS test + Evidently
* **Continuous Training** — Automated retraining after drift detection

---

# 📌 Project Status

### Completed

* Iris Random Forest model training
* MLflow experiment tracking
* MLflow model registration
* FastAPI prediction API
* Automated API tests
* Docker containerization
* GitHub Actions CI/CD
* Docker Hub image publishing
* AWS EC2 Docker deployment
* Jenkins CI/CD pipeline
* SonarQube analysis
* Kubernetes deployment
* Argo CD GitOps synchronization
* Data drift detection
* Evidently HTML drift report
* Automated retraining pipeline

---

## 👩‍💻 Author

**Stella Biju**

GitHub: [@stelabiju](https://github.com/stelabiju)

---

## 📄 License

This project is intended as an MLOps learning and demonstration project.
