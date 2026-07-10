# 🌍 Land Type Classification using Satellite Images

A complete Deep Learning pipeline for **land cover classification** using the **EuroSAT RGB dataset** based on Sentinel-2 satellite imagery.

The project covers the entire machine learning lifecycle, including data preprocessing, model training, evaluation, deployment with FastAPI, an interactive Streamlit interface, and Docker containerization.

The API is deployed live on **Railway**, and the Streamlit interface is deployed on **Streamlit Community Cloud**.


# 📌 Project Overview

This project aims to automatically classify satellite images into different land cover categories using Convolutional Neural Networks (CNNs) and Transfer Learning.

The final deployed model is **EfficientNetB3**, which achieved the best overall performance on the validation and test datasets. The trained model is publicly available on Hugging Face Hub.


# 🚀 Features

- End-to-end Deep Learning pipeline
- Image preprocessing using OpenCV
- Transfer Learning & Fine-Tuning
- Model evaluation with multiple metrics
- FastAPI REST API
- Streamlit Web Application
- Dockerized deployment
- Configurable through `.env`
- Trained model hosted on Hugging Face Hub
- Deployed on Railway (API) and Streamlit Community Cloud (UI)


# 🌐 Live Demo

| Service | Platform | Link |
|---------|----------|------|
| REST API | Railway | [https://dp-production-2c54.up.railway.app/docs](https://dp-production-2c54.up.railway.app/docs) |
| Web Interface | Streamlit Community Cloud | [https://hd385cjd4pb6pcczhc9zb4.streamlit.app/](https://hd385cjd4pb6pcczhc9zb4.streamlit.app/) |
| Trained Model | Hugging Face Hub | [https://huggingface.co/Mowael1/efficientnetb3-eurosat](https://huggingface.co/Mowael1/efficientnetb3-eurosat) |


# 🧠 Trained Model

The final trained **EfficientNetB3** model is hosted on Hugging Face Hub and can be downloaded or loaded directly from there:

```
https://huggingface.co/Mowael1/efficientnetb3-eurosat
```

This is the same model used in production by the deployed API on Railway.


# 🛰 Dataset

**EuroSAT RGB Dataset**

Source:

https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

The dataset consists of Sentinel-2 satellite images covering various land use and land cover classes across Europe.

### Number of Classes

- 🌾 Annual Crop
- 🌲 Forest
- 🌱 Herbaceous Vegetation
- 🛣 Highway
- 🏭 Industrial
- 🌄 Pasture
- 🌳 Permanent Crop
- 🏠 Residential
- 🌊 River
- 🟦 Sea / Lake

### Image Information

- Image Size: 64 × 64 RGB
- Total Images: 27,000
- Classes: 10
- Balanced Dataset


# 📊 Dataset Split

| Dataset | Percentage |
|----------|------------|
| Train | 70% |
| Validation | 20% |
| Test | 10% |

The split is performed automatically before training.


# 🧹 Data Preprocessing

The preprocessing pipeline is implemented inside:

```
src/data/transforms.py
```

### Image Enhancement

- Median Blur
- CLAHE (Contrast Limited Adaptive Histogram Equalization)


### Image Processing

- Resize
- Normalization (0-1)

### Data Augmentation (Training Only)

- Random Horizontal Flip
- Random Brightness
- Random Contrast

Validation and Test images only receive preprocessing without augmentation.



# ⚙️ Training

Training pipeline includes:

- TensorFlow / Keras
- Adam Optimizer
- Early Stopping
- ReduceLROnPlateau
- Fine-Tuning
- Model Checkpointing

Training parameters are managed through:

```
src/.env
```

Example:

```env
IMAGE_SIZE=224
BATCH_SIZE=32
LEARNING_RATE=0.001
```


# 📈 Evaluation

The trained models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

Training curves include:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

---

# 🏗 Project Structure

```text
eurosat-land-classification/

├── frontend/
│   └── app.py
│
├── src/
│   ├── api/
│   ├── constants/
│   ├── data/
│   ├── helpers/
│   ├── models/
│   ├── prediction/
│   ├── training/
│   ├── trained_models/
│   ├── .env
│   └── main.py
│
├── notebooks/
│
├── reports/
│
├── dataset/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
```


# 🌐 REST API

The project exposes a REST API using FastAPI.

### Run API Locally

```bash
python -m src.main
```

Interactive documentation:

```
http://localhost:8000/docs
```

Prediction endpoint:

```
POST /predict
```

Input:

```
Multipart Form Data

image: satellite_image.jpg
```

Example Response

```json
{
    "predicted_class": "Forest",
    "confidence": 0.9823
}
```


# 🖥 Streamlit Interface

A lightweight web interface is included for testing the model.

### Run Locally

```bash
streamlit run frontend/app.py
```

Features:

- Upload image
- Display uploaded image
- Call FastAPI
- Show predicted class
- Display confidence score

The hosted version on Streamlit Community Cloud calls the live Railway API instead of a local backend, so the app works out of the box without any local setup.


# 🐳 Docker

The project can be deployed inside a Docker container.

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

The API will be available at

```
http://localhost:8000
```


# ☁️ Deployment

### API — Railway

The FastAPI backend is containerized and deployed on **Railway** directly from the project's `Dockerfile`.

Deployment steps:

1. Push the repository to GitHub.
2. Create a new Railway project and link it to the GitHub repo.
3. Railway detects the `Dockerfile` and builds the image automatically.
4. Set the required environment variables (from `.env`) in the Railway project settings.
5. Railway assigns a public domain to the service, exposing the FastAPI app on port `8000`.

Once deployed, the interactive API docs are available at:

```
https://dp-production-2c54.up.railway.app/docs
```

### Web Interface — Streamlit Community Cloud

The Streamlit frontend (`frontend/app.py`) is deployed separately on **Streamlit Community Cloud**.

Deployment steps:

1. Push the repository to GitHub (same repo or a dedicated frontend repo).
2. Create a new app on [Streamlit Community Cloud](https://streamlit.io/cloud), pointing to `frontend/app.py`.
3. Add any required secrets (such as the Railway API URL) via the app's **Secrets** settings.
4. Streamlit Community Cloud builds and hosts the app automatically on every push.

The deployed Streamlit app sends prediction requests to the live Railway API endpoint instead of `localhost`.

### Model — Hugging Face Hub

The trained **EfficientNetB3** model weights are hosted on Hugging Face Hub:

```
https://huggingface.co/Mowael1/efficientnetb3-eurosat
```

The deployed API on Railway loads the model from Hugging Face Hub instead of bundling it inside the Docker image, keeping the image lightweight.


# ⚙️ Configuration

All project settings are managed through environment variables.

Example:

```env
MODEL_NAME=efficientnetb3
MODEL_SAVE_DIR=./src/trained_models

IMAGE_SIZE=224
BATCH_SIZE=32
LEARNING_RATE=0.001

API_HOST=0.0.0.0
API_PORT=8000
```


# 📷 Sample Prediction

| Input Image | Prediction |
|-------------|------------|
| Satellite Image | Forest |
| Satellite Image | River |
| Satellite Image | Residential |



# 📚 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- FastAPI
- Streamlit
- Docker
- Uvicorn
- Pydantic
- Railway
- Streamlit Community Cloud
- Hugging Face Hub


# 📄 License

This project is licensed under the MIT License.