# Regulatory Affairs Road Accident Prediction

This project predicts the number of road accident incidents in Indian Million-Plus Cities based on historical data. It includes a complete data science pipeline — from data preprocessing and model training to evaluation and deployment via FastAPI.

## Dataset
- Features:
  - `Million Plus Cities`: Name of the city
  - `Cause category`: Broad cause category (e.g., Road Features, Traffic Control)
  - `Cause Subcategory`: Specific cause
  - `Outcome of Incident`: Type of outcome (Minor Injury, Killed, etc.)
  - `Count`: Number of incidents (target variable)

## Project Structure
## Regulatory Affairs Road Accident
    --notebook/
        --EDA.ipynb

    --src/
        --data_preprocessing.py
        --model_training.py
        --model_evaluation.py
        --inference.py

    --app.py

    --encoders.pkl

    --main.py

    --model.pkl

    --requirements.txt

## Install Dependencies
    pip install -r requirements.txt

## Run the pipeline
    python main.py

## Run the FastAPI Server
    uvicorn app:app --reload
