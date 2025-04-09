import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split

def evaluate_model(data_path: str, model_path: str = "model.pkl"):
    X, y, _ = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print("Model Evaluation Metrics:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    evaluate_model("road_accident_data.csv")
