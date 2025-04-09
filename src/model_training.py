import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_and_preprocess_data

def train_model(data_path: str, model_path: str = "model.pkl"):
    X, y, encoders = load_and_preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Validation MSE: {mse:.2f}")

    joblib.dump(model, model_path)
    joblib.dump(encoders, "encoders.pkl")

if __name__ == "__main__":
    train_model("road_accident_data.csv")
