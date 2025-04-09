import joblib
import numpy as np

def predict(city, cause, subcause, outcome):
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")

    city_enc = encoders["city"].transform([city])[0]
    cause_enc = encoders["cause"].transform([cause])[0]
    subcause_enc = encoders["subcause"].transform([subcause])[0]
    outcome_enc = encoders["outcome"].transform([outcome])[0]

    input_data = np.array([[city_enc, cause_enc, subcause_enc, outcome_enc]])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    result = predict("Agra", "Traffic Control", "Flashing Signal/Blinker", "Minor Injury")
    print(f"Predicted Incident Count: {result:.2f}")
