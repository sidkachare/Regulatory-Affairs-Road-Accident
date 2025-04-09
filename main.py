from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.inference import predict

def main():
    print("Regulatory Affairs Road Accident Analysis")
    train_model("road_accident_data.csv")
    evaluate_model("road_accident_data.csv")
    test_prediction = predict("Delhi", "Road Features", "Others", "Persons Killed")
    print(f"Test Prediction: {test_prediction:.2f}")

if __name__ == "__main__":
    main()
