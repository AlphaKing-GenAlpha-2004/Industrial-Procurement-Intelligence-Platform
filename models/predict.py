import joblib

def predict(model_path, data):

    print("\nRunning Predictions...\n")

    model = joblib.load(model_path)

    predictions = model.predict(data)

    return predictions