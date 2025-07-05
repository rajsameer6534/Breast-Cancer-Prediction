from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and preprocessor
try:
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model/preprocessor: {e}")
    model = None
    preprocessor = None

# Hardcoded hospitals data
HOSPITALS = [
    {
        "name": "Apollo Hospitals, Delhi",
        "contact": "+91 11 7179 0000",
        "estimate": "₹50,000–₹2,50,000",
        "link": "https://goo.gl/maps/FjZKj"
    },
    {
        "name": "Tata Memorial Hospital, Mumbai",
        "contact": "+91 22 2417 7000",
        "estimate": "₹30,000–₹2,00,000",
        "link": "https://goo.gl/maps/zVN2W"
    }
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET'])
def predictdata():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            if model is None or preprocessor is None:
                return "Error: Model not loaded."

            data = request.form.to_dict()
            df = pd.DataFrame([data])
            df = df.apply(pd.to_numeric, errors='coerce')

            if df.isnull().any().any():
                nan_columns = df.columns[df.isnull().any()].tolist()
                return f"Error: Invalid input values in columns: {nan_columns}"

            if 'id' in df.columns:
                df.drop(columns=['id'], inplace=True)

            transformed = preprocessor.transform(df)
            pred = model.predict(transformed)[0]
            proba = model.predict_proba(transformed)[0][1]  # Probability of Malignant (class 1)
            percent_chance = round(proba * 100, 2)

            diagnosis = 'Malignant' if pred == 1 else 'Benign'

            suggestions = (
                ["Maintain a healthy lifestyle",
                 "Schedule regular follow-ups",
                 "Continue routine screenings"]
                if diagnosis == 'Benign' else
                ["Consult an oncologist immediately",
                 "Schedule a biopsy",
                 "Start early treatment options",
                 "Get a second opinion"]
            )

            return render_template('report.html',
                                   data=data,
                                   result=diagnosis,
                                   cancer_percent=percent_chance,
                                   suggestions=suggestions,
                                   hospitals=HOSPITALS)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during prediction: {str(e)}"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
