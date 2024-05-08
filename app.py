from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all form data
    age = request.form.get('age', type=float)
    bmi = request.form.get('bmi', type=float)
    children = request.form.get('children', type=int)
    sex_male = 1 if request.form.get('sex_male') == 'on' else 0
    smoker_yes = 1 if request.form.get('smoker_yes') == 'on' else 0
    region_northwest = 1 if request.form.get('region_northwest') == 'on' else 0
    region_southeast = 1 if request.form.get('region_southeast') == 'on' else 0
    region_southwest = 1 if request.form.get('region_southwest') == 'on' else 0

    # Create DataFrame
    features = [age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]
    features_df = pd.DataFrame([features], columns=[
        'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest'])

    # Predict
    prediction = model.predict(features_df)
    return jsonify({'prediction': float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

