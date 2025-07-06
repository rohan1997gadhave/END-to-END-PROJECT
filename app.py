from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model and scaler
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')  # ✅ Updated from 'index.html' to 'home.html'

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        input_array = np.array(list(data.values())).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        output = regmodel.predict(scaled_input)
        return jsonify({'prediction': output[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]
        return render_template("home.html", prediction_text=f"The predicted value is: {output}")  # ✅ Updated here too
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {e}")  # ✅ Also updated here

if __name__ == "__main__":
    app.run(debug=True)
