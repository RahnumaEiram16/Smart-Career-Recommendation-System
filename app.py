from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    python = int(request.form['python'])
    java = int(request.form['java'])
    web = int(request.form['web'])
    ml = int(request.form['ml'])
    communication = int(request.form['communication'])
    problem = int(request.form['problem'])
    interest = request.form['interest']

    interest = encoder.transform([interest])[0]

    prediction = model.predict([[python, java, web, ml, communication, problem, interest]])

    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)