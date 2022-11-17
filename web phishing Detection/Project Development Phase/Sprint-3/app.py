from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from inputScript import FeatureExtraction

app = Flask(__name__, template_folder='templates')

app = Flask(__name__)
model = pickle.load(open('Phishing_Website.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def predict():
    return render_template('output.html')


@app.route('/y_predict', methods=['POST'])
def y_predict():
    url = request.form['URL']
    url1 = FeatureExtraction(url)
    x = np.array(url1.getFeaturesList()).reshape(1, 30)

    prediction = model.predict(x)[0]

    print(prediction)
    if (prediction == 1):
        return "Your are safe!! This is a Legitimate website"
    else:
        return "You are on the wrong site Be cautious!"


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.ger_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run()
