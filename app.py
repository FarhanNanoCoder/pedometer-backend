from flask import Flask, jsonify, request
from decimal import Decimal
import joblib
import pandas as pd
import numpy as np
import sklearn

ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/step", methods=["POST"])
def countStep():
    if request.method == "POST":
        if 'data' not in request.json:
            return jsonify({"error": "No data found to process"}, 400)
            
        stepCountModel = joblib.load("stepCounterModel.joblib")
        stepSizeModel = joblib.load("stepSizePredictionModel.joblib")

        stepCountPrediction = stepCountModel.predict(request.json["data"])
        stepSizePrediction = stepSizeModel.predict(
            [[request.json["age"], request.json["height"], request.json["weight"], request.json["gender"]]])
       
        count = np.sum(stepCountPrediction, dtype=np.int64)
        stepSize = '{:.3f}'.format(np.array(stepSizePrediction)[0])
        stepSize =  float(stepSize)
        return jsonify({"code": 200, "count": count.item(), "step_size": stepSize})

    return jsonify({"error": "Request error"}, 400)


@app.route("/step-bulk", methods=["POST"])
def countStepBulk():
    if request.method == "POST":
        if 'data' not in request.files:
            return jsonify({"error": "No file found in request"}, 400)
        file = request.files['data']
        if file.filename == '':
            return jsonify({"error": "No file found in request"}, 400)
        if file and allowed_file(file.filename):
            print(file.filename)
            # model will run here
            temp = pd.read_csv(file)
            test = temp.drop(columns=['timestamp', 'original_steps'])
            model = joblib.load("stepCounterModel.joblib")
            prediction = model.predict(test)
            count = np.sum(prediction, dtype=np.int64)
            # print(count)
            count = np.sum(prediction, dtype=np.int64)
            return jsonify({"code": 200, "count": count.item()})
        return "File type is not allowed"
    return jsonify({"error": "Not a post req"}, 400)


if __name__ == "__main__":
    app.run(debug=True)
