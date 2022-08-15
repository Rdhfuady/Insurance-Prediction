from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

#initiate model and columns
LABEL = ["Customers Not Claim Insurance", "Customers Claim Insurance"]
with open("final_pipe.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def welcome():
    return "<h3>Selamat Datang di Program Car Claim Insurance Prediction</h3>"

@app.route("/predict", methods=["GET", "POST"])
def predict_insurance():
    if request.method == "POST":
        content = request.json
        try:
            new_data = {"AGE": content["AGE"],
                        "DRIVING_EXPERIENCE": content["DRIVING_EXPERIENCE"],
                        "EDUCATION": content["EDUCATION"],
                        "INCOME": content["INCOME"],
                        "CREDIT_SCORE": content["CREDIT_SCORE"],
                        "VEHICLE_OWNERSHIP": content["VEHICLE_OWNERSHIP"],
                        "VEHICLE_YEAR": content["VEHICLE_YEAR"],
                        "SPEEDING_VIOLATIONS": content["SPEEDING_VIOLATIONS"],
                        "PAST_ACCIDENTS": content["PAST_ACCIDENTS"]}
            new_data = pd.DataFrame([new_data])
            res = model.predict(new_data)
            result = {'class':res[0],
                      'class_name':LABEL[int(res[0])]}
            response = jsonify(success=True,
                               result=result)
            return response, 200
        except Exception as e:
            response = jsonify(success=False,
                               message=str(e))
            return response, 400
    # return dari method get
    return "<p>Silahkan gunakan method POST untuk mode <em>inference model</em></p>"

#app.run(debug=True)