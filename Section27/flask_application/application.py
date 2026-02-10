import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scaler pickle
ridge_model=ridge_regressor=pickle.load(open("../models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("../models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        Humidity=float(request.form.get("RH"))
        wind_speed=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        class_val=int(request.form.get("Classes"))
        Region=int(request.form.get("Region"))

        new_data_scaled=standard_scaler.transform([[Temperature,Humidity,wind_speed,Rain,FFMC,DMC,ISI,class_val,Region]])
        prediction=ridge_model.predict(new_data_scaled)

        return render_template("home.html",prediction_text=f"Predicted Value: {prediction[0]:.2f}")
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)