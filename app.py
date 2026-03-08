from flask import Flask, render_template, request
import pickle
import numpy as np 
import os

# load trained model
modelobject = pickle.load(open("model/model.pkl","rb"))

# create flask app
app = Flask(__name__)

# home page
@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

# history page
@app.route('/history')
def history():
    return render_template("history.html")

# prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # numeric inputs
    area = float(request.form.get('area',0))
    bedrooms = float(request.form.get('bedrooms',0))
    bathrooms = float(request.form.get('bathrooms',0))
    stories = float(request.form.get('stories',0))

    # binary inputs
    mainroad = int(request.form.get('mainroad',0))
    guestroom = int(request.form.get('guestroom',0))
    basement = int(request.form.get('basement',0))
    hotwaterheating = int(request.form.get('hotwaterheating',0))
    airconditioning = int(request.form.get('airconditioning',0))

    parking = float(request.form.get('parking',0))

    prefarea = int(request.form.get('prefarea',0))
    furnishingstatus = int(request.form.get('furnishingstatus',0))

    # correct feature order (VERY IMPORTANT)
    features = np.array([[area, bedrooms, bathrooms, stories,
                          mainroad, guestroom, basement,
                          hotwaterheating, airconditioning,
                          parking, prefarea, furnishingstatus]])

    prediction = modelobject.predict(features)

    return render_template("index.html",
                           prediction_text=f"Predicted House Price: ₹ {round(prediction[0],2)}")

# run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
