from flask import Flask, request, jsonify
import joblib
from flask import Flask, send_file
import pandas as pd

app = Flask(__name__)

## CHARGEMENT DU MODÈLE
model = joblib.load("model.pkl")

## TEST API
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Bienvenue sur l'API ML"
    })


##  PRÉDICTION
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    return jsonify({
        "received_data": data
    })

##  INFO MODÈLE
@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify({
        "model_type": str(type(model).__name__),
        "features": [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
    })
## Route pour télécharger le dataset nettoyé
@app.route("/table")
def afficher_table():
    df = pd.read_csv("dataset_nettoye.csv")
    return df.to_html()

## Route pour télécharger le dataset nettoyé
@app.route("/data")
def download_data():
    return send_file("dataset_nettoye.csv", as_attachment=True) 


if __name__ == "__main__":
    app.run(debug=True)