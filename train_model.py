import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from flask import app, send_file


 ## 1. CHARGEMENT des donnes
df = pd.read_csv("data/train.csv")
print(" Dataset chargé")
print(df.head())

## 2.faire la description du dataset
print(" Description du dataset :")
print(df.describe())

## 3. Nettoyage des données
print("Nettoyage des valeurs manquantes...")

## 3.1 Identifier les valeurs manquantes
missing_rate = df.isnull().sum()
print("Valeurs manquantes :")
print(missing_rate.sort_values(ascending=False))

## 3.2 Traitement des valeurs manquantes
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("Nettoyage terminé ✔️")

## 4. ENCODAGE DES VARIABLES CATÉGORIQUES
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})


## 5.Separer les variables features et target
target = "Survived" 
X = df.drop(target, axis=1)
y = df[target]

## 6.Separation des donnes tain/test_split
print("Separation des donnes tain/test_split")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

## 7. MODÈLE
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

## 8.verification des donnes
print(" vérification ")
print(df.head())

## 9. ENTRAÎNEMENT du modele
model.fit(X_train, y_train)
print(" Modèle entraîné")
 

## 10. ÉVALUATION
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(" Accuracy :", acc)
 
## 11test de prédiction
sample = X_test.iloc[0:1]
pred = model.predict(sample)
print(" Prédiction :", pred[0])

## 12. SAUVEGARDE
pickle.dump(model, open("model.pkl", "wb"))

print("\n Modèle sauvegardé : model.pkl")
print(" Pipeline terminé avec succès")

## Sauvegarde du dataset nettoyé
df = pd.read_csv("dataset_nettoye.csv")
print(df)