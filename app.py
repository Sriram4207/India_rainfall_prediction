# ----------------------------------------------------
# Rainfall Tomorrow Prediction System
# Improved Model (Target: ~90% Accuracy)
# 5 Features + Median Imputation + Scaling + EDA
# ----------------------------------------------------

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
ACCURACY_FILE = "accuracy.pkl"

# ----------------------------------------------------
# TRAIN MODEL (ONLY FIRST TIME)
# ----------------------------------------------------

if not os.path.exists(MODEL_FILE):

    print("Training high-accuracy model...")

    df = pd.read_csv("india_weather_rainfall_data.csv")

    selected_columns = [
        "avg_temp",
        "min_temp",
        "wind_speed",
        "air_pressure",
        "rainfall",
        "rainfall tomorrow"
    ]

    df = df[selected_columns]

    # -------- Missing Value Handling --------
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert target Yes/No → 1/0
    df["rainfall tomorrow"] = df["rainfall tomorrow"].map({"Yes": 1, "No": 0})
    df = df.dropna()

    # --------- CREATE SMALL EDA PLOTS ----------
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.figure(figsize=(4,3))
    sns.histplot(df["rainfall"], kde=True)
    plt.title("Rainfall Distribution")
    plt.tight_layout()
    plt.savefig("static/rainfall_dist.png")
    plt.close()

    plt.figure(figsize=(4,3))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("static/correlation.png")
    plt.close()

    plt.figure(figsize=(4,3))
    sns.scatterplot(x=df["avg_temp"], y=df["rainfall"])
    plt.title("Avg Temp vs Rainfall")
    plt.tight_layout()
    plt.savefig("static/temp_vs_rain.png")
    plt.close()

    # --------- FEATURES & TARGET ----------
    X = df.drop("rainfall tomorrow", axis=1)
    y = df["rainfall tomorrow"]

    # --------- SCALING ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------- STRATIFIED SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # --------- RANDOM FOREST WITH BALANCING ----------
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    # --------- HYPERPARAMETER TUNING ----------
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [15, 20, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    # --------- ACCURACY ----------
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    print("Best Parameters:", grid.best_params_)
    print("Final Accuracy:", accuracy, "%")

    # --------- SAVE MODEL ----------
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(accuracy, ACCURACY_FILE)

else:
    print("Loading saved model...")

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

if os.path.exists(ACCURACY_FILE):
    accuracy = joblib.load(ACCURACY_FILE)
else:
    accuracy = "Not Available"

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", accuracy=accuracy)

@app.route("/predict", methods=["POST"])
def predict():

    input_data = [
        float(request.form["avg_temp"]),
        float(request.form["min_temp"]),
        float(request.form["wind_speed"]),
        float(request.form["air_pressure"]),
        float(request.form["rainfall"])
    ]

    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    result = "Yes 🌧 Rain Expected Tomorrow" if prediction == 1 else "No ☀ No Rain Tomorrow"

    return render_template(
        "index.html",
        prediction=result,
        accuracy=accuracy
    )

# ----------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
