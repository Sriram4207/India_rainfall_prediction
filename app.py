# ----------------------------------------------------
# RainSense - Rainfall Tomorrow Prediction System
# 5 Features + Median Imputation + Scaling + EDA
# Separate Input & Result Page + Accuracy Display
# ----------------------------------------------------

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
ACCURACY_FILE = "accuracy.pkl"

# ----------------------------------------------------
# TRAIN MODEL (FIRST TIME ONLY)
# ----------------------------------------------------

if not os.path.exists(MODEL_FILE):

    print("Training model...")

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

    # Missing Value Handling
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert Yes/No → 1/0
    df["rainfall tomorrow"] = df["rainfall tomorrow"].map({"Yes": 1, "No": 0})
    df = df.dropna()

    # Sample for speed
    df = df.sample(n=20000, random_state=42)

    # Create static folder if not exists
    if not os.path.exists("static"):
        os.makedirs("static")

    # -------- EDA PLOTS --------
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

    # -------- MODEL TRAINING --------
    X = df.drop("rainfall tomorrow", axis=1)
    y = df["rainfall tomorrow"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(accuracy, ACCURACY_FILE)

    print(f"Model trained successfully! Accuracy: {accuracy}%")

else:
    print("Loading saved model...")

# ----------------------------------------------------
# LOAD SAVED FILES
# ----------------------------------------------------

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
accuracy = joblib.load(ACCURACY_FILE)

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", accuracy=accuracy)


@app.route("/predict", methods=["POST"])
def predict():

    try:
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

        if prediction == 1:
            result = "Yes 🌧 Rain Expected Tomorrow"
            background = "rain"
        else:
            result = "No ☀ No Rain Tomorrow"
            background = "sun"

        return render_template(
            "result.html",
            prediction=result,
            background=background,
            accuracy=accuracy
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error in input values",
            background="sun",
            accuracy=accuracy
        )


# ----------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
