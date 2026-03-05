import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Cleaned_Burnout_Dataset.csv")

# Create Burnout Score
df["Burnout_Score"] = (
    df["Stress_Score"] * 10 +
    (10 - df["Sleep_Hours"]) * 5 +
    df["Screen_Time"] * 4 +
    df["Workload_Level"] * 8
)

df["Burnout_Score"] = (
    (df["Burnout_Score"] - df["Burnout_Score"].min()) /
    (df["Burnout_Score"].max() - df["Burnout_Score"].min())
) * 100

# Feature selection
X = df.drop("Burnout_Risk", axis=1)
y = df["Burnout_Risk"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -------- STREAMLIT UI --------

st.title("Career Burnout Early Warning System")

st.header("Enter Your Details")

sleep = st.slider("Sleep Hours", 0, 10, 6)
stress = st.slider("Stress Score (1-10)", 1, 10, 5)
screen = st.slider("Screen Time (Hours)", 0, 15, 5)
workload = st.slider("Workload Level (1-5)", 1, 5, 3)
career_clarity = st.slider("Career Clarity (1-5)", 1, 5, 3)

if st.button("Predict Burnout Risk"):

    burnout_score = (
    stress * 10 +
    (10 - sleep) * 5 +
    screen * 4 +
    workload * 8
)

burnout_score = (
(burnout_score - df["Burnout_Score"].min()) /
(df["Burnout_Score"].max() - df["Burnout_Score"].min())
) * 100

    input_data = pd.DataFrame([{
        "Sleep_Hours": sleep,
        "Stress_Score": stress,
        "Screen_Time": screen,
        "Workload_Level": workload,
        "Burnout_Score": burnout_score,
        "Careee_Clarity": career_clarity 
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")
    st.success(f"Burnout Risk Level: {prediction}")

    reasons = []
    if sleep < 6:
        reasons.append("Low sleep duration")
    if stress > 7:
        reasons.append("High stress level")
    if screen > 8:
        reasons.append("Excessive screen time")
    if workload > 4:
        reasons.append("High workload")

    if reasons:
        st.subheader("Main Reasons:")
        for r in reasons:
            st.write("-", r)

    recommendations = []
    if "Low sleep duration" in reasons:
        recommendations.append("Improve sleep routine")
    if "High stress level" in reasons:
        recommendations.append("Practice relaxation techniques")
    if "Excessive screen time" in reasons:
        recommendations.append("Reduce screen exposure")
    if "High workload" in reasons:
        recommendations.append("Plan workload effectively")

    if recommendations:
        st.subheader("Preventive Recommendations:")
        for rec in recommendations:
            st.write("-", rec)
