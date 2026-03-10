import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ------------------ LOAD DATA ------------------

df = pd.read_csv("Cleaned_Burnout_Dataset.csv")

# Create Burnout Score
df["Burnout_Score"] = (
    df["Stress_Score"] * 10 +
    (10 - df["Sleep_Hours"]) * 5 +
    df["Screen_Time"] * 4 +
    df["Workload_Level"] * 8
)

# Normalize Burnout Score (0–100)
df["Burnout_Score"] = (
    (df["Burnout_Score"] - df["Burnout_Score"].min()) /
    (df["Burnout_Score"].max() - df["Burnout_Score"].min())
    ) * 100

# ------------------ MODEL TRAINING ------------------

X = df.drop("Burnout_Risk", axis=1)
y = df["Burnout_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
feature_order = X.columns

# ------------------ STREAMLIT UI ------------------

st.title("Career Burnout Early Warning System")

with st.expander("Self-Assessment Guidelines"):
    st.markdown("""
    ### 🔹 Stress Score Guide
    1–3 → Low stress  
    4–6 → Moderate stress  
    7–8 → High stress  
    9–10 → Extremely high stress 

    ### 🔹 Workload Level Guide (1–5)
    1 → Very light workload  
    2 → Light workload  
    3 → Manageable  
    4 → Heavy  
    5 → Extremely heavy workload  

    ### 🔹 Career Clarity Guide (1–5)
    1 → Completely confused about career  
    2 → Slightly unclear  
    3 → Somewhat clear  
    4 → Clear  
    5 → Very clear and confident  
    """) 

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

    # Create dictionary first
    input_dict = {
        "Sleep_Hours": sleep,
        "Stress_Score": stress,
        "Screen_Time": screen,
        "Workload_Level": workload,
        "Careee_Clarity": career_clarity,
        "Burnout_Score": burnout_score
    }

    # Convert to DataFrame
    input_data = pd.DataFrame([input_dict])

    # Reorder columns EXACTLY like training
    input_data = input_data[feature_order]

    prediction = model.predict(input_data)[0]

    

    st.subheader("Prediction Result:")

    if prediction == "High":
        st.error("⚠ High Risk of Burnout")


    # -------- Explainable AI Section --------

        reasons = []

        if sleep < 6:
            reasons.append("Low sleep duration")
        if stress > 7:
            reasons.append("High stress level")
        if screen > 8:
            reasons.append("Excessive screen time")
        if workload > 4:
            reasons.append("High workload")
        if career_clarity <= 2:
            reasons.append("Low career clarity")

        if reasons:
            st.subheader("Main Reasons:")
            for r in reasons:
                st.write("-", r)
                    
    
    # -------- Intervention Section Based on Risk --------

    
        st.subheader("Personalized Burnout Recovery Strategy")

        if sleep < 6:
            st.markdown("### Sleep Recovery Strategy")
            st.markdown("""
                        **Risk Insight:**
                        Sleeping less than 6 hours increases cortisol (stress hormone) and reduces emotional regulation capacity.
                        
                        **⚠ Burnout Impact:**
                        Chronic sleep deprivation accelerates fatigue, reduces focus, and increases emotional exhaustion.
                        
                        **7-Day Action Plan:**
                        • Maintain fixed sleep schedule  
                        • Avoid screens 30 minutes before bed  
                        • Reduce caffeine after evening  
                        • Keep bedroom environment dark and cool
                        
                        **Expected Outcome:**
                        Improved energy, better mood stability, and reduced burnout symptoms.
                        """)

        if stress > 7:
            st.markdown("### Stress Regulation Plan")
            st.markdown("""
                        **Risk Insight:**
                        High stress elevates cortisol and adrenaline levels continuously.
                        
                        **⚠ Burnout Impact:**
                        Accelerates emotional exhaustion and reduces productivity.
                        
                        **7-Day Action Plan:**
                        • Practice 10-minute deep breathing daily  
                        • Use 50-10 focus technique  
                        • Prioritize only 3 key tasks daily  
                        • Include 20 min physical activity  

                        **Expected Outcome:**
                        Better emotional control and reduced pressure.
                        """)

        if screen > 8:
            st.markdown("### Digital Detox Strategy")
            st.markdown("""
                        **Risk Insight:**
                        Excessive screen exposure disrupts circadian rhythm and dopamine balance.

                        **⚠ Burnout Impact:**  
                        Causes attention fatigue and sleep disturbance.

                        **7-Day Action Plan:**
                        • Follow 20-20-20 rule  
                        • Avoid screens 1 hour before sleep  
                        • Schedule daily offline time  
                        • Enable blue light filter  

                        **Expected Outcome:**  
                        Improved focus and sleep quality.
                        """)

        if workload > 4:
            st.markdown("### Workload Optimization Plan")
            st.markdown("""
                        **Risk Insight:**
                        High workload increases cognitive overload.
                        
                        **⚠ Burnout Impact:**
                        Leads to reduced efficiency and mental exhaustion.

                        **7-Day Action Plan:**
                        • Break tasks into smaller goals  
                        • Use priority matrix  
                        • Delegate when possible  
                        • Schedule fixed relaxation period  

                        **Expected Outcome:**  
                        Better productivity and reduced overwhelm.
                        """)

        if career_clarity <= 2:
            st.markdown("### Career Clarity Development Plan")
            st.markdown("""
                        **Risk Insight:**
                        Uncertainty increases anxiety due to lack of future direction.

                        **⚠ Burnout Impact:**  
                        Creates chronic mental stress and low motivation.

                        **7-Day Action Plan:**  
                        • Identify 3 career interests  
                        • Research required skills  
                        • Talk to mentor or senior  
                        • Set a 30-day learning goal  

                        **Expected Outcome:**  
                        Improved motivation and reduced uncertainty stress.
                        """)
            
    elif prediction == "Medium":
        st.warning("⚠ Moderate Risk of Burnout")
        
        st.markdown("### 🟡 Early Intervention Advisory")
        st.markdown("""
                    You are entering the Stress Accumulation Zone.
                    Your current patterns are not critical yet, but without correction, they can gradually convert into high burnout risk within weeks.
                    
                    **Recommended Immediate Actions:**
                    
                    Activate the 90-Minute Focus Cycle Rule
                    
                    • Work in 90-minute deep focus blocks
                            
                    • Take 15-minute recovery break
                    
                    • No multitasking during focus cycle
                    
                    """)
        
        
        st.info("📊 Risk Escalation Probability: If current patterns continue unchanged, risk intensity may escalate within 2–4 weeks.")

            
    else:
        
        st.success("Low Risk of Burnout")
        st.success("Great! Your lifestyle indicators are within healthy range. Maintain consistency and monitor regularly.")

    

    