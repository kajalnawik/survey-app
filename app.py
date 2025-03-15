import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")
doctor_activity = pd.read_csv("doctor_activity.csv")

def predict_doctors(survey_hour):
    """Predict doctors likely to attend the survey at a given hour."""
    X_input = pd.DataFrame([[survey_hour]], columns=["Avg_Login_Hour"])
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        return doctor_activity[doctor_activity["Avg_Login_Hour"].round() == survey_hour]["NPI"].tolist()
    return []

# Streamlit UI
st.title("Doctor Survey Prediction")
survey_hour = st.slider("Select Survey Time (Hour)", 0, 23, 12)

if st.button("Get Predicted Doctors"):
    doctors = predict_doctors(survey_hour)
    if doctors:
        df_output = pd.DataFrame({"NPI": doctors})
        st.write(df_output)
        df_output.to_csv("predicted_doctors.csv", index=False)
        st.download_button("Download CSV", df_output.to_csv(index=False), "predicted_doctors.csv", "text/csv")
    else:
        st.write("No doctors predicted for this time.")
