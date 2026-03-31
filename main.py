import streamlit as st
import pandas as pd
import joblib 

model=joblib.load("LG.pkl")
scaler=joblib.load("scaler.pkl")
cols=joblib.load("col.pkl")

st.title("Heart Stroke Prediction System")
st.markdown("Enter following details")
age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",['M', 'F' ])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200,120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st. selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("0ldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
    'Age' : age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex' : sex,
    'ChestPainType' : chest_pain,
    'RestingECG' : resting_ecg,
    'ExerciseAngina' : exercise_angina,
    'ST_Slope' : st_slope
    }

    input_df=pd.DataFrame([raw_input])
    input_df = pd.get_dummies(input_df)

    # for _ in cols:
    #     if _ not in input_df.columns:
    #         input_df[_]=0

    # input_df=input_df[cols]
    input_df = input_df.reindex(columns=cols, fill_value=0)



    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("Heart disease risk")
    else:
        st.success("No heart disease")