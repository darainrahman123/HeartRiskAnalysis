import streamlit as st
import pandas as pd
import joblib 

# Load assets
model = joblib.load("LG.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("col.pkl")

st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Styling
st.markdown("""
<style>

.stButton>button {
    width:100%;
    height:55px;
    font-size:18px;
    border-radius:10px;
    background-color:#ff4b4b;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# Header
st.title("❤️ Heart Disease Risk Prediction")

st.markdown("""
Predict heart disease risk using clinical health parameters.
""")

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age",18,100,40)

    sex = st.selectbox(
        "Sex",
        ['M','F']
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","TA","ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80,200,120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100,600,200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar >120",
        [0,1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    max_hr = st.slider(
        "Max Heart Rate",
        60,220,150
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,6.0,1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"]
    )

st.divider()

# Prediction
# Prediction
if st.button("Predict Heart Risk ❤️"):

    raw_input = {

        'Age':age,
        'RestingBP':resting_bp,
        'Cholesterol':cholesterol,
        'FastingBS':fasting_bs,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex':sex,
        'ChestPainType':chest_pain,
        'RestingECG':resting_ecg,
        'ExerciseAngina':exercise_angina,
        'ST_Slope':st_slope
    }

    input_df = pd.DataFrame([raw_input])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=cols,fill_value=0)

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()

    st.markdown("## Heart Risk Analysis")

    # metrics row
    m1,m2,m3 = st.columns(3)

    with m1:

        if prediction == 1:
            st.metric("Risk Level","High")
        else:
            st.metric("Risk Level","Low")

    with m2:

        st.metric(
            "Risk Probability",
            f"{round(probability*100,1)}%"
        )

    with m3:

        st.metric(
            "Max Heart Rate",
            max_hr
        )

    # Risk gauge style indicator
    risk_percent = probability*100

    st.progress(int(risk_percent))

    # Interpretation
    if probability > 0.7:

        st.error("🔴 High cardiovascular risk detected")

        st.markdown("""
        **Recommendation:**
        - Consult cardiologist
        - Lifestyle modification advised
        - Further diagnostic tests recommended
        """)

    elif probability > 0.4:

        st.warning("🟡 Moderate risk detected")

        st.markdown("""
        **Recommendation:**
        - Regular monitoring suggested
        - Improve diet and exercise
        """)

    else:

        st.success("🟢 Low cardiovascular risk")

        st.markdown("""
        **Recommendation:**
        - Maintain healthy lifestyle
        - Regular checkups
        """)

    # Medical insights
    st.markdown("### Health Insights")

    info1,info2 = st.columns(2)

    with info1:

        if cholesterol > 240:
            st.warning("High cholesterol detected")

        if resting_bp > 140:
            st.warning("High blood pressure")

    with info2:

        if max_hr < 100:
            st.info("Low exercise heart rate")

        if oldpeak > 2:
            st.warning("Possible ECG abnormality")

    # Debug panel
    with st.expander("Model Features"):

        st.dataframe(input_df)


st.divider()

st.caption("ML Model | Streamlit Deployment | AI Engineer Portfolio Project")
