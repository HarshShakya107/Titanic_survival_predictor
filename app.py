import streamlit as st
import numpy as np
import joblib


model = joblib.load("titanic_model.pkl") 

st.title("🚢 Titanic Survival Prediction ")
st.write("Enter passenger details to predict survival")

sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings / Spouse Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Embarked (0=C, 1=Q, 2=S)", [0, 1, 2])
pclass = st.selectbox("Class (1=First, 2=Second, 3=Third)", [1, 2, 3])
who = st.selectbox("Who (0=child, 1=female, 2=male)", [0, 1, 2])
alone = st.selectbox("Alone (0=No, 1=Yes)", [0, 1])

features = np.array([[sex, age, sibsp, parch, fare, embarked, pclass, who, alone]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"Passenger will SURVIVE! (Probability: {probability:.2f})")
    else:
        st.error(f"Passenger will NOT survive (Probability: {probability:.2f})")

