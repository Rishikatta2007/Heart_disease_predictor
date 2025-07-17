import streamlit as st
import pandas as pd
import pickle

# Load or train mode
model = pickle.load(open('random_forest_model_1', 'rb'))

# App title
st.title('Heart Disease Prediction App')

# User input section
st.header('Patient Information')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', [
        'Typical angina', 
        'Atypical angina', 
        'Non-anginal pain', 
        'Asymptomatic'
    ])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])

with col2:
    restecg = st.selectbox('Resting Electrocardiographic Results', [
        'Normal',
        'ST-T wave abnormality',
        'Left ventricular hypertrophy'
    ])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', [
        'Upsloping',
        'Flat',
        'Downsloping'
    ])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'])
    thal = st.selectbox('Thalassemia', [
        'Normal',
        'Fixed defect',
        'Reversible defect'
    ])

# Convert inputs to model format
sex = 1 if sex == 'Male' else 0
cp_dict = {'Typical angina': 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3}
cp = cp_dict[cp]
fbs = 1 if fbs == 'Yes' else 0
restecg_dict = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
restecg = restecg_dict[restecg]
exang = 1 if exang == 'Yes' else 0
slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
slope = slope_dict[slope]
ca = int(ca)
thal_dict = {'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}
thal = thal_dict[thal]

# Create feature array
features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

# Make prediction
if st.button('Predict Heart Disease'):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error(f'High risk of heart disease ({probability[0][1]*100:.2f}% probability)')
    else:
        st.success(f'Low risk of heart disease ({probability[0][0]*100:.2f}% probability)')
    
    
