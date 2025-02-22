import os
import pickle  # For loading pre-trained models
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon='🏥')

# Load models
model_paths = {
    "diabetes": "D:/ML_Project/Disease_Prediction_Outbreak/training_models/diabetes_model.sav",
    "heart": "D:/ML_Project/Disease_Prediction_Outbreak/training_models/heart_model.sav",
    "parkinsons": "D:/ML_Project/Disease_Prediction_Outbreak/training_models/parkinsons_model.sav"
}

models = {}
for key, path in model_paths.items():
    try:
        with open(path, 'rb') as file:
            models[key] = pickle.load(file)
        print(f"{key} model loaded successfully!")
    except Exception as e:
        print(f"Error loading {key} model: {e}")

# Sidebar Menu
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age of the Person")

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]

        prediction = models["diabetes"].predict([user_input])
        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        st.success(result)

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
        cp = st.text_input("Chest Pain Type (cp)")
        chol = st.text_input("Cholesterol Level")
        restecg = st.text_input("Resting ECG Result")
        oldpeak = st.text_input("ST Depression Induced")
    with col2:
        sex = st.text_input("Sex (1 = Male, 0 = Female)")
        trestbps = st.text_input("Resting Blood Pressure")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)")
        thalach = st.text_input("Maximum Heart Rate Achieved")
        slope = st.text_input("Slope of the Peak Exercise ST Segment")
    with col3:
        exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
        ca = st.text_input("Number of Major Vessels (0-3)")
        thal = st.text_input("Thalassemia (0-3)")

    if st.button("Heart Disease Test Result"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]

        prediction = models["heart"].predict([user_input])
        result = "The person has a risk of heart disease" if prediction[0] == 1 else "The person is healthy"
        st.success(result)

# Parkinson’s Prediction
elif selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        Fo = st.text_input("MDVP:Fo(Hz)")
        Jitter_perc = st.text_input("MDVP:Jitter(%)")
        RAP = st.text_input("MDVP:RAP")
        Shimmer = st.text_input("MDVP:Shimmer")
        APQ5 = st.text_input("Shimmer:APQ5")
        NHR = st.text_input("NHR")
        DFA = st.text_input("DFA")
    with col2:
        Fhi = st.text_input("MDVP:Fhi(Hz)")
        Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
        PPQ = st.text_input("MDVP:PPQ")
        Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
        APQ = st.text_input("MDVP:APQ")
        HNR = st.text_input("HNR")
        spread1 = st.text_input("Spread1")
    with col3:
        Flo = st.text_input("MDVP:Flo(Hz)")
        Jitter_DDP = st.text_input("Jitter:DDP")
        Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
        Shimmer_DDA = st.text_input("Shimmer:DDA")
        RPDE = st.text_input("RPDE")
        spread2 = st.text_input("Spread2")
        D2 = st.text_input("D2")
        PPE = st.text_input("PPE")

    if st.button("Parkinson's Disease Test Result"):
        user_input = [Fo, Fhi, Flo, Jitter_perc, Jitter_Abs, RAP, PPQ, Jitter_DDP, Shimmer, Shimmer_dB, Shimmer_APQ3,
                      APQ5, APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = [float(x) for x in user_input]

        prediction = models["parkinsons"].predict([user_input])
        result = "The person is at risk of Parkinson’s" if prediction[0] == 1 else "The person is healthy"
        st.success(result)
