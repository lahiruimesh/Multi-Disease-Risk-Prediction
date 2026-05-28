import streamlit as st
import numpy as np
import pickle
from xgboost import XGBClassifier

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🩺", layout="centered")

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
st.html("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 8px;
        padding: 10px 24px; font-weight: bold; width: 100%; border: none;
    }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .report-box {
        padding: 20px; border-radius: 10px; background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    </style>
""")

# --- 2. LOAD MODELS AND SCALER ---
@st.cache_resource
def load_resources():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    heart_mdl = XGBClassifier()
    heart_mdl.load_model('heart_model.json')
    
    diabetes_mdl = XGBClassifier()
    diabetes_mdl.load_model('diabetes_model.json')
    
    chol_mdl = XGBClassifier()
    chol_mdl.load_model('chol_model.json')
    
    return scaler, heart_mdl, diabetes_mdl, chol_mdl

try:
    scaler, heart_model, diabetes_model, cholesterol_model = load_resources()
except Exception as e:
    st.error(f"Error loading models or scaler: {e}")

# --- 3. WEB APP UI TITLE ---
st.html("<h1 style='text-align: center; color: #1E3A8A; font-family: sans-serif;'>🩺 Multi-Disease Risk Prediction System</h1>")
st.html("<p style='text-align: center; color: #4B5563; font-family: sans-serif;'>Enter your health indicators to simultaneously assess risks for Heart Disease, Diabetes, and High Cholesterol.</p>")
st.markdown("---")

# --- 4. USER INPUTS ---
st.subheader("👤 Personal Health Indicators")

col1, col2 = st.columns(2)

with col1:
    age_group = st.slider("Age Group Index:", 1, 13, 5, 
                          help="1: 18-24, 2: 25-29, ... 9: 60-64, 13: 80 or older")
    bmi = st.number_input("BMI (Body Mass Index):", min_value=12.0, max_value=70.0, value=25.0)
    high_bp = st.selectbox("Do you have High Blood Pressure (High BP)?", [("No", 0), ("Yes", 1)])[1]
    high_chol = st.selectbox("Do you have High Cholesterol (High Chol)?", [("No", 0), ("Yes", 1)])[1]
    chol_check = st.selectbox("Have you had a cholesterol check in the past 5 years?", [("Yes", 1), ("No", 0)])[1]

with col2:
    smoker = st.selectbox("Have you smoked at least 100 cigarettes in your entire life?", [("No", 0), ("Yes", 1)])[1]
    stroke = st.selectbox("Have you ever been told you had a stroke?", [("No", 0), ("Yes", 1)])[1]
    phys_activity = st.selectbox("Physical activity in the past 30 days (excluding your job)?", [("Yes", 1), ("No", 0)])[1]
    fruits = st.selectbox("Do you consume fruit 1 or more times per day?", [("Yes", 1), ("No", 0)])[1]
    gender = st.selectbox("Gender:", [("Male", 1), ("Female", 0)])[1]

# Default placeholders for features not gathered via inputs to match the expected 17-column format
veggie = 1
hvy_alcohol = 0
gen_hlth = 3
ment_hlth = 3
phys_hlth = 2
diff_walk = 0

# --- 5. PREDICTION LOGIC ---
if st.button("📊 Check Risk Analysis"):
    
    input_data = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, 
                            0.0, # Dummy placeholder
                            phys_activity, fruits, veggie, hvy_alcohol, gen_hlth, ment_hlth, phys_hlth, diff_walk, gender, age_group]])
    
    scaled_data = scaler.transform(input_data)
    
    # Predict Probabilities
    heart_risk = heart_model.predict_proba(scaled_data)[0][1] * 100
    diabetes_risk = diabetes_model.predict_proba(scaled_data)[0][1] * 100
    chol_risk = cholesterol_model.predict_proba(scaled_data)[0][1] * 100
    
    # --- 6. DISPLAY RESULTS ---
    st.markdown("---")
    st.html("<h3 style='color: #1E3A8A; font-family: sans-serif;'>📋 Your Personal Risk Analysis Report</h3>")
    
    rc1, rc2, rc3 = st.columns(3)
    
    with rc1:
        st.html("<div class='report-box'>")
        st.metric(label="🫀 Heart Disease Risk", value=f"{heart_risk:.1f}%")
        if heart_risk > 50: st.error("🚨 High Risk Profile")
        else: st.success("✅ Safe / Low Risk")
        st.html("</div>")
        
    with rc2:
        st.html("<div class='report-box'>")
        st.metric(label="🩸 Diabetes Risk", value=f"{diabetes_risk:.1f}%")
        if diabetes_risk > 50: st.error("🚨 High Risk Profile")
        else: st.success("✅ Safe / Low Risk")
        st.html("</div>")
        
    with rc3:
        st.html("<div class='report-box'>")
        st.metric(label="🧪 Cholesterol Risk", value=f"{chol_risk:.1f}%")
        if chol_risk > 50: st.error("🚨 High Risk Profile")
        else: st.success("✅ Safe / Low Risk")
        st.html("</div>")
        
    # --- 7. MEDICAL RECOMMENDATIONS ---
    st.markdown("---")
    if heart_risk > 50 or diabetes_risk > 50 or chol_risk > 50:
        st.warning("⚠️ **Health Advisory:** You present an elevated risk for one or more conditions. It is highly recommended to consult a physician for professional clinical screenings. Consider modifying dietary habits and incorporating regular cardio exercise.")
    else:
        st.success("✅ **Healthy Profile:** Your metrics indicate a low risk pattern. Maintain your healthy habits, balanced diet, and active lifestyle!")