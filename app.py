# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import pandas as pd

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# TITLE
# =========================
st.title("💼 Salary Prediction App (KNN Improved)")

# =========================
# USER INPUT
# =========================
age = st.number_input("Experience (years)", 0, 30)
skills = st.number_input("Skills Count", 0, 50)
cert = st.number_input("Certifications", 0, 20)

job = st.selectbox("Job Role", ["Data Scientist", "Data Analyst", "Software Engineer"])
edu = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
loc = st.selectbox("Location", ["India", "USA", "Canada"])
ind = st.selectbox("Industry", ["Technology", "Finance", "Healthcare"])
company = st.selectbox("Company Size", ["Small", "Medium", "Large"])
remote = st.selectbox("Remote Work", ["Yes", "No"])

# =========================
# CREATE INPUT
# =========================
input_dict = {
    "experience_years": age,
    "skills_count": skills,
    "certifications": cert,
    "job_title": job,
    "education_level": edu,
    "location": loc,
    "industry": ind,
    "company_size": company,
    "remote_work": remote
}

input_df = pd.DataFrame([input_dict])

# =========================
# SAME FEATURE ENGINEERING
# =========================
input_df['exp_squared'] = input_df['experience_years'] ** 2
input_df['skill_per_exp'] = input_df['skills_count'] / (input_df['experience_years'] + 1)
input_df['cert_per_skill'] = input_df['certifications'] / (input_df['skills_count'] + 1)

input_df['seniority'] = pd.cut(
    input_df['experience_years'],
    bins=[0, 2, 5, 10, 20],
    labels=['Fresher', 'Junior', 'Mid', 'Senior']
)

# =========================
# DUMMIES + ALIGN
# =========================
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# =========================
# SCALE
# =========================
num_cols = ['experience_years', 'skills_count', 'certifications',
            'exp_squared', 'skill_per_exp', 'cert_per_skill']

input_df[num_cols] = scaler.transform(input_df[num_cols])

# =========================
# PREDICTION
# =========================
if st.button("Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Salary: {int(prediction[0])}")
    st.balloons() 