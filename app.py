# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Salary Predictor", layout="centered")

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main {
    background-color: #f5f7fa;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.title {
    text-align: center;
    color: #6C63FF;
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<div class='title'>💼 Salary Prediction App</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =========================
# HELPER FUNCTION
# =========================
def get_options(prefix):
    opts = [col.replace(prefix, "") for col in columns if col.startswith(prefix)]
    return ["Other"] + sorted(list(set(opts)))

job_options = get_options("job_title_")
edu_options = get_options("education_level_")
loc_options = get_options("location_")
ind_options = get_options("industry_")
company_options = get_options("company_size_")
remote_options = get_options("remote_work_")

# =========================
# INPUT CARD
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

exp = st.number_input("Experience (years)", 0, 30)
skills = st.number_input("Skills Count", 0, 50)
cert = st.number_input("Certifications", 0, 20)

job = st.selectbox("Job Role", job_options)
edu = st.selectbox("Education", edu_options)
loc = st.selectbox("Location", loc_options)
ind = st.selectbox("Industry", ind_options)
company = st.selectbox("Company Size", company_options)
remote = st.selectbox("Remote Work", remote_options)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CREATE INPUT
# =========================
input_dict = {
    "experience_years": exp,
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
# FEATURE ENGINEERING
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
# PREDICTION BUTTON
# =========================
if st.button("Predict Salary"):

    prediction = model.predict(input_df)

    # 🎉 Toast
    st.toast("Prediction Ready! 🎉")

    # 💎 Styled Result Card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    ">
        <h2>💰 Predicted Salary</h2>
        <h1 style="font-size: 40px;">₹ {int(prediction[0])}</h1>
        <p>Based on your profile</p>
    </div>
    """, unsafe_allow_html=True)

    # ✨ Snow effect
    st.snow()
