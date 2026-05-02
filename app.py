# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Salary Predictor", layout="wide")

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
page = st.sidebar.radio("Navigate", ["🏠 Prediction", "📊 Analytics"])

# =========================
# THEME CSS
# =========================
if mode == "Dark":
    bg_color = "#0f172a"
    text_color = "white"
    card_color = "rgba(255,255,255,0.05)"
else:
    bg_color = "#f5f7fa"
    text_color = "black"
    card_color = "white"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
    color: {text_color};
}}

.card {{
    background: {card_color};
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    transition: 0.3s ease;
}}

.card:hover {{
    transform: scale(1.02);
}}

.title {{
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

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
# PAGE 1: PREDICTION
# =========================
if page == "🏠 Prediction":

    st.markdown("<div class='title'>💼 Salary Prediction App</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        exp = st.number_input("Experience (years)", 0, 30)
        skills = st.number_input("Skills Count", 0, 50)

    with col2:
        cert = st.number_input("Certifications", 0, 20)
        job = st.selectbox("Job Role", job_options)

    with col3:
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

    # FEATURE ENGINEERING
    input_df['exp_squared'] = input_df['experience_years'] ** 2
    input_df['skill_per_exp'] = input_df['skills_count'] / (input_df['experience_years'] + 1)
    input_df['cert_per_skill'] = input_df['certifications'] / (input_df['skills_count'] + 1)

    input_df['seniority'] = pd.cut(
        input_df['experience_years'],
        bins=[0, 2, 5, 10, 20],
        labels=['Fresher', 'Junior', 'Mid', 'Senior']
    )

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    num_cols = ['experience_years', 'skills_count', 'certifications',
                'exp_squared', 'skill_per_exp', 'cert_per_skill']

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # =========================
    # PREDICT
    # =========================
    if st.button("Predict Salary"):
        prediction = model.predict(input_df)

        st.toast("Prediction Ready! 🎉")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-top: 20px;
        ">
            <h2>💰 Predicted Salary</h2>
            <h1>₹ {int(prediction[0])}</h1>
        </div>
        """, unsafe_allow_html=True)

        st.snow()

# =========================
# PAGE 2: ANALYTICS
# =========================
elif page == "📊 Analytics":

    st.markdown("<div class='title'>📊 Salary Insights</div>", unsafe_allow_html=True)

    # Dummy data (or replace with your dataset)
    exp_range = np.arange(1, 21)
    salary_trend = 20000 + exp_range * 8000 + np.random.randint(-10000, 10000, 20)

    fig, ax = plt.subplots()
    ax.plot(exp_range, salary_trend)
    ax.set_xlabel("Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Salary vs Experience")

    st.pyplot(fig)

    # Bar chart
    roles = ["Data Scientist", "Engineer", "Analyst"]
    salaries = [120000, 90000, 70000]

    fig2, ax2 = plt.subplots()
    ax2.bar(roles, salaries)
    ax2.set_title("Salary by Role")

    st.pyplot(fig2)
