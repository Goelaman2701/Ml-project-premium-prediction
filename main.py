
import streamlit as st

from prediction_helper import predict


st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
)

st.title("Health Insurance Cost Predictor")
st.write("Enter the details below to estimate your health insurance cost.")

OPTIONS = {
    "Gender": ["Male", "Female"],
    "Marital Status": ["Unmarried", "Married"],
    "BMI Category": ["Normal", "Obesity", "Overweight", "Underweight"],
    "Smoking Status": ["No Smoking", "Regular", "Occasional"],
    "Employment Status": ["Salaried", "Self-Employed", "Freelancer", ""],
    "Region": ["Northwest", "Southeast", "Northeast", "Southwest"],
    "Medical History": [
        "No Disease",
        "Diabetes",
        "High blood pressure",
        "Diabetes & High blood pressure",
        "Thyroid",
        "Heart disease",
        "High blood pressure & Heart disease",
        "Diabetes & Thyroid",
        "Diabetes & Heart disease",
    ],
    "Insurance Plan": ["Bronze", "Silver", "Gold"],
}

with st.form("prediction_form"):
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)

    with row1[0]:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=25, step=1
        )
    with row1[1]:
        dependants = st.number_input(
            "Number of Dependants", min_value=0, max_value=20, value=0, step=1
        )
    with row1[2]:
        income = st.number_input(
            "Income in Lakhs", min_value=0, max_value=200, value=5, step=1
        )

    with row2[0]:
        genetic_risk = st.number_input(
            "Genetical Risk", min_value=0, max_value=5, value=0, step=1
        )
    with row2[1]:
        plan = st.selectbox("Insurance Plan", OPTIONS["Insurance Plan"])
    with row2[2]:
        employment = st.selectbox(
            "Employment Status",
            OPTIONS["Employment Status"],
            format_func=lambda value: value if value else "Not specified",
        )

    with row3[0]:
        gender = st.selectbox("Gender", OPTIONS["Gender"])
    with row3[1]:
        marital_status = st.selectbox(
            "Marital Status", OPTIONS["Marital Status"]
        )
    with row3[2]:
        bmi = st.selectbox("BMI Category", OPTIONS["BMI Category"])

    with row4[0]:
        smoking = st.selectbox("Smoking Status", OPTIONS["Smoking Status"])
    with row4[1]:
        region = st.selectbox("Region", OPTIONS["Region"])
    with row4[2]:
        medical_history = st.selectbox(
            "Medical History", OPTIONS["Medical History"]
        )

    submitted = st.form_submit_button(
        "Predict Insurance Cost", use_container_width=True
    )

if submitted:
    input_dict = {
        "Age": age,
        "Number of Dependants": dependants,
        "Income in Lakhs": income,
        "Genetical Risk": genetic_risk,
        "Insurance Plan": plan,
        "Employment Status": employment,
        "Gender": gender,
        "Marital Status": marital_status,
        "BMI Category": bmi,
        "Smoking Status": smoking,
        "Region": region,
        "Medical History": medical_history,
    }

    try:
        with st.spinner("Generating prediction..."):
            cost = predict(input_dict)

        st.success("Prediction completed.")
        st.metric(
            "Predicted Health Insurance Cost",
            f"₹{cost:,.0f}",
        )
        st.caption(
            "This is an estimate from your model, not a guaranteed insurance quote."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the artifacts folder is beside these Python files "
            "and that the saved model and scaler features match the code."
        )
