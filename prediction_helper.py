
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

EXPECTED_COLUMNS = [
    "age",
    "number_of_dependants",
    "income_lakhs",
    "insurance_plan",
    "genetical_risk",
    "normalized_risk_score",
    "gender_Male",
    "region_Northwest",
    "region_Southeast",
    "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity",
    "bmi_category_Overweight",
    "bmi_category_Underweight",
    "smoking_status_Occasional",
    "smoking_status_Regular",
    "employment_status_Salaried",
    "employment_status_Self-Employed",
]

PLAN_ENCODING = {"Bronze": 1, "Silver": 2, "Gold": 3}

RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}

ARTIFACT_NAMES = (
    "model_young.joblib",
    "model_rest.joblib",
    "scaler_young.joblib",
    "scaler_rest.joblib",
)

_artifacts = None


def load_artifacts():
    """Load artifacts once, when a prediction is requested."""
    global _artifacts

    if _artifacts is None:
        missing = [
            name for name in ARTIFACT_NAMES
            if not (ARTIFACTS_DIR / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing artifact files in {ARTIFACTS_DIR}: "
                f"{', '.join(missing)}"
            )

        _artifacts = {
            name: joblib.load(ARTIFACTS_DIR / name)
            for name in ARTIFACT_NAMES
        }

    return _artifacts


def calculate_normalized_risk(medical_history):
    if not isinstance(medical_history, str) or not medical_history.strip():
        raise ValueError("Please select a valid medical history.")

    diseases = [
        item.strip().lower()
        for item in medical_history.split("&")
        if item.strip()
    ]

    total = sum(RISK_SCORES.get(item, 0) for item in diseases)

    # Maximum supported combined score is 14.
    return min(total, 14) / 14


def preprocess_input(input_dict):
    required = [
        "Age",
        "Number of Dependants",
        "Income in Lakhs",
        "Genetical Risk",
        "Insurance Plan",
        "Employment Status",
        "Gender",
        "Marital Status",
        "BMI Category",
        "Smoking Status",
        "Region",
        "Medical History",
    ]

    missing = [key for key in required if key not in input_dict]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    age = float(input_dict["Age"])
    dependants = float(input_dict["Number of Dependants"])
    income = float(input_dict["Income in Lakhs"])
    genetic_risk = float(input_dict["Genetical Risk"])

    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if not 0 <= dependants <= 20:
        raise ValueError("Dependants must be between 0 and 20.")
    if not 0 <= income <= 200:
        raise ValueError("Income must be between 0 and 200 lakhs.")
    if not 0 <= genetic_risk <= 5:
        raise ValueError("Genetical risk must be between 0 and 5.")

    plan = input_dict["Insurance Plan"]
    if plan not in PLAN_ENCODING:
        raise ValueError(f"Invalid insurance plan: {plan}")

    df = pd.DataFrame(
        0.0, index=[0], columns=EXPECTED_COLUMNS
    )

    df.loc[0, "age"] = age
    df.loc[0, "number_of_dependants"] = dependants
    df.loc[0, "income_lakhs"] = income
    df.loc[0, "insurance_plan"] = PLAN_ENCODING[plan]
    df.loc[0, "genetical_risk"] = genetic_risk

    categorical_mapping = {
        ("Gender", "Male"): "gender_Male",
        ("Region", "Northwest"): "region_Northwest",
        ("Region", "Southeast"): "region_Southeast",
        ("Region", "Southwest"): "region_Southwest",
        ("Marital Status", "Unmarried"): "marital_status_Unmarried",
        ("BMI Category", "Obesity"): "bmi_category_Obesity",
        ("BMI Category", "Overweight"): "bmi_category_Overweight",
        ("BMI Category", "Underweight"): "bmi_category_Underweight",
        ("Smoking Status", "Occasional"): "smoking_status_Occasional",
        ("Smoking Status", "Regular"): "smoking_status_Regular",
        ("Employment Status", "Salaried"): "employment_status_Salaried",
        ("Employment Status", "Self-Employed"):
            "employment_status_Self-Employed",
    }

    for (field, category), column in categorical_mapping.items():
        if input_dict[field] == category:
            df.loc[0, column] = 1.0

    df.loc[0, "normalized_risk_score"] = calculate_normalized_risk(
        input_dict["Medical History"]
    )

    return df


def handle_scaling(age, df, scaler_object):
    """Apply the scaler saved for the relevant age group."""
    if isinstance(scaler_object, dict):
        if "scaler" not in scaler_object or "cols_to_scale" not in scaler_object:
            raise ValueError(
                "Scaler artifact must contain 'scaler' and 'cols_to_scale'."
            )
        scaler = scaler_object["scaler"]
        columns = list(scaler_object["cols_to_scale"])
    else:
        scaler = scaler_object
        columns = list(getattr(scaler, "feature_names_in_", []))

        if not columns:
            raise ValueError(
                "The scaler has no saved feature names. "
                "Check the scaler artifact created during training."
            )

    # The original preprocessing expects income_level in some scalers.
    # It is temporary and removed before model prediction.
    temporary_columns = []

    for column in columns:
        if column not in df.columns:
            if column == "income_level":
                df[column] = 0.0
                temporary_columns.append(column)
            else:
                raise ValueError(
                    f"Scaler expects missing input column '{column}'. "
                    "Check the preprocessing used during model training."
                )

    if columns:
        df.loc[:, columns] = scaler.transform(df.loc[:, columns])

    for column in temporary_columns:
        df.drop(columns=column, inplace=True)

    return df


def predict(input_dict):
    artifacts = load_artifacts()

    input_df = preprocess_input(input_dict)
    age = float(input_dict["Age"])

    if age <= 25:
        model = artifacts["model_young.joblib"]
        scaler_object = artifacts["scaler_young.joblib"]
    else:
        model = artifacts["model_rest.joblib"]
        scaler_object = artifacts["scaler_rest.joblib"]

    input_df = handle_scaling(age, input_df, scaler_object)

    # Match the trained model's feature order when it is available.
    model_columns = getattr(model, "feature_names_in_", None)

    if model_columns is not None:
        model_columns = list(model_columns)
        missing = [col for col in model_columns if col not in input_df.columns]

        if missing:
            raise ValueError(
                f"Model expects features missing from preprocessing: {missing}"
            )

        input_df = input_df.loc[:, model_columns]
    else:
        if input_df.shape[1] != len(EXPECTED_COLUMNS):
            raise ValueError(
                "The processed input has an unexpected number of features. "
                "Check the original model's training columns."
            )
        input_df = input_df.loc[:, EXPECTED_COLUMNS]

    result = model.predict(input_df)
    return int(round(float(result[0])))
