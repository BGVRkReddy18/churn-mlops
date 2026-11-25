"""Single source of truth for the dataset schema (used by validation, training and the API)."""

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

_YES_NO = ["No", "Yes"]
_INTERNET_ADDON = ["No", "Yes", "No internet service"]

CATEGORY_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": _YES_NO,
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

CATEGORICAL_FEATURES = list(CATEGORY_VALUES)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
