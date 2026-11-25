import pytest

from churn.data.prepare import DataValidationError, clean, validate
from churn.schema import TARGET


def test_clean_handles_blank_total_charges(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_clean_maps_senior_and_target(raw_df):
    df = clean(raw_df)
    assert set(df["SeniorCitizen"]) <= {"No", "Yes"}
    assert set(df[TARGET]) <= {0, 1}
    assert "customerID" not in df.columns


def test_validate_passes_on_clean_data(clean_df):
    validate(clean_df)


def test_validate_rejects_unknown_category(clean_df):
    clean_df.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(DataValidationError, match="Contract"):
        validate(clean_df)


def test_validate_rejects_missing_column(clean_df):
    with pytest.raises(DataValidationError, match="tenure"):
        validate(clean_df.drop(columns=["tenure"]))


def test_validate_rejects_negative_numbers(clean_df):
    clean_df.loc[0, "MonthlyCharges"] = -5
    with pytest.raises(DataValidationError, match="MonthlyCharges"):
        validate(clean_df)
