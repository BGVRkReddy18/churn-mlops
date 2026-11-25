from churn.monitoring.drift import drift_report, psi_categorical, psi_numeric


def test_psi_is_zero_for_identical_data(clean_df):
    assert psi_numeric(clean_df["tenure"], clean_df["tenure"]) < 1e-6
    assert psi_categorical(clean_df["Contract"], clean_df["Contract"]) < 1e-6


def test_drift_detected_on_shifted_data(clean_df):
    shifted = clean_df.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 2
    shifted["Contract"] = "Month-to-month"
    report = drift_report(clean_df, shifted)
    assert report["drift_detected"]
    assert {"MonthlyCharges", "Contract"} <= set(report["drifted_features"])
    assert "gender" not in report["drifted_features"]
