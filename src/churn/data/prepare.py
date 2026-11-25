"""Stage 2 - clean, validate and split the data."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import PROCESSED_DIR, RAW_DATA_PATH, TEST_PATH, TRAIN_PATH, load_params
from churn.schema import CATEGORY_VALUES, FEATURES, ID_COLUMN, NUMERIC_FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the data does not match the expected schema."""


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[ID_COLUMN], errors="ignore")
    # TotalCharges is a string with blanks for brand-new customers (tenure 0) -> 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # SeniorCitizen is 0/1 while every other flag is Yes/No - make it consistent
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail fast if the data breaks the schema, instead of silently training on bad data."""
    errors: list[str] = []

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")

    null_counts = df[FEATURES + [TARGET]].isna().sum()
    for col, n in null_counts[null_counts > 0].items():
        errors.append(f"{col}: {n} null values")

    for col, allowed in CATEGORY_VALUES.items():
        unexpected = set(df[col].dropna().unique()) - set(allowed)
        if unexpected:
            errors.append(f"{col}: unexpected values {sorted(unexpected)}")

    for col in NUMERIC_FEATURES:
        if (df[col] < 0).any():
            errors.append(f"{col}: negative values")

    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        errors.append(f"{TARGET}: must be binary 0/1")

    if errors:
        raise DataValidationError("Data validation failed:\n  " + "\n  ".join(errors))


def main() -> None:
    params = load_params()["data"]
    raw = pd.read_csv(RAW_DATA_PATH)
    log.info("Loaded %d rows x %d columns", *raw.shape)

    df = clean(raw)
    validate(df)
    log.info("Validation passed. Churn rate: %.1f%%", 100 * df[TARGET].mean())

    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[TARGET],
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    log.info("Train: %d rows, Test: %d rows", len(train), len(test))


if __name__ == "__main__":
    main()
