import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

labeled_cohort_file_path = PULL_2023 / "Amyloidosis Patients Cohort Entry - Labeled"


def csv_to_parquet(path: Path = labeled_cohort_file_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): labeled cohort file path. Defaults to labeled_cohort_file_path.
    """
    # Load labeled cohort file
    df = pd.read_csv(path.with_suffix(".csv"))
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.Age_cohort = df.Age_cohort.astype(int)
    df.Gender_EDW = df.Gender_EDW.astype("string")
    df.Race_EDW = df.Race_EDW.astype("string")
    df.Ethnicity_EDW = df.Ethnicity_EDW.astype("string")
    df.race_ethncty_combined = df.race_ethncty_combined.astype("string")
    df.Insurance_EDW_cohort = df.Insurance_EDW_cohort.astype("string")
    df.Insurance_Mapped_cohort = df.Insurance_Mapped_cohort.astype("string")

    # columns with improper names (e.g. date not in name if date)
    df.HFrecEF_followupecho = pd.to_datetime(df.HFrecEF_followupecho)
    
    # flags for label sources and chart review status
    df.full_chart_review = df.full_chart_review.astype(int)
    df.label__chart_review = df.label__chart_review.astype(int)
    df.pyp_or_tafamidis_only = df.pyp_or_tafamidis_only.astype(int)
    df.label__definitive = df.label__definitive.astype(int)
    df.label__missing_diagnosis = df.label__missing_diagnosis.astype(int)
    
    remaining_columns = [
        c
        for c in df.columns
        if c
        not in [
            "ir_id",
            "Age_cohort",
            "Gender_EDW",
            "Race_EDW",
            "Ethnicity_EDW",
            "race_ethncty_combined",
            "Insurance_EDW_cohort",
            "Insurance_Mapped_cohort",
            "HFrecEF_followupecho",
            "full_chart_review",
            "label__chart_review",
            "pyp_or_tafamidis_only",
            "label__definitive",
            "label__missing_diagnosis",
        ]
    ]
    for column in remaining_columns:
        if "date" in column.lower():
            # rounding to microseconds because some of the cols in this file are in nanoseconds,
            # and this causes an issue with pyarrow
            df[column] = pd.to_datetime(df[column]).dt.round("us")
        elif any(word in column.lower() for word in ["code", "label"]):
            df[column] = df[column].astype("string")
        elif "cohort_entry" in column.lower():
            df[column] = df[column].astype("Int64")
        elif "patient_group" in column.lower():
            df[column] = df[column].astype(bool)
        else:
            df[column] = df[column].astype("Int64")

    # Save as parquet
    df.to_parquet(labeled_cohort_file_path.with_suffix(".parquet"))


def load_labeled_cohort(path: Path = labeled_cohort_file_path) -> pd.DataFrame:
    """Reads labeled cohort file parquet into dataframe

    Args:
        path (Path, optional): labeled cohort file path. Defaults to labeled_cohort_file_path.

    Returns:
        pd.DataFrame: labeled cohort file dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load labeled cohort file csv and save as parquet
    csv_to_parquet(labeled_cohort_file_path)
