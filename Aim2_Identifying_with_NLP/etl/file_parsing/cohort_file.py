import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

cohort_entry_file_path = PULL_2023 / "Amyloidosis Patients Cohort Entry 2023"


def csv_to_parquet(path: Path = cohort_entry_file_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): cohort_entry file path. Defaults to cohort_entry_file_path.
    """
    # Load cohort_entry file
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)
    
    # Set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.HF_cohort_entry = df.HF_cohort_entry.astype("Int64")
    df.HF_cohort_entry_date = pd.to_datetime(df.HF_cohort_entry_date)
    df.CA_cohort_entry = df.CA_cohort_entry.astype("Int64")
    df.CA_cohort_entry_date = pd.to_datetime(df.CA_cohort_entry_date)
    df.CM_cohort_entry = df.CM_cohort_entry.astype("Int64")
    df.CM_cohort_entry_date = pd.to_datetime(df.CM_cohort_entry_date)
    df.PYP_cohort_entry = df.PYP_cohort_entry.astype("Int64")
    df.PYP_cohort_entry_date = pd.to_datetime(df.PYP_cohort_entry_date)
    df.Tafamidis_cohort_entry = df.Tafamidis_cohort_entry.astype("Int64")
    df.Tafamidis_cohort_entry_date = pd.to_datetime(df.Tafamidis_cohort_entry_date)
    df.cMRI_cohort_entry = df.cMRI_cohort_entry.astype("Int64")
    df.cMRI_cohort_entry_date = pd.to_datetime(df.cMRI_cohort_entry_date)
    df.HF_stricter_definition_date = pd.to_datetime(df.HF_stricter_definition_date)

    # Save as parquet
    df.to_parquet(cohort_entry_file_path.with_suffix(".parquet"))


def load_cohort_entry(path: Path = cohort_entry_file_path) -> pd.DataFrame:
    """Reads cohort_entry file parquet into dataframe

    Args:
        path (Path, optional): cohort_entry file path. Defaults to cohort_entry_file_path.

    Returns:
        pd.DataFrame: cohort entry file dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load cohort entry file csv and save as parquet
    csv_to_parquet(cohort_entry_file_path)
