import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

demographics_path = PULL_2023 / "Amyloidosis Patients Demographics 2023"


def csv_to_parquet(path: Path = demographics_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): demographics file path. Defaults to demographics_path.
    """
    # Load demographics
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.Age_cohort = df.Age_cohort.astype(int)
    df.Gender_EDW = df.Gender_EDW.astype("string")
    df.Race_EDW = df.Race_EDW.astype("string")
    if "Ethnicity_EDW" in df.columns:
        df.Ethnicity_EDW = df.Ethnicity_EDW.astype("string")
    df.race_ethncty_combined = df.race_ethncty_combined.astype("string")
    df.Insurance_EDW_cohort = df.Insurance_EDW_cohort.astype("string")
    df.Insurance_Mapped_cohort = df.Insurance_Mapped_cohort.astype("string")

    # Save as parquet
    df.to_parquet(demographics_path.with_suffix(".parquet"))


def load_demographics(path: Path = demographics_path) -> pd.DataFrame:
    """Reads demographics parquet into dataframe

    Args:
        path (Path, optional): demographics file path. Defaults to demographics_path.

    Returns:
        pd.DataFrame: demographics dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load demographics csv and save as parquet
    csv_to_parquet(demographics_path)
