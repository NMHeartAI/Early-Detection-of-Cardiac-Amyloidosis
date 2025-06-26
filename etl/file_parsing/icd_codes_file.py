import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

icd_codes_path = PULL_2023 / "Amyloidosis Patients ICD Codes 2023"


def csv_to_parquet(path: Path = icd_codes_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): icd codes file path. Defaults to icd_codes_path.
    """
    # Load icd codes
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.ICD_code = df.ICD_code.astype("string")
    df.ICD_code_type = df.ICD_code_type.astype("string")
    df.ICD_code_source = df.ICD_code_source.astype("string")
    if "consolidated_encounter_key" in df.columns:
        df.consolidated_encounter_key = df.consolidated_encounter_key.astype("Int64")
    df.ICD_code_date = pd.to_datetime(df.ICD_code_date)
    df.ICD_code_setting = df.ICD_code_setting.astype("string")

    # Save as parquet
    df.to_parquet(icd_codes_path.with_suffix(".parquet"))


def load_icd_codes(path: Path = icd_codes_path) -> pd.DataFrame:
    """Reads icd codes parquet into dataframe

    Args:
        path (Path, optional): icd codes file path. Defaults to icd_codes_path.

    Returns:
        pd.DataFrame: icd codes dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load icd codes csv and save as parquet
    csv_to_parquet(icd_codes_path)
