import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

hf_subtype_path = PULL_2023 / "Amyloidosis Patients HF_Subtype 2023"

def csv_to_parquet(path: Path = hf_subtype_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): hf_subtype file path. Defaults to hf_subtype_path.
    """
    # Load hf_subtype
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.HFrecEF_followupecho = pd.to_datetime(df.HFrecEF_followupecho)
    
    remaining_columns = [c for c in df.columns if c not in ["ir_id", "HFrecEF_followupecho"]]
    for column in remaining_columns:
        if "date" in column.lower():
            # rounding to microseconds because some of the cols in this file are in nanoseconds, 
            # and this causes an issue with pyarrow
            df[column] = pd.to_datetime(df[column]).dt.round('us')
        elif "code" in column.lower():
            df[column] = df[column].astype("string")
        else:
            df[column] = df[column].astype("Int64")

    # Save as parquet
    df.to_parquet(hf_subtype_path.with_suffix(".parquet"))


def load_hf_subtype(path: Path = hf_subtype_path) -> pd.DataFrame:
    """Reads hf_subtype parquet into dataframe

    Args:
        path (Path, optional): hf_subtype file path. Defaults to hf_subtype_path.

    Returns:
        pd.DataFrame: hf_subtype dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load hf_subtype csv and save as parquet
    csv_to_parquet(hf_subtype_path)
