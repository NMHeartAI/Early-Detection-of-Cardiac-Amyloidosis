import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

comorbitities_path = PULL_2023 / "Amyloidosis Patients Comorbidities 2023"


def csv_to_parquet(path: Path = comorbitities_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): comorbitities file path. Defaults to comorbitities_path.
    """
    # Load comorbitities
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.smoking_sh = df.smoking_sh.astype("string")
    
    remaining_columns = [c for c in df.columns if c not in ["ir_id", "smoking_sh"]]
    for column in remaining_columns:
        if "date" in column.lower():
            df[column] = pd.to_datetime(df[column])
        else:
            df[column] = df[column].astype("Int64")

    # Save as parquet
    df.to_parquet(comorbitities_path.with_suffix(".parquet"))


def load_comorbitities(path: Path = comorbitities_path) -> pd.DataFrame:
    """Reads comorbitities parquet into dataframe

    Args:
        path (Path, optional): comorbitities file path. Defaults to comorbitities_path.

    Returns:
        pd.DataFrame: comorbitities dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load comorbitities csv and save as parquet
    csv_to_parquet(comorbitities_path)
