import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

outpt_encounters_path = PULL_2023 / "Amyloidosis Patients Outpt Clinic Encounters 2023"


def csv_to_parquet(path: Path = outpt_encounters_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): outpt encounters file path. Defaults to outpt_encounters_path.
    """
    # Load outpt_encounters
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    integers = [
        "ir_id",
        "enc_type",
        "enc_id",
        "encounter_outpatient_key",
        "Cards_encounter_filter",
        "PCP_encounter_filter",
        "pregnancy_flag"
    ]
    strings = [
        "telehealth_reason", "Telehealth_Visit_type"
    ]
    floats = ["height", "weight", "bmi"]

    for column in df.columns:
        # dates as datetime
        if column in integers:
            df[column] = df[column].astype(int)
        elif "date" in column.lower():
            df[column] = pd.to_datetime(df[column])
        # ICD codes as string
        elif "code" in column.lower() or column.isdigit() or column in strings:
            df[column] = df[column].astype("string")
        elif column in floats:
            df[column] = df[column].astype(float)
        else:
            df[column] = df[column].astype("Int64")

    # Save as parquet
    df.to_parquet(outpt_encounters_path.with_suffix(".parquet"))


def load_outpt_encounters(path: Path = outpt_encounters_path) -> pd.DataFrame:
    """Reads outpt encounters parquet into dataframe

    Args:
        path (Path, optional): outpt encounters file path. Defaults to outpt_encounters_path.

    Returns:
        pd.DataFrame: outpt encounters dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load outpt encounters csv and save as parquet
    csv_to_parquet(outpt_encounters_path)
