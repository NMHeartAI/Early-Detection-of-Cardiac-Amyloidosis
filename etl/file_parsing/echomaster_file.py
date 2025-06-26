import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

echomaster_path = PULL_2023 / "Amyloidosis Patients EchoMaster 2023"


def csv_to_parquet(path: Path = echomaster_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): echomaster file path. Defaults to echomaster_path.
    """
    # Load echomaster
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Set column dtypes
    df.patient_ir_id = df.patient_ir_id.astype(int)
    df.master_echo_id = df.master_echo_id.astype(int)
    df.echo_date = pd.to_datetime(df.echo_date)
    df.echo_description = df.echo_description.astype("string")
    df.echo_type = df.echo_type.astype("string")
    df.accession_num = df.accession_num.astype("string")
    df.study_uid = df.study_uid.astype("string")
    df.department = df.department.astype("string")
    df.doppler = df.doppler.astype("Int64")
    df.limited_echo = df.limited_echo.astype(int)
    df.echo_extractor_id = df.echo_extractor_id.astype("Int64")

    df.echo_type = df.echo_type.str.replace("  ", " ").str.strip()
    df.rename(columns={"patient_ir_id": "ir_id"}, inplace=True)
    # Save as parquet
    df.to_parquet(echomaster_path.with_suffix(".parquet"))


def load_echomaster(path: Path = echomaster_path) -> pd.DataFrame:
    """Reads echomaster parquet into dataframe

    Args:
        path (Path, optional): echomaster file path. Defaults to echomaster_path.

    Returns:
        pd.DataFrame: echomaster dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load echomaster csv and save as parquet
    csv_to_parquet(echomaster_path)
