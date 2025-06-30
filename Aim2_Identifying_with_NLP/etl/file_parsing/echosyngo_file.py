import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

echosyngo_path = PULL_2023 / "Amyloidosis Patients EchoSyngo 2023"


def csv_to_parquet(path: Path = echosyngo_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): echosyngo file path. Defaults to echosyngo_path.
    """
    # Load echosyngo
    df = pd.read_csv(path.with_suffix(".csv"), sep="|", low_memory=False)
    # drop last 2 rows because they contain SQL info
    df.drop(df.tail(2).index, inplace=True)

    # Save as parquet
    # df.to_parquet(echosyngo_path.with_suffix(".parquet"))
    pass


def load_echosyngo(path: Path = echosyngo_path) -> pd.DataFrame:
    """Reads echosyngo parquet into dataframe

    Args:
        path (Path, optional): echosyngo file path. Defaults to echosyngo_path.

    Returns:
        pd.DataFrame: echosyngo dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    # return df
    pass


if __name__ == "__main__":
    # Load echosyngo csv and save as parquet
    csv_to_parquet(echosyngo_path)
