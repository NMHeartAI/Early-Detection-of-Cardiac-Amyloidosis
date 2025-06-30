import csv
import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
BASE = Path("/data/datasets/Amyloidosis/")
PULL_2023 = BASE / "2023 pull"

cardiac_mri_path = PULL_2023 / "Amyloidosis Patients Cardiac MRI 2023"


def csv_to_parquet(path: Path = cardiac_mri_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): Cardiac MRIs File Path. Defaults to cardiac_mri_path.
    """
    # Load Cardiac MRIs

    header = pd.read_csv(path.with_suffix(".csv"), sep="|", nrows=1)
    names = [name.strip() for name in list(header.columns)]
    assert names == [
        "ir_id",
        "procedure_name",
        "Cardiac_MRI_date",
        "Cardiac_MRI_text",
    ], "check header"

    # Read notes
    df = pd.read_csv(
        path.with_suffix(".csv"),
        sep="|",
        on_bad_lines="warn",
        header=None,
        names=names,
        skiprows=1,
        engine="python",
        index_col=False,
        quoting=3, 
    )

    # drop the last 2 rows
    df.drop(df.tail(2).index, inplace=True)

    # set column dtypes
    df.ir_id = df.ir_id.astype(int)
    df.procedure_name = df.procedure_name.astype("string")
    df.Cardiac_MRI_date = pd.to_datetime(df.Cardiac_MRI_date).dt.date
    df.Cardiac_MRI_text = df.Cardiac_MRI_text.astype("string")

    # sort chronologically so that the aggregation gives us a list of notes information in chronological order
    df.sort_values(by=["ir_id", "Cardiac_MRI_date"], inplace=True)
    # Save as parquet
    df.to_parquet(path.with_suffix(".parquet"))


def load_cardiac_mris(
    path: Path = cardiac_mri_path,
) -> pd.DataFrame:
    """Reads Cardiac MRIs parquet into dataframe

    Args:
        path (Path, optional): Cardiac MRIs file path. Defaults to cardiac_mri_path.

    Returns:
        pd.DataFrame: Cardiac MRIs dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load Cardiac MRIs csv and save as parquet
    csv_to_parquet(cardiac_mri_path)
