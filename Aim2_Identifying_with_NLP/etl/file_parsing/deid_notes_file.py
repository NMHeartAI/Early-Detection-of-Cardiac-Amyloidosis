import pandas as pd
import numpy as np
from pathlib import Path

# The path to the Amyloid data
DATASET_PATH = Path("/data/datasets/Amyloidosis/")

# clinical notes
notes_path = DATASET_PATH / "Amyloidosis Patients OutpatientNotesDeid"


def csv_to_parquet(path: Path = notes_path) -> None:
    """Reads CSV, sets dtypes and converts to parquet

    Args:
        path (Path, optional): Clinical Notes File Path. Defaults to notes_path.
    """
    # Load DEID Notes
    df = pd.read_csv(path.with_suffix(".csv"))
    df.created_date_key = pd.to_datetime(df.created_date_key)
    df.deid_note_text = df.deid_note_text.astype("string")
    # sort chronologically so that the aggregation gives us a list of notes information in chronological order
    df.sort_values(by=["ir_id", "created_date_key"], inplace=True)
    # Save as parquet
    df.to_parquet(path.with_suffix(".parquet"))


def load_notes(path: Path = notes_path) -> pd.DataFrame:
    """Reads clinical notes parquet into dataframe

    Args:
        path (Path, optional): Clinical notes file path. Defaults to notes_path.

    Returns:
        pd.DataFrame: Clinical notes dataframe
    """
    df = pd.read_parquet(path.with_suffix(".parquet"))
    return df


if __name__ == "__main__":
    # Load deid clinical notes csv and save as parquet
    csv_to_parquet(notes_path)
