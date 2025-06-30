from enum import Enum
from pathlib import Path
import pandas as pd

from text_processing import clean_cardiac_path, clean_pyp

DATASET_PATH = Path("/data/datasets/Amyloidosis/datasets/")
ANNOTATIONS_PATH = Path("/data/datasets/Amyloidosis/annotations/")
PATIENT_DIAGNOSIS_PATH = Path("/data/datasets/Amyloidosis/patient_amyloid_diagnosis/")


class Datasets(str, Enum):
    CARDIAC_PATH_REPORTS = "cardiac_path_reports"
    PYP_REPORTS = "pyp_reports"
    MAYO_LABS = "mayo_labs"
    HF_SUBTYPE = "hf_subtype"


dataset_config_mapping = {
    "hf_subtype": {
        "dataset_name": "hf_subtype",
        "path": DATASET_PATH / "Amyloidosis Patients HF_Subtype.csv",
        "sep": "|",
        "encoding": "utf8",
    },
    "pyp_reports": {
        "dataset_name": "pyp_reports",
        "path": DATASET_PATH / "pyp_reports" / "Amyloidosis Patients PYP Reports.csv",
        "document": "reg1",
        "date": "created_date_key",
        "annotations": ANNOTATIONS_PATH / "pyp_reports" / "pyp_annotations_v6.csv",
        "patient_diagnosis": PATIENT_DIAGNOSIS_PATH
        / "pyp_reports"
        / "patient_amyloid_diagnosis.csv",
        "dataset_prefix": "pyp",
    },
    "cardiac_path_reports": {
        "dataset_name": "cardiac_path_reports",
        "path": DATASET_PATH
        / "cardiac_path_reports"
        / "Amyloidosis Patients Cardiac Path.csv",
        "document": "cardiac_path_report",
        "date": "report_date",
        "annotations": ANNOTATIONS_PATH
        / "cardiac_path_reports"
        / "cardiac_path_annotations_v12.csv",
        "patient_diagnosis": PATIENT_DIAGNOSIS_PATH
        / "cardiac_path_reports"
        / "patient_amyloid_diagnosis.csv",
        "dataset_prefix": "cp",
    },
    "mayo_labs": {
        "dataset_name": "mayo_labs",
        "path": DATASET_PATH
        / "mayo_labs"
        / "Amyloidosis Patients Mayo Lab Results - GROUPED - cp_pyp labels.csv",
        "document": "result_note",
        "date": "order_date_key",
        "annotations": ANNOTATIONS_PATH / "mayo_labs" / "mayo_labs_annotations_v14.csv",
        "dataset_prefix": "mayo",
    },
}


def load_dataset(dataset: Datasets) -> pd.DataFrame:
    """Reads dataset csv and outputs dataframe

    Args:
        dataset (Datasets): "cardiac_path_reports", "pyp_reports", "mayo_labs", or "hf_subtype"

    Returns:
        pd.DataFrame: dataframe of the dataset
    """
    # check if dataset has been preprocessed
    dataset_preprocessed = [
        Datasets.CARDIAC_PATH_REPORTS,
        Datasets.PYP_REPORTS,
        Datasets.MAYO_LABS,
    ]
    if dataset not in dataset_preprocessed:
        raise Exception(
            f"Dataset not preprocessed: {dataset_config_mapping[dataset]['dataset_name']}"
        )
    document = dataset_config_mapping[dataset]["document"]
    date = dataset_config_mapping[dataset]["date"]

    df = pd.read_csv(dataset_config_mapping[dataset]["path"])
    df["document_ID"] = pd.to_numeric(df["document_ID"])
    df["ir_id"] = pd.to_numeric(df["ir_id"])
    df[date] = pd.to_datetime(df[date])

    if dataset == Datasets.CARDIAC_PATH_REPORTS:
        df["text"] = df[document].apply(lambda x: clean_cardiac_path(x))
    elif dataset == Datasets.PYP_REPORTS:
        df["text"] = df[document].apply(lambda x: clean_pyp(x))

    """ 
    # this code is for mayo_labs but will change when they have been preprocessed
    else:
        # read entries, keep relevant columns, change column names
        df = pd.read_csv(
            dataset_config_mapping[dataset]["path"],
            sep=dataset_config_mapping[dataset]["sep"],
            encoding=dataset_config_mapping[dataset]["encoding"],
            error_bad_lines=False,
        )
        # drop last 2 rows
        df.drop(df.tail(2).index, inplace=True)
    # read source of labels and merge
    """

    return df


def load_annotations(dataset: Datasets) -> pd.DataFrame:
    """loads annotations for a dataset

    Args:
        dataset (Datasets): "cardiac_path_reports", "pyp_reports", "mayo_labs", or "hf_subtype"

    Returns:
        pd.DataFrame: dataframe of annotations
    """
    # check if dataset has annotations
    annotations_available = [
        Datasets.CARDIAC_PATH_REPORTS,
        Datasets.PYP_REPORTS,
        Datasets.MAYO_LABS,
    ]
    if dataset not in annotations_available:
        raise Exception(
            f"No annotations for the dataset: {dataset_config_mapping[dataset]['dataset_name']}"
        )

    # read entries, keep relevant columns, change column types
    date = dataset_config_mapping[dataset]["date"]

    df = pd.read_csv(dataset_config_mapping[dataset]["annotations"])
    df["ir_id"] = pd.to_numeric(df["ir_id"])
    df["document_ID"] = pd.to_numeric(df["document_ID"])
    df[date] = pd.to_datetime(df[date])

    return df


def load_patient_diagnosis(dataset: Datasets) -> pd.DataFrame:
    """loads patient level diagnosis for a dataset

    Args:
        dataset (Datasets): "cardiac_path_reports", "pyp_reports", "mayo_labs", or "hf_subtype"

    Returns:
        pd.DataFrame: dataframe of patient diagnosis
    """
    # check if dataset has annotations
    diagnosis_available = [Datasets.CARDIAC_PATH_REPORTS, Datasets.PYP_REPORTS]
    if dataset not in diagnosis_available:
        raise Exception(
            f"No diagnosis data for the dataset: {dataset_config_mapping[dataset]['dataset_name']}"
        )

    # read entries, keep relevant columns, change column types
    date = dataset_config_mapping[dataset]["date"]
    dataset_prefix = dataset_config_mapping[dataset]["dataset_prefix"]
    diagnosis = f"{dataset_prefix}__amyloid_diagnosis"
    diagnosis_date = f"{dataset_prefix}__amyloid_diagnosis_date"

    df = pd.read_csv(dataset_config_mapping[dataset]["patient_diagnosis"])
    df["ir_id"] = pd.to_numeric(df["ir_id"])
    df[diagnosis] = pd.to_numeric(df[diagnosis])
    df[diagnosis_date] = pd.to_datetime(df[diagnosis_date])

    return df
