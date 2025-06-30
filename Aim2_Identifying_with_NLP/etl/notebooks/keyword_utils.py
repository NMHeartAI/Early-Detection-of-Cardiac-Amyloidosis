from typing import List
import pandas as pd


def keyword_lookup(x: str, keywords):
    return any(word in x.lower() for word in keywords)


def find_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    column_name: str,
    example_column_name: str = "example",
):
    df[column_name] = df[example_column_name].apply(
        lambda x: keyword_lookup(x, keywords)
    )
    return df


bkr = ["bkr"]
formal = ["formal diagnosis"]
attr = [
    "transthyretin",
    "prealbumin",
    "attr",
    "amyloidosis, al (lambda) type",
    "lambda light chain",
    "kappa light chain",
    "light chain",
]

AMYLOID_KEYWORDS = {
    "amyloid": ["amyloid", "amyloidosis", "cardiac amyloid", "cardiac amyloidosis"],
    "ttr": [
        "attr",
        " ttr",
        "transthyretin",
        "prealbumin",
        "httr",
        "hereditary",
        "familial",
        "wttr",
        "wild type",
        "wild-type",
        "wtATTR",
        "hattr",
    ],
    "al": [
        " al",
        "lambda light chain",
        "light chain",
        "kappa light chain",
        "al type",
        "al (lambda) type",
        "al (lambda)-type",
        "al lambda type",
        "lambda",
        "al "
    ],
    "wttr": [
        "wild type",
        "wild-type",
        "wttr",
        "wtATTR",
        "agerelated",
        "age-related",
        "age related",
        "senile",
    ],
    "httr": ["hereditary", "httr", "hattr", "familial"],
    "congo red stain": ["congo", "congo red", "congo red stain"],
    "heart biopsy": [
        "endomyocardial tissue",
        "cardiac",
        "endomyocardial biopsy",
        "endomyocardium",
    ],
    "not heart biopsy": ["autopsy", "valve"],
}


