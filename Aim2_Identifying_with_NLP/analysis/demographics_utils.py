from typing import List
import pandas as pd
from collections.abc import Callable
from scipy import stats


def _mean_sd_fmt(group: pd.Series) -> str:
    # Formatting helper
    return f"{group['mean']:.1f} ({group['std']:.1f})"


def _mean_sd_fmt_2(group: pd.Series) -> str:
    # Formatting helper with more decimal places
    return f"{group['mean']:.2f} ({group['std']:.2f})"


def sorter(idx, categories: List[str] = None) -> dict:
    # Util to re-sort pd.MultiIndex
    #   most useful for LVH categories
    if not categories:
        categories = ["Normal", "Mild", "Moderate", "Severe"]
    return idx.map({v: k for k, v in enumerate(categories)})


def numerical(
    col: str, df: pd.DataFrame, missing_pred: str = None, fmt: Callable = _mean_sd_fmt
) -> pd.DataFrame:
    """Formatting of numerical columns where we are interested in
        mean and standard deviation.

    Args:
        col (str): Column with information
        df (pd.DataFrame): dataframe
        missing_pred (str, optional): Which category to use of additional
            aggregation level. Column should be True when prediction was
            made, False if model failed to produce a prediction.
            Defaults to None.
        fmt (Callable, optional): How to format result

    Returns:
        pd.DataFrame: Formatted df.
    """
    result = []
    for mask, name in [
        ((df.ttr_ca == 1.0), "Cases"),
        ((df.ttr_ca == 0.0), "Controls"),
        ((df.index == df.index), "Overall"),
    ]:
        if missing_pred:
            if name in ["Overall"]:
                continue
            for missing_mask, cat in [
                (df[missing_pred] == True, "_present"),
                (df[missing_pred] == False, "_missing"),
            ]:
                result.append(
                    df.loc[mask & missing_mask, col]
                    .describe()
                    .loc[["mean", "std"]]
                    .rename(
                        name + cat + f" (n={df.loc[mask & missing_mask, col].shape[0]})"
                    )
                )
        else:
            result.append(
                df.loc[mask, col]
                .describe()
                .loc[["mean", "std"]]
                .rename(name + f"(n={mask.sum()})")
            )
    _temp = pd.DataFrame(
        result,
    ).T.apply(fmt, axis=0)

    # We do not compute p-values for the supplemental tables of
    #   demographics for specific models successful models.
    if not missing_pred:
        pval = stats.ttest_ind(
            df.loc[df.ttr_ca == 0.0, col],
            df.loc[df.ttr_ca == 1.0, col],
            equal_var=False,
        ).pvalue
        _temp["p-value"] = f"{pval:.3f}" if pval > 1e-4 else "<0.0001"
    return pd.DataFrame([_temp]).set_index(
        pd.MultiIndex.from_tuples([(f"{col}, mean(sd)", "")])
    )


def categorical_var(col: str, df: pd.DataFrame, missing_pred=None) -> pd.DataFrame:
    """
    Formatting of categorical columns for demographic table where
        the desired output is number and percentage per category.
        Often will have defined cutoffs.

    Args:
        col (str): Column with information
        df (pd.DataFrame): dataframe
        missing_pred (str, optional): Which category to use of
            additional aggregation level. Column should be True
            when prediction was made, False if model failed to
            produce a prediction. Defaults to None.

    Returns:
        pd.DataFrame: Formatted df
    """
    result = []
    for mask, name in [
        ((df.ttr_ca == 1.0), "Cases"),
        ((df.ttr_ca == 0.0), "Controls"),
        ((df.index == df.index), "Overall"),
    ]:
        if missing_pred:
            if name in ["Overall"]:
                continue
            for missing_mask, cat in [
                (df[missing_pred] == True, "_present"),
                (df[missing_pred] == False, "_missing"),
            ]:
                val_cnts = df.loc[mask & missing_mask, col].value_counts()
                if name == "Cases":
                    f_obs = val_cnts
                elif name == "Controls":
                    f_exp = val_cnts
                record = {
                    key: f"{val} ({val/df.loc[mask & missing_mask].shape[0]*100:.1f})"
                    for key, val in val_cnts.items()
                }
                record[col] = f"{name+cat} (n={(mask & missing_mask).sum()})"
                result.append(record)
        else:
            val_cnts = df.loc[mask, col].value_counts()
            if name == "Cases":
                f_obs = val_cnts
            elif name == "Controls":
                f_exp = val_cnts
            record = {
                key: f"{val} ({val/val_cnts.sum()*100:.1f}%)"
                for key, val in val_cnts.items()
            }
            record[col] = f"{name}(n={mask.sum()})"
            result.append(record)
    _temp = pd.DataFrame(result).set_index(col).T.sort_index()
    if not missing_pred:
        pval = stats.chisquare(
            f_obs=f_obs, f_exp=f_exp / f_exp.sum() * f_obs.sum()
        ).pvalue
        _temp.loc[_temp.index[0], "p-value"] = (
            f"{pval:.3f}" if pval > 1e-4 else "< 0.0001"
        )
    _temp.index = pd.MultiIndex.from_tuples(
        (f"{col}, N (%)", ind) for ind in _temp.index
    )

    return _temp
