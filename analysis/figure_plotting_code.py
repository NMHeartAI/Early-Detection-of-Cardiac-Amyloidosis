from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")


def fig_pr_auc(
    df: pd.DataFrame, conf_int: List[Tuple[float, float]], FIGDST: Path | None = None
) -> plt.Figure:
    """PR_AUC plotting code.

    Args:
        df (pd.DataFrame): _description_
        conf_int (List[Tuple[float, float]]): Confidence intervals to display
        FIGDST (Path, optional): If set, saves to FIGDST/pr_curves_whole_CI.tiff. Defaults to None.

    Returns:
        plt.Figure: PR_AUC figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), layout="constrained")
    name_map = {
        "Pfizer": "Huda et al.",
        "Mayo Score": "Mayo ATTR-CM Score",
        "Echonet-LVH": "EchoNet-LVH",
        "Ultromics": "EchoGo Amyloidosis",
    }

    # Loop over models and predictions to generate PR_AUC curves
    for name, y_pred in [
        ("Pfizer", np.vstack(df.pfizer_prediction.values)[:, 1]),
        ("Mayo Score", (df.mayo_score / 10).values),
        ("Echonet-LVH", (df.echonet_prediction).values),
        ("Ultromics", (df.ultromics_prediction).values),
    ]:
        y_true = df.true_label.notna().values
        viz = PrecisionRecallDisplay.from_predictions(
            y_true,
            y_pred,
            name=name_map[name],
            ax=ax,
        )
        y_metric_arr = []

        # Bootstrap CI
        for _ in range(200):
            sample = np.random.choice(range(len(y_pred)), len(y_pred), replace=True)
            y_metric, x_metric, thresholds = precision_recall_curve(
                y_true[sample], y_pred[sample]
            )
            y_metric_arr.append(
                np.interp(viz.recall[::-1], x_metric[::-1], y_metric[::-1])
            )

        y_metric_upper = np.percentile(y_metric_arr, 97.5, axis=0)[::-1]
        y_metric_lower = np.percentile(y_metric_arr, 2.5, axis=0)[::-1]
        ax.fill_between(viz.recall, y_metric_lower, y_metric_upper, alpha=0.15)

    ax.set_ylabel(ax.get_ylabel().split(" ")[0], fontsize=14)
    ax.set_xlabel(ax.get_xlabel().split(" ")[0], fontsize=14)
    L = ax.legend(loc="upper right", fontsize=10, frameon=True)

    # Update label text using CI for models in desired format
    for t, CI in zip(L.texts, conf_int):
        new_text = (
            t.get_text()
            .replace("(", "[")
            .replace(")", f", 95% CI: {CI[0]:0.2f} - {CI[1]:0.2f}]")
        )
        t.set_text(new_text)
    if FIGDST:
        fig.savefig(FIGDST / "pr_curves_whole_CI.tiff", dpi=300)
    return fig


def fig_roc_auc(
    df: pd.DataFrame, conf_int: List[Tuple[float, float]], FIGDST: Path = None
) -> plt.Figure:
    """ROC_AUC plotting code.

    Args:
        df (pd.DataFrame): _description_
        conf_int (List[Tuple[float, float]]): Confidence intervals to display
        FIGDST (Path, optional): If set, saves to FIGDST/roc_curves_whole_CI.tiff. Defaults to None.

    Returns:
        plt.Figure: ROC_AUC figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), layout="constrained")
    name_map = {
        "Pfizer": "Huda et al.",
        "Mayo Score": "Mayo ATTR-CM Score",
        "Echonet-LVH": "EchoNet-LVH",
        "Ultromics": "EchoGo Amyloidosis",
    }

    # Loop over models and predictions to generate ROC_AUC curves
    for name, y_pred in [
        ("Pfizer", np.vstack(df.pfizer_prediction.values)[:, 1]),
        ("Mayo Score", (df.mayo_score / 10).values),
        ("Echonet-LVH", (df.echonet_prediction).values),
        ("Ultromics", (df.ultromics_prediction).values),
    ]:
        y_true = df.true_label.notna().values
        viz = RocCurveDisplay.from_predictions(
            y_true,
            y_pred,
            name=name_map[name],
            ax=ax,
        )

        # Bootstrap CI
        y_metric_arr = []
        for _ in range(200):
            sample = np.random.choice(range(len(y_pred)), len(y_pred), replace=True)
            x_metric, y_metric, thresholds = roc_curve(y_true[sample], y_pred[sample])
            y_metric_arr.append(np.interp(viz.fpr, x_metric, y_metric))

        y_metric_upper = np.percentile(y_metric_arr, 97.5, axis=0)
        y_metric_lower = np.percentile(y_metric_arr, 2.5, axis=0)
        ax.fill_between(viz.fpr, y_metric_lower, y_metric_upper, alpha=0.15)

    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    L = ax.legend(loc="lower right", fontsize=10, frameon=True)

    # Update label text using CI for models in desired format
    for t, CI in zip(L.texts, conf_int):
        new_text = (
            t.get_text()
            .replace("(", "[")
            .replace(")", f", 95% CI: {CI[0]:0.2f} - {CI[1]:0.2f}]")
        )
        t.set_text(new_text)

    if FIGDST:
        fig.savefig(FIGDST / "roc_curves_whole_CI.tiff", dpi=300)
    return fig
