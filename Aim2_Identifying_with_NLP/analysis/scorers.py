from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
    auc,
    confusion_matrix,
    multilabel_confusion_matrix,
)
import numpy as np
from functools import partial
from typing import List, Callable, Tuple

try:
    from imblearn.metrics import geometric_mean_score
except ImportError:
    import warnings

    warnings.warn("Imblearn failed to import!\nSome score functions will not be run.")

    def geometric_mean_score(*arg, **kwargs):
        # TODO: None or np.nan??
        return None


np.random.seed(2556)


def _score_helper(
    score_func: Callable,
    y_true: np.array,
    y_pred: np.array,
    average: str = "macro",
    labels: List[str] = None,
    pos_label: None = None,
) -> float:
    mlcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = mlcm[:, 0, 0], mlcm[:, 0, 1], mlcm[:, 1, 0], mlcm[:, 1, 1]

    num, denom = score_func(tn, fp, fn, tp)

    if average is None:
        return num / denom
    elif average == "micro":
        return num.sum() / sum(i.sum() for i in denom)
    elif average in ["macro", "weighted"]:
        if average == "macro":
            class_weights = None
        else:
            class_weights = tp + fn

        return np.average(num / sum(denom), weights=class_weights)
    elif average == "binary":
        return num[-1] / sum(i[-1] for i in denom)
    else:
        raise NotImplementedError("value for average of '{average}' not supported")


def fnr_score(y_true, y_pred, average="macro", labels=None):
    # fn / (fn + tp)
    def _inner_score(tn, fp, fn, tp):
        return fn, (tp, fn)

    return _score_helper(_inner_score, y_true, y_pred, average, labels)


def npv_score(y_true, y_pred, average="macro", labels=None):
    # tn / (tn + fn)
    def _inner_score(tn, fp, fn, tp):
        return tn, (tn, fn)

    return _score_helper(_inner_score, y_true, y_pred, average, labels)


def specificity_score(y_true, y_pred, average="macro", labels=None):
    # tn / (tn + fp)
    def _inner_score(tn, fp, fn, tp):
        return tn, (tn, fp)

    return _score_helper(_inner_score, y_true, y_pred, average, labels)


def fdr_score(y_true, y_pred, average="macro", labels=None):
    # fp / (tp + fp)
    def _inner_score(tn, fp, fn, tp):
        return fp, (fp, tp)

    return _score_helper(_inner_score, y_true, y_pred, average, labels)


def fpr_score(y_true, y_pred, average="macro", labels=None):
    # fp / (tn + fp)
    def _inner_score(tn, fp, fn, tp):
        return fp, (fp, tn)

    return _score_helper(_inner_score, y_true, y_pred, average, labels)


def conf_interval(
    score_functions: List[Callable],
    y_true: np.array,
    y_data: np.array = None,  # on user to entire pred or pred_proba
    n: int = 500,
    per: Tuple[float] = (2.5, 97.5),
) -> List[Tuple[float]]:
    cis = []
    for _ in range(n):
        aucs = []
        for score_func in score_functions:
            sample = np.random.choice(range(len(y_true)), len(y_true), replace=True)
            aucs.append(score_func(y_true[sample], y_data[sample]))
        cis.append(aucs)
    return np.round(np.percentile(cis, per, axis=0), 2).T


# Define the PR AUC scorer function
def pr_auc_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(
        average_precision_score, y_true, y_pred_proba
    )
    pr_auc = auc(recall, precision)
    return pr_auc


def compute_scores(
    y_true,
    y_pred,
    y_pred_proba=None,
    ci_conf=None,
    auc_conf=None,
    ap_conf=None,
    average="binary",
):
    """
    auc_conf (Union[None, Tuple[float, float]]): None or Tuple of percentages to calculate CI for.
    """

    # Assume y_true and y_pred are the true labels and predicted labels, respectively
    # You can obtain them using the model.predict() method or any other means
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = specificity_score(y_true, y_pred, average=average)
    fnr = 1 - recall_score(y_true, y_pred, average=average)
    fdr = 1 - precision_score(y_true, y_pred, average=average)
    lr_pos = recall_score(y_true, y_pred, average=average) / fpr_score(
        y_true, y_pred, average=average
    )
    lr_neg = fnr / specificity
    dor = lr_pos / lr_neg
    ts = tp / (tp + fn + fp)

    scoring = {
        "f1_score": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, average=average),
        "specificity": specificity,
        "ppv": precision_score(y_true, y_pred, average=average),
        "npv": npv_score(y_true, y_pred, average=average),
        "fnr": fnr,
        "fdr": fdr,
        "fpr": fpr_score(y_true, y_pred, average=average),
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "dor": dor,
        "ts": ts,
        "gmean": geometric_mean_score(y_true, y_pred, average=average),
    }
    if ci_conf is not None:
        metric_names = [
            "f1_score_CI",
            "accuracy_CI",
            "sensitivity_CI",
            "specificity_CI",
            "ppv_CI",
            "npv_CI",
            "fnr_CI",
        ]
        CIs = conf_interval(
            y_true=y_true,
            y_data=y_pred,
            score_functions=[
                partial(f1_score, average=average),
                accuracy_score,
                partial(recall_score, average=average),
                partial(specificity_score, average=average),
                partial(precision_score, average=average),
                partial(npv_score, average=average),
                partial(fnr_score, average=average),
            ],
            per=ci_conf,
        )
        scoring.update({key: value for key, value in zip(metric_names, CIs)})
    if y_pred_proba is not None:
        scoring.update(
            {
                "pr_auc": average_precision_score(y_true, y_pred_proba[:, 1]),
                "roc_auc": roc_auc_score(y_true, y_pred_proba[:, 1]),
            }
        )
        if auc_conf is not None:
            scoring.update(
                {
                    "auc_CI": conf_interval(
                        y_true=y_true,
                        y_data=y_pred_proba[:, 1],
                        per=auc_conf,
                        score_functions=[roc_auc_score],
                    )[0],
                }
            )
        if ap_conf is not None:
            scoring.update(
                {
                    "ap_CI": conf_interval(
                        y_true=y_true,
                        y_data=y_pred_proba[:, 1],
                        per=auc_conf,
                        score_functions=[average_precision_score],
                    )[0],
                }
            )
    return scoring


# Create the custom scorer using make_scorer
pr_auc_scorer = make_scorer(pr_auc_score, greater_is_better=True, needs_proba=True)
f1_score_scorer = make_scorer(f1_score, greater_is_better=True)
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
