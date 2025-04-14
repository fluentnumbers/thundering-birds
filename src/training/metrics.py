import base64
import io
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def macro_auc(truth, pred, labels, return_per_class=False):
    """
    Computes the macro AUC score.

    Args:
        truth (array-like): Ground truth labels.
        pred (array-like): Predicted probabilities.
        return_per_class (bool, optional): Return AUC scores per class. Defaults to False.

    Returns:
        float or tuple: Macro AUC score or tuple of mean AUC and AUC scores per class.
    """
    aucs = []
    aucs_per_class = {}

    if isinstance(truth, list):
        truth_ = np.zeros_like(pred)
        for i in range(len(truth)):
            if isinstance(truth[i], str):
                truth_[i, labels.index(truth[i])] = 1
            elif isinstance(truth[i], list) and isinstance(truth[i][0], str):
                for t in truth[i]:
                    truth_[i, labels.index(t)] = 1
            else:
                raise NotImplementedError(
                    "Expects list of strings or list of list of strings"
                )
        truth = truth_

    for i in range(pred.shape[1]):
        if truth[:, i].min() != truth[:, i].max():
            auc = roc_auc_score(truth[:, i], pred[:, i])
            aucs.append(auc)
            aucs_per_class[labels[i]] = auc
        else:
            aucs_per_class[labels[i]] = -1

    if return_per_class:
        return np.mean(aucs), aucs_per_class
    return np.mean(aucs)


def calculate_class_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    labels: List[str],
    thresholds: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calculate comprehensive class-specific metrics.

    Args:
        truth: Ground truth labels (one-hot encoded)
        pred: Predicted probabilities
        labels: List of class names
        thresholds: Optional class-specific thresholds

    Returns:
        Dictionary containing various metrics
    """
    if thresholds is None:
        thresholds = np.array([0.5] * len(labels))

    # Convert probabilities to binary predictions
    binary_preds = (pred > thresholds).astype(int)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, binary_preds, average=None, zero_division=0
    )

    # Calculate AUC per class
    aucs = []
    for i in range(pred.shape[1]):
        if np.sum(truth[:, i]) > 0:
            auc = roc_auc_score(truth[:, i], pred[:, i])
            aucs.append(auc)
        else:
            aucs.append(-1)

    # Calculate confusion matrix
    cm = confusion_matrix(truth.argmax(axis=1), binary_preds.argmax(axis=1))

    # Calculate top-k accuracy
    k = 3
    top_k_preds = np.argsort(pred, axis=1)[:, -k:]
    top_k_accuracy = np.mean(
        [1 if truth[i].argmax() in top_k_preds[i] else 0 for i in range(len(truth))]
    )

    # Compile results
    results = {
        "per_class": {
            label: {"precision": p, "recall": r, "f1": f, "auc": a, "support": s}
            for label, p, r, f, a, s in zip(
                labels, precision, recall, f1, aucs, support
            )
        },
        "confusion_matrix": cm,
        "top_k_accuracy": top_k_accuracy,
        "macro_metrics": {
            "precision": np.mean(precision),
            "recall": np.mean(recall),
            "f1": np.mean(f1),
            "auc": np.mean([a for a in aucs if a != -1]),
        },
    }

    return results


def plot_class_metrics(metrics: Dict, labels: List[str]) -> go.Figure:
    """
    Create interactive plots for class-specific metrics.

    Args:
        metrics: Dictionary containing metrics from calculate_class_metrics
        labels: List of class names

    Returns:
        Plotly figure with multiple subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Precision by Class",
            "Recall by Class",
            "F1 Score by Class",
            "AUC by Class",
        ),
    )

    # Extract metrics
    precisions = [metrics["per_class"][label]["precision"] for label in labels]
    recalls = [metrics["per_class"][label]["recall"] for label in labels]
    f1s = [metrics["per_class"][label]["f1"] for label in labels]
    aucs = [metrics["per_class"][label]["auc"] for label in labels]
    supports = [metrics["per_class"][label]["support"] for label in labels]

    # Add traces
    fig.add_trace(go.Bar(x=labels, y=precisions, name="Precision"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=recalls, name="Recall"), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=f1s, name="F1"), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=aucs, name="AUC"), row=2, col=2)

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text="Class-wise Performance Metrics",
        showlegend=False,
    )

    # Add support as hover text
    for i in range(len(labels)):
        fig.update_traces(
            hovertemplate=f"Class: {labels[i]}<br>Support: {supports[i]}<extra></extra>",
            selector={"name": "Precision"},
        )

    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str]) -> np.ndarray:
    """
    Create and save confusion matrix plot.

    Args:
        cm: Confusion matrix
        labels: List of class names

    Returns:
        Numpy array of the image
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    # Convert to numpy array
    img = PIL.Image.open(buf)
    img_array = np.array(img)
    return img_array


def analyze_class_performance(
    metrics: Dict, labels: List[str], class_distribution: Dict[str, int]
) -> Dict:
    """
    Analyze class performance in relation to class distribution.

    Args:
        metrics: Dictionary containing metrics from calculate_class_metrics
        labels: List of class names
        class_distribution: Dictionary of class names to sample counts

    Returns:
        Dictionary containing analysis results
    """
    # Calculate correlation between class size and performance
    class_sizes = [class_distribution[label] for label in labels]
    f1_scores = [metrics["per_class"][label]["f1"] for label in labels]
    auc_scores = [metrics["per_class"][label]["auc"] for label in labels]

    # Identify hard and easy classes
    hard_classes = []
    easy_classes = []
    for label in labels:
        if metrics["per_class"][label]["f1"] < 0.5:
            hard_classes.append(label)
        elif metrics["per_class"][label]["f1"] > 0.8:
            easy_classes.append(label)

    return {
        "class_size_correlation": {
            "f1": np.corrcoef(class_sizes, f1_scores)[0, 1],
            "auc": np.corrcoef(class_sizes, auc_scores)[0, 1],
        },
        "hard_classes": hard_classes,
        "easy_classes": easy_classes,
        "class_performance_summary": {
            label: {
                "size": class_distribution[label],
                "f1": metrics["per_class"][label]["f1"],
                "auc": metrics["per_class"][label]["auc"],
            }
            for label in labels
        },
    }
