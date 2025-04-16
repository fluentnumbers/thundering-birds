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
        truth: Ground truth labels (continuous or binary)
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

    # Convert continuous truth values to binary if they're not already
    binary_truth = (truth > 0.5).astype(int)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        binary_truth, binary_preds, average=None, zero_division=0
    )

    # Calculate AUC per class - use original continuous values for AUC
    aucs = []
    for i in range(pred.shape[1]):
        if np.sum(binary_truth[:, i]) > 0:
            auc = roc_auc_score(binary_truth[:, i], pred[:, i])
            aucs.append(auc)
        else:
            aucs.append(-1)

    # Calculate confusion matrix with proper label mapping
    true_labels = np.argmax(binary_truth, axis=1)
    pred_labels = np.argmax(binary_preds, axis=1)
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(labels)))

    # Calculate top-k accuracy using binary truth values
    k = 3
    top_k_preds = np.argsort(pred, axis=1)[:, -k:]
    top_k_accuracy = np.mean(
        [
            1 if binary_truth[i].argmax() in top_k_preds[i] else 0
            for i in range(len(binary_truth))
        ]
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


def plot_class_metrics(metrics: Dict, labels: List[str]) -> List[go.Figure]:
    """
    Create interactive plots for class-specific metrics.

    Args:
        metrics: Dictionary containing metrics from calculate_class_metrics
        labels: List of class names

    Returns:
        List of Plotly figures, one for each metric type
    """
    # Extract metrics
    precisions = [metrics["per_class"][label]["precision"] for label in labels]
    recalls = [metrics["per_class"][label]["recall"] for label in labels]
    f1s = [metrics["per_class"][label]["f1"] for label in labels]
    aucs = [metrics["per_class"][label]["auc"] for label in labels]
    supports = [metrics["per_class"][label]["support"] for label in labels]

    # Create separate figures for each metric
    figures = []
    metric_data = [
        ("Precision", precisions),
        ("Recall", recalls),
        ("F1 Score", f1s),
        ("AUC", aucs),
    ]

    for title, values in metric_data:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=values, name=title))

        # Update layout
        fig.update_layout(
            height=400,
            width=600,
            title_text=f"{title} by Class",
            showlegend=False,
            xaxis_title="Class",
            yaxis_title=title,
            yaxis_range=[0, 1],  # Set y-axis range to [0, 1] for all metrics
        )

        # Add support as hover text
        fig.update_traces(
            hovertemplate="Class: %{x}<br>"
            + f"{title}: %{{y:.3f}}<br>Support: %{{customdata}}<extra></extra>",
            customdata=supports,
        )

        figures.append(fig)

    return figures


def plot_confusion_matrix(cm: np.ndarray, labels: List[str]) -> plt.Figure:
    """
    Create a confusion matrix plot using matplotlib.

    Args:
        cm: Confusion matrix
        labels: List of class names

    Returns:
        matplotlib Figure object
    """
    # Create figure and axes with larger size for high resolution
    plt.close("all")  # Close any existing figures

    # Calculate figure size based on number of classes
    n_classes = len(labels)
    fig_size = min(
        20, max(8, n_classes * 0.5)
    )  # Scale figure size with number of classes
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=200)
    ax = fig.add_subplot(111)

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    # Create heatmap
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Ratio", rotation=-90, va="bottom", fontsize=10)

    # Set up axes
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, pad=20)

    # Add ticks
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    # Determine fontsize based on number of classes
    fontsize = min(10, max(6, int(200 / n_classes)))

    # Add tick labels
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)

    # Add text annotations
    thresh = cm_norm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:  # Only show non-zero values
                text = f"{cm[i, j]}\n{cm_norm[i, j]:.1%}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=max(6, int(fontsize * 0.8)),
                )

    # Adjust layout
    plt.subplots_adjust(bottom=0.2)

    return fig


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


def create_sampling_plots(segment_usage_stats, epoch):
    """
    Create Plotly figures for sampling statistics that can be tracked over epochs.

    Args:
        segment_usage_stats: Dictionary containing sampling statistics
        epoch: Current epoch number

    Returns:
        Dictionary of plot names to plotly figures and summary statistics
    """
    plots = {}
    classes = segment_usage_stats["classes"]

    # Define metrics to plot with their titles and y-axis labels
    metrics_info = [
        ("total_segments", "Total Segments per Class", "Count"),
        ("total_segments_drawn", "Segments Drawn per Class", "Count"),
        ("mean_usage_per_class", "Mean Usage per Class", "Times Used"),
        ("max_usage_per_class", "Maximum Usage per Class", "Times Used"),
        ("unused_segments_per_class", "Unused Segments per Class", "Count"),
    ]

    for metric_key, title, y_label in metrics_info:
        values = segment_usage_stats[metric_key]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=classes,
                y=[v for v in values],
                name=metric_key,
                hovertemplate="Class: %{x}<br>"
                + f"{y_label}: %{{y:.2f}}<extra></extra>",
                orientation="v",
            )
        )

        # Update layout
        fig.update_layout(
            height=400,
            width=800,
            title_text=f"{title} (Epoch {epoch})",
            showlegend=False,
            xaxis_title="Class",
            yaxis_title=y_label,
            xaxis=dict(tickangle=45, type="category"),  # Force categorical axis
            margin=dict(b=100),  # Add bottom margin for rotated labels
        )

        # If all values are zero, add annotation
        if all(v == 0 for v in values):
            fig.add_annotation(
                text="No data recorded yet",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )

        # Store plot and summary statistic
        plot_key = f"{metric_key}"
        stat_key = f"{metric_key}_mean"

        plots[plot_key] = fig
        plots[stat_key] = float(np.mean(values))

    return plots
