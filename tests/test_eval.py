"""Test evaluation/metric calculation functions (non-plotting)."""

import pytest
import numpy as np
from tn4ml.eval import (
    get_roc_curve_data,
    get_precision_recall_curve_data,
    get_FPR_for_fixed_TPR,
    get_TPR_for_fixed_FPR,
    get_mean_and_error,
)


# --- get_roc_curve_data ---


def test_get_roc_curve_data_binary():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr = get_roc_curve_data(y_true, y_scores)
    assert len(fpr) == len(tpr)
    assert fpr[0] == 0.0
    assert tpr[-1] == 1.0


def test_get_roc_curve_data_anomaly_det():
    y_true = np.array([0.1, 0.2, 0.3])
    y_scores = np.array([0.8, 0.9, 0.7])
    fpr, tpr = get_roc_curve_data(y_true, y_scores, anomaly_det=True)
    assert len(fpr) == len(tpr)


def test_get_roc_curve_data_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.0, 0.0, 1.0, 1.0])
    fpr, tpr = get_roc_curve_data(y_true, y_scores)
    # Perfect classifier should have AUC = 1.0
    from sklearn.metrics import auc

    assert auc(fpr, tpr) == pytest.approx(1.0)


# --- get_precision_recall_curve_data ---


def test_get_precision_recall_curve_data():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall = get_precision_recall_curve_data(y_true, y_scores)
    assert len(precision) == len(recall)


def test_get_precision_recall_curve_data_anomaly():
    y_true = np.array([0.1, 0.2])
    y_scores = np.array([0.9, 0.8])
    precision, recall = get_precision_recall_curve_data(
        y_true, y_scores, anomaly_det=True
    )
    assert len(precision) == len(recall)


# --- get_FPR_for_fixed_TPR ---


def test_get_FPR_for_fixed_TPR():
    fpr = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
    result = get_FPR_for_fixed_TPR(0.8, fpr, tpr, tolerance=0.1)
    assert isinstance(result, (float, np.floating))


# --- get_TPR_for_fixed_FPR ---


def test_get_TPR_for_fixed_FPR():
    fpr = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
    result = get_TPR_for_fixed_FPR(0.2, fpr, tpr, tolerance=0.1)
    assert isinstance(result, (float, np.floating))


# --- get_mean_and_error ---


def test_get_mean_and_error_1d():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, std = get_mean_and_error(data)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(np.std(data))


def test_get_mean_and_error_2d():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, std = get_mean_and_error(data)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    np.testing.assert_allclose(mean, [3.0, 4.0])


def test_get_mean_and_error_single():
    data = np.array([[1.0, 2.0, 3.0]])
    mean, std = get_mean_and_error(data)
    np.testing.assert_allclose(mean, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(std, [0.0, 0.0, 0.0])
