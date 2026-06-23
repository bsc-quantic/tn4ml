"""Test evaluation/metric calculation functions."""

import matplotlib as mpl

mpl.use("Agg")  # headless backend so plotting functions never open a window

import numpy as np
import pytest

from tn4ml.eval import (
    compare_AUC,
    compare_FPR_per_TPR,
    compare_TPR_per_FPR,
    get_FPR_for_fixed_TPR,
    get_mean_and_error,
    get_precision_recall_curve_data,
    get_roc_curve_data,
    get_TPR_for_fixed_FPR,
    plot_accuracy,
    plot_loss,
    plot_PR_curve,
    plot_ROC_curve_from_data,
    plot_ROC_curve_from_metrics,
)

# --- get_roc_curve_data ---


def test_get_roc_curve_data_binary():
    """Test get roc curve data binary."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr = get_roc_curve_data(y_true, y_scores)
    assert len(fpr) == len(tpr)
    assert fpr[0] == 0.0
    assert tpr[-1] == 1.0


def test_get_roc_curve_data_anomaly_det():
    """Test get roc curve data anomaly det."""
    y_true = np.array([0.1, 0.2, 0.3])
    y_scores = np.array([0.8, 0.9, 0.7])
    fpr, tpr = get_roc_curve_data(y_true, y_scores, anomaly_det=True)
    assert len(fpr) == len(tpr)


def test_get_roc_curve_data_perfect():
    """Test get roc curve data perfect."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.0, 0.0, 1.0, 1.0])
    fpr, tpr = get_roc_curve_data(y_true, y_scores)
    # Perfect classifier should have AUC = 1.0
    from sklearn.metrics import auc

    assert auc(fpr, tpr) == pytest.approx(1.0)


# --- get_precision_recall_curve_data ---


def test_get_precision_recall_curve_data():
    """Test get precision recall curve data."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall = get_precision_recall_curve_data(y_true, y_scores)
    assert len(precision) == len(recall)


def test_get_precision_recall_curve_data_anomaly():
    """Test get precision recall curve data anomaly."""
    y_true = np.array([0.1, 0.2])
    y_scores = np.array([0.9, 0.8])
    precision, recall = get_precision_recall_curve_data(
        y_true, y_scores, anomaly_det=True
    )
    assert len(precision) == len(recall)


# --- get_FPR_for_fixed_TPR ---


def test_get_FPR_for_fixed_TPR():  # noqa: N802
    """Test get FPR for fixed TPR."""
    fpr = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
    result = get_FPR_for_fixed_TPR(0.8, fpr, tpr, tolerance=0.1)
    assert isinstance(result, (float, np.floating))


# --- get_TPR_for_fixed_FPR ---


def test_get_TPR_for_fixed_FPR():  # noqa: N802
    """Test get TPR for fixed FPR."""
    fpr = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
    result = get_TPR_for_fixed_FPR(0.2, fpr, tpr, tolerance=0.1)
    assert isinstance(result, (float, np.floating))


# --- get_mean_and_error ---


def test_get_mean_and_error_1d():
    """Test get mean and error 1d."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, std = get_mean_and_error(data)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(np.std(data))


def test_get_mean_and_error_2d():
    """Test get mean and error 2d."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, std = get_mean_and_error(data)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    np.testing.assert_allclose(mean, [3.0, 4.0])


def test_get_mean_and_error_single():
    """Test get mean and error single."""
    data = np.array([[1.0, 2.0, 3.0]])
    mean, std = get_mean_and_error(data)
    np.testing.assert_allclose(mean, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(std, [0.0, 0.0, 0.0])


# --- plot_loss / plot_accuracy ---


def test_plot_loss_show():
    """Test plot loss show."""
    history = {"loss": [1.0, 0.5, 0.2], "val_loss": [1.1, 0.6, 0.3]}
    plot_loss(history)  # save_path None -> show() path (no-op under Agg)


def test_plot_loss_save(tmp_path):
    """Test plot loss save."""
    history = {"loss": [1.0, 0.5, 0.2], "val_loss": [1.1, 0.6, 0.3]}
    save_path = tmp_path / "loss"
    plot_loss(history, validation=False, save_path=str(save_path))
    assert save_path.with_suffix(".pdf").exists()


def test_plot_accuracy(tmp_path):
    """Test plot accuracy."""
    history = {"val_acc": [0.5, 0.7, 0.9]}
    save_path = tmp_path / "acc"
    plot_accuracy(history, save_path=str(save_path))
    assert save_path.with_suffix(".pdf").exists()


# --- ROC / PR plots ---


def test_plot_ROC_curve_from_metrics(tmp_path):  # noqa: N802
    """Test plot ROC curve from metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    save_path = tmp_path / "roc"
    plot_ROC_curve_from_metrics(y_true, y_scores, save_path=str(save_path))
    assert save_path.with_suffix(".pdf").exists()


def test_plot_ROC_curve_from_metrics_show():  # noqa: N802
    """Test plot ROC curve from metrics show."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    plot_ROC_curve_from_metrics(y_true, y_scores)


def test_plot_ROC_curve_from_data(tmp_path):  # noqa: N802
    """Test plot ROC curve from data."""
    fpr = np.array([0.0, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.6, 0.8, 1.0])
    save_path = tmp_path / "roc_data"
    plot_ROC_curve_from_data(fpr, tpr, save_path=str(save_path))
    assert save_path.with_suffix(".pdf").exists()


def test_plot_PR_curve_binary(tmp_path):  # noqa: N802
    """Test plot PR curve binary."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    save_path = tmp_path / "pr"
    plot_PR_curve(y_true, y_scores, save_path=str(save_path))
    assert save_path.with_suffix(".pdf").exists()


# --- compare_* hyperparameter sweep plots ---


def _make_results_tree(root, inits, bonds, spacings, embedding, nruns):
    """Create the directory layout that the compare_* functions read from."""
    rng = np.random.default_rng(0)
    for init in inits:
        for bond in bonds:
            for spacing in spacings:
                for j in range(1, nruns + 1):
                    leaf = f"{init}/bond_{bond}/spacing_{spacing}/{embedding}"
                    if nruns != 1:
                        leaf += f"/run_{j}"
                    d = root / leaf
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / "normal_score.npy", rng.random(20))
                    np.save(d / "anomaly_score.npy", rng.random(20) + 1.0)


def _labels(bonds):
    palette = ["#016c59", "#7a5195", "#67a9cf", "#ffa600"]
    markers = ["o", "X", "v", "d"]
    return {
        str(b): (f"bond = {b}", markers[i % 4], palette[i % 4])
        for i, b in enumerate(bonds)
    }


@pytest.mark.parametrize("nruns", [1, 2])
def test_compare_AUC(tmp_path, nruns):  # noqa: N802
    """Test compare AUC."""
    inits, bonds, spacings, emb = ["randn_1e-1"], [5, 10], [2], "trigonometric"
    _make_results_tree(tmp_path, inits, bonds, spacings, emb, nruns)
    compare_AUC(
        save_dir=str(tmp_path),
        bond_dims=bonds,
        spacings=spacings,
        initializers=inits,
        embedding=emb,
        nruns=nruns,
        labels=_labels(bonds),
        anomaly_det=True,
    )
    assert (tmp_path / "results" / "plots" / "AUC" / "spacing_2.pdf").exists()


def test_compare_TPR_per_FPR(tmp_path):  # noqa: N802
    """Test compare TPR per FPR."""
    inits, bonds, spacings, emb = ["randn_1e-1"], [5, 10], [2], "trigonometric"
    _make_results_tree(tmp_path, inits, bonds, spacings, emb, nruns=2)
    compare_TPR_per_FPR(
        save_dir=str(tmp_path),
        FPR_fixed=0.1,
        bond_dims=bonds,
        spacings=spacings,
        initializers=inits,
        embedding=emb,
        nruns=2,
        labels=_labels(bonds),
        anomaly_det=True,
    )
    assert (tmp_path / "results" / "plots" / "TPR" / "spacing_2_FPR_0.1.pdf").exists()


def test_compare_FPR_per_TPR(tmp_path):  # noqa: N802
    """Test compare FPR per TPR."""
    inits, bonds, spacings, emb = ["randn_1e-1"], [5, 10], [2], "trigonometric"
    _make_results_tree(tmp_path, inits, bonds, spacings, emb, nruns=2)
    compare_FPR_per_TPR(
        save_dir=str(tmp_path),
        TPR_fixed=0.95,
        bond_dims=bonds,
        spacings=spacings,
        initializers=inits,
        embedding=emb,
        nruns=2,
        labels=_labels(bonds),
        anomaly_det=True,
    )
    assert (tmp_path / "results" / "plots" / "FPR" / "spacing_2_TPR_0.95.pdf").exists()
