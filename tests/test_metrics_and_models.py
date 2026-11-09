"""Metrics on hand-checked examples, and the shared model interface.

Authors: Maria Habak, [Partner name]
"""

import numpy as np
import pytest
import torch

from conftest import make_clip_set
from skytrace.evaluation import metrics
from skytrace.evaluation.uncertainty import UNCERTAIN, RejectionRule, TemperatureScaler
from skytrace.models.baseline import FeatureBaseline
from skytrace.models.temporal import Aggregator


def test_confusion_and_macro_f1_by_hand():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    assert metrics.confusion_matrix(y_true, y_pred).tolist() == [[2, 1], [1, 1]]
    # bird: P = 2/3, R = 2/3 -> F1 = 2/3 ; drone: P = 1/2, R = 1/2 -> F1 = 1/2
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_error_coverage_rejects_least_confident_first():
    scores = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.45, 0.55]])
    y = np.array([0, 1, 1, 0])  # the two least confident are wrong
    coverage, error = metrics.error_coverage(y, scores)
    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert error.tolist() == [0.0, 0.0, pytest.approx(1 / 3), 0.5]


def test_rejection_threshold_comes_from_validation():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 1000)
    scores = np.stack([p, 1 - p], axis=1)
    rule = RejectionRule(target_coverage=0.8)
    rule.fit(scores)
    decided = rule.decide(scores)
    assert (decided != UNCERTAIN).mean() == pytest.approx(0.8, abs=0.01)


def test_temperature_scaling_softens_overconfident_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    correct = rng.random(2000) < 0.7  # right only 70% of the time
    pred = np.where(correct, y, 1 - y)
    scores = np.where(pred[:, None] == np.arange(2), 0.99, 0.01)  # but 99% sure
    scaler = TemperatureScaler()
    assert scaler.fit(y, scores) > 1.0
    assert metrics.expected_calibration_error(y, scaler.transform(scores)) < \
        metrics.expected_calibration_error(y, scores)


def test_baseline_follows_the_interface(tmp_path):
    train, test = make_clip_set(seed=0), make_clip_set(seed=1)
    model = FeatureBaseline(epochs=300)
    model.fit(train, train, tmp_path)
    scores = model.predict_scores(test)
    assert scores.shape == (len(test), 2)
    assert np.allclose(scores.sum(axis=1), 1)
    assert metrics.macro_f1(test.labels.numpy(), scores.argmax(1)) > 0.9
    model.save(tmp_path)
    reloaded = FeatureBaseline()
    reloaded.load(tmp_path)
    assert np.allclose(reloaded.predict_scores(test), scores)


def test_mean_aggregator_ignores_frame_order_but_gru_does_not():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 512)
    reordered = x[:, torch.randperm(8)]
    mean = Aggregator("mean", 512, 64, dropout=0.0).eval()
    gru = Aggregator("gru", 512, 64, dropout=0.0).eval()
    assert torch.allclose(mean(x), mean(reordered), atol=1e-5)
    assert not torch.allclose(gru(x), gru(reordered), atol=1e-5)
