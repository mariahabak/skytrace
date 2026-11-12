"""Metrics, the shared model interface, the Evaluator and Grad-CAM.

Author: Mariah Habak
"""

import pytest
import torch

from conftest import EVALUATION, make_bank
from dronebird import metrics
from dronebird.evaluation import Evaluator
from dronebird.gradcam import GradCAM
from dronebird.models.majority import MajorityClassifier
from dronebird.models.network_classifier import NetworkClassifier


def test_metrics_by_hand():
    y_true = torch.tensor([0, 0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 1, 1, 0])
    assert metrics.confusion_matrix(y_true, y_pred).tolist() == [[2, 1], [1, 1]]
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(0.6)
    # bird F1 = 2/3, drone F1 = 1/2
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_network_follows_the_interface_and_reloads(tmp_path):
    train, val = make_bank(seed=0), make_bank(seed=1)
    model = NetworkClassifier("small_cnn", margin=1.5, canvas=96, seed=0, pretrained=False,
                              epochs=1, batch_size=8, num_workers=0, device="cpu")
    model.fit(train, val, tmp_path)
    images = val.images.float() / 255
    scores = model.predict_scores(images)
    assert scores.shape == (len(val), 2)
    assert torch.allclose(scores.sum(dim=1), torch.ones(len(val)), atol=1e-5)
    assert (tmp_path / "best.pt").exists()
    again = NetworkClassifier("small_cnn", margin=1.5, canvas=96, seed=0, pretrained=False,
                              device="cpu")
    again.load(tmp_path)
    assert torch.allclose(again.predict_scores(images), scores, atol=1e-5)


def test_evaluator_builds_the_size_curve_dictionary(bank):
    model = MajorityClassifier()
    model.fit(bank, bank, None)
    results = Evaluator(EVALUATION, margin=1.5, canvas=96).size_curve(model, bank, "val")
    for size in ("full", "48", "32", "24", "16"):
        assert ("synthetic", size) in results
    for size in ("48", "32", "24", "16", "all"):
        assert ("real_small", size) in results
    synthetic_n = int((bank.object_size >= 64).sum())
    assert results[("synthetic", "16")]["n"] == synthetic_n
    assert results[("all", "full")]["accuracy"] == pytest.approx(0.5)


def test_gradcam_shape_and_range():
    model = NetworkClassifier("small_cnn", margin=1.5, canvas=96, seed=0, pretrained=False,
                              device="cpu")
    net = model.network
    cam, cls, scores = GradCAM(net, net.cam_layer())(torch.rand(3, 96, 96))
    assert cam.shape == (96, 96)
    assert cam.min() >= 0 and cam.max() <= 1
    assert cls in (0, 1) and scores.shape == (2,)
