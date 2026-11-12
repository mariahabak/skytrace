"""Model registry: the JSON 'model.type' picks the class.

Author: Mariah Habak
"""

from dronebird.models.base import TargetClassifier


def build_model(settings: dict, seed: int, margin: float, canvas: int) -> TargetClassifier:
    settings = dict(settings)
    kind = settings.pop("type")
    if kind == "majority":
        from dronebird.models.majority import MajorityClassifier
        return MajorityClassifier()
    if kind == "network":
        from dronebird.models.network_classifier import NetworkClassifier
        return NetworkClassifier(margin=margin, canvas=canvas, seed=seed, **settings)
    raise ValueError(f"Unknown model type '{kind}'")
