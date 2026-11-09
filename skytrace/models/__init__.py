"""Model registry: the JSON 'model.type' picks the class.

Authors: Maria Habak, [Partner name]
"""

from skytrace.models.base import ClipClassifier


def build_model(settings: dict, seed: int) -> ClipClassifier:
    settings = dict(settings)
    kind = settings.pop("type")
    if kind == "feature_baseline":
        from skytrace.models.baseline import FeatureBaseline
        return FeatureBaseline(**settings)
    if kind == "frame_cnn":
        from skytrace.models.frame_cnn import FrameCNN
        return FrameCNN(seed=seed, **settings)
    if kind == "temporal":
        from skytrace.models.temporal import TemporalClassifier
        return TemporalClassifier(seed=seed, **settings)
    raise ValueError(f"Unknown model type '{kind}'")
