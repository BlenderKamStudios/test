"""EXPERIMENTAL spatial heatmap algorithms. Register new ones in METHODS."""
from .base import Heatmap, Link, SpatialEstimator, links_from_config  # noqa: F401  (public API)
from .ellipse_weighting import EllipseWeighting
from .link_pattern import LinkPattern

METHODS = {
    EllipseWeighting.name: EllipseWeighting,
    LinkPattern.name: LinkPattern,
}


def make_estimator(cfg: dict, room: dict | None = None, links: list | None = None) -> SpatialEstimator:
    """room/links default to config.yaml's room + receivers; the dashboard
    passes its own TX-centred layout instead."""
    lcfg = cfg["localization"]
    try:
        cls = METHODS[lcfg["method"]]
    except KeyError:
        raise ValueError(f"unknown localization.method {lcfg['method']!r}; known: {list(METHODS)}") from None
    return cls(room or cfg["room"], links if links is not None else links_from_config(cfg), lcfg)
