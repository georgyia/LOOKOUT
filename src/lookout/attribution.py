"""Attribution: from a screen point to a target, against a viewer's layout.

Given a gaze point on a viewer's screen and the layout valid at that time, decide
which region it falls in and how confident that is. Confidence comes from how far
the point sits from the nearest border relative to the tile size, so central hits
are trusted and near-border hits degrade to ``low_confidence``. Genuine ambiguity
(a point shared by two regions) is ``unknown``. A target is never invented, and
the layout provenance is propagated into every result.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Attribution, GazePoint, GazeTarget, Layout, Region, RegionKind

__all__ = ["AttributionParams", "attribute_point", "attribute_target"]


@dataclass(frozen=True)
class AttributionParams:
    """Thresholds for turning geometry into a target and a confidence."""

    confidence_floor: float = 0.15


def _region_target(region: Region) -> str:
    if region.kind is RegionKind.PARTICIPANT:
        assert region.participant_id is not None
        return region.participant_id
    return region.kind.value


def attribute_point(
    point: GazePoint,
    layout: Layout,
    params: AttributionParams | None = None,
) -> Attribution:
    """Attribute one gaze point against a viewer's layout."""

    params = params or AttributionParams()
    containing = layout.regions_containing(point.x, point.y)

    if not containing:
        return Attribution(GazeTarget.UNKNOWN.value, point.confidence, "no region", layout.source)

    if len(containing) > 1:
        return Attribution(
            GazeTarget.UNKNOWN.value, point.confidence, "ambiguous overlap", layout.source
        )

    region = containing[0]
    half_min = min(region.width, region.height) / 2.0
    margin = region.edge_distance(point.x, point.y)
    margin_ratio = max(0.0, min(1.0, margin / half_min)) if half_min > 0 else 0.0
    confidence = point.confidence * margin_ratio
    target = _region_target(region)

    if confidence < params.confidence_floor:
        return Attribution(
            GazeTarget.LOW_CONFIDENCE.value,
            confidence,
            f"near border: {target}",
            layout.source,
            margin_ratio=margin_ratio,
        )

    reason = "self" if target == layout.viewer_id else "center hit"
    return Attribution(target, confidence, reason, layout.source, margin_ratio=margin_ratio)


def attribute_target(
    target: GazeTarget,
    layout: Layout,
    confidence: float = 1.0,
) -> Attribution:
    """Wrap an upstream non-participant result (e.g. ``off_screen``,
    ``not_visible``) as an :class:`Attribution`, preserving provenance."""

    return Attribution(target.value, confidence, target.value, layout.source)
