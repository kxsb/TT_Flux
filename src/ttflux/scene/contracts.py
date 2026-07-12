from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any


SCENE_SCHEMA_VERSION = 1
TABLE_GEOMETRY_KIND = "table_geometry_2d"

_EPSILON = 1e-9


def _finite_float(
    name: str,
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"{name} must be a finite number."
        )

    number = float(value)

    if not isfinite(number):
        raise ValueError(
            f"{name} must be finite."
        )

    return number


def _positive_int(
    name: str,
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )

    return value


def _text(
    name: str,
    value: Any,
    *,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string."
        )

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return cleaned


def _mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be an object."
        )

    return value


@dataclass(frozen=True)
class Point2D:
    """Point expressed in source-image pixel coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x",
            _finite_float("x", self.x),
        )
        object.__setattr__(
            self,
            "y",
            _finite_float("y", self.y),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> Point2D:
        data = _mapping(
            "point",
            payload,
        )

        return cls(
            x=data.get("x"),
            y=data.get("y"),
        )


def _cross(
    first: Point2D,
    second: Point2D,
    third: Point2D,
) -> float:
    return (
        (second.x - first.x)
        * (third.y - first.y)
        - (second.y - first.y)
        * (third.x - first.x)
    )


@dataclass(frozen=True)
class TableGeometry2D:
    """
    Convex table quadrilateral in image coordinates.

    Corners follow the semantic perimeter order:
    near_left, near_right, far_right, far_left.
    """

    image_width: int
    image_height: int
    near_left: Point2D
    near_right: Point2D
    far_right: Point2D
    far_left: Point2D

    def __post_init__(self) -> None:
        width = _positive_int(
            "image_width",
            self.image_width,
        )
        height = _positive_int(
            "image_height",
            self.image_height,
        )

        object.__setattr__(
            self,
            "image_width",
            width,
        )
        object.__setattr__(
            self,
            "image_height",
            height,
        )

        names = (
            "near_left",
            "near_right",
            "far_right",
            "far_left",
        )

        for name, point in zip(
            names,
            self.corners,
            strict=True,
        ):
            if not isinstance(point, Point2D):
                raise TypeError(
                    f"{name} must be a Point2D."
                )

            if not (
                0.0 <= point.x <= width
                and 0.0 <= point.y <= height
            ):
                raise ValueError(
                    f"{name} lies outside "
                    "the image bounds."
                )

        unique_points = {
            (point.x, point.y)
            for point in self.corners
        }

        if len(unique_points) != 4:
            raise ValueError(
                "Table corners must be distinct."
            )

        turns = tuple(
            _cross(
                self.corners[index],
                self.corners[
                    (index + 1) % 4
                ],
                self.corners[
                    (index + 2) % 4
                ],
            )
            for index in range(4)
        )

        if any(
            abs(turn) <= _EPSILON
            for turn in turns
        ):
            raise ValueError(
                "Table corners cannot be collinear."
            )

        has_positive = any(
            turn > 0
            for turn in turns
        )
        has_negative = any(
            turn < 0
            for turn in turns
        )

        if has_positive and has_negative:
            raise ValueError(
                "Table quadrilateral must be "
                "convex and ordered."
            )

        if self.polygon_area_px2 <= _EPSILON:
            raise ValueError(
                "Table polygon area must be positive."
            )

    @property
    def corners(
        self,
    ) -> tuple[
        Point2D,
        Point2D,
        Point2D,
        Point2D,
    ]:
        return (
            self.near_left,
            self.near_right,
            self.far_right,
            self.far_left,
        )

    @property
    def polygon_area_px2(self) -> float:
        points = self.corners

        twice_area = sum(
            points[index].x
            * points[(index + 1) % 4].y
            - points[(index + 1) % 4].x
            * points[index].y
            for index in range(4)
        )

        return abs(twice_area) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                SCENE_SCHEMA_VERSION
            ),
            "coordinate_space": (
                "image_pixels"
            ),
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "corners": {
                "near_left": (
                    self.near_left.to_dict()
                ),
                "near_right": (
                    self.near_right.to_dict()
                ),
                "far_right": (
                    self.far_right.to_dict()
                ),
                "far_left": (
                    self.far_left.to_dict()
                ),
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TableGeometry2D:
        data = _mapping(
            "table geometry",
            payload,
        )

        if (
            data.get("schema_version")
            != SCENE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported table geometry "
                "schema version."
            )

        if (
            data.get("coordinate_space")
            != "image_pixels"
        ):
            raise ValueError(
                "Unsupported table coordinate space."
            )

        image_size = _mapping(
            "image_size",
            data.get("image_size"),
        )
        corners = _mapping(
            "corners",
            data.get("corners"),
        )

        return cls(
            image_width=image_size.get(
                "width"
            ),
            image_height=image_size.get(
                "height"
            ),
            near_left=Point2D.from_dict(
                _mapping(
                    "near_left",
                    corners.get("near_left"),
                )
            ),
            near_right=Point2D.from_dict(
                _mapping(
                    "near_right",
                    corners.get("near_right"),
                )
            ),
            far_right=Point2D.from_dict(
                _mapping(
                    "far_right",
                    corners.get("far_right"),
                )
            ),
            far_left=Point2D.from_dict(
                _mapping(
                    "far_left",
                    corners.get("far_left"),
                )
            ),
        )


@dataclass(frozen=True)
class SceneProvenance:
    """Origin of one scene observation."""

    source_kind: str
    method: str
    frame_index: int | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_kind",
            _text(
                "source_kind",
                self.source_kind,
            ),
        )
        object.__setattr__(
            self,
            "method",
            _text(
                "method",
                self.method,
            ),
        )

        if self.frame_index is not None:
            if (
                isinstance(
                    self.frame_index,
                    bool,
                )
                or not isinstance(
                    self.frame_index,
                    int,
                )
            ):
                raise TypeError(
                    "frame_index must be "
                    "an integer or None."
                )

            if self.frame_index < 0:
                raise ValueError(
                    "frame_index cannot be negative."
                )

        object.__setattr__(
            self,
            "model_id",
            _text(
                "model_id",
                self.model_id,
                optional=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "method": self.method,
            "frame_index": self.frame_index,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> SceneProvenance:
        data = _mapping(
            "provenance",
            payload,
        )

        return cls(
            source_kind=data.get(
                "source_kind"
            ),
            method=data.get("method"),
            frame_index=data.get(
                "frame_index"
            ),
            model_id=data.get("model_id"),
        )


@dataclass(frozen=True)
class TableObservation2D:
    """
    Versioned table observation with validity and uncertainty.
    """

    geometry: TableGeometry2D | None
    provenance: SceneProvenance
    confidence: float
    uncertainty_px: float
    valid: bool
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            self.geometry is not None
            and not isinstance(
                self.geometry,
                TableGeometry2D,
            )
        ):
            raise TypeError(
                "geometry must be a "
                "TableGeometry2D or None."
            )

        if not isinstance(
            self.provenance,
            SceneProvenance,
        ):
            raise TypeError(
                "provenance must be "
                "a SceneProvenance."
            )

        confidence = _finite_float(
            "confidence",
            self.confidence,
        )
        uncertainty = _finite_float(
            "uncertainty_px",
            self.uncertainty_px,
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if uncertainty < 0.0:
            raise ValueError(
                "uncertainty_px cannot be negative."
            )

        if not isinstance(self.valid, bool):
            raise TypeError(
                "valid must be a boolean."
            )

        reason = _text(
            "invalid_reason",
            self.invalid_reason,
            optional=True,
        )

        if self.valid and self.geometry is None:
            raise ValueError(
                "A valid observation requires geometry."
            )

        if self.valid and reason is not None:
            raise ValueError(
                "A valid observation cannot have "
                "an invalid_reason."
            )

        if (
            not self.valid
            and self.geometry is not None
        ):
            raise ValueError(
                "An invalid observation cannot "
                "expose geometry."
            )

        if not self.valid and reason is None:
            raise ValueError(
                "An invalid observation requires "
                "an invalid_reason."
            )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )
        object.__setattr__(
            self,
            "uncertainty_px",
            uncertainty,
        )
        object.__setattr__(
            self,
            "invalid_reason",
            reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                SCENE_SCHEMA_VERSION
            ),
            "kind": TABLE_GEOMETRY_KIND,
            "valid": self.valid,
            "confidence": self.confidence,
            "uncertainty_px": (
                self.uncertainty_px
            ),
            "invalid_reason": (
                self.invalid_reason
            ),
            "provenance": (
                self.provenance.to_dict()
            ),
            "geometry": (
                self.geometry.to_dict()
                if self.geometry is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TableObservation2D:
        data = _mapping(
            "table observation",
            payload,
        )

        if (
            data.get("schema_version")
            != SCENE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported table observation "
                "schema version."
            )

        if (
            data.get("kind")
            != TABLE_GEOMETRY_KIND
        ):
            raise ValueError(
                "Unsupported scene observation kind."
            )

        valid = data.get("valid")

        if not isinstance(valid, bool):
            raise TypeError(
                "valid must be a boolean."
            )

        geometry_payload = data.get(
            "geometry"
        )

        geometry = (
            None
            if geometry_payload is None
            else TableGeometry2D.from_dict(
                _mapping(
                    "geometry",
                    geometry_payload,
                )
            )
        )

        return cls(
            geometry=geometry,
            provenance=(
                SceneProvenance.from_dict(
                    _mapping(
                        "provenance",
                        data.get("provenance"),
                    )
                )
            ),
            confidence=data.get(
                "confidence"
            ),
            uncertainty_px=data.get(
                "uncertainty_px"
            ),
            valid=valid,
            invalid_reason=data.get(
                "invalid_reason"
            ),
        )


__all__ = [
    "Point2D",
    "SCENE_SCHEMA_VERSION",
    "SceneProvenance",
    "TABLE_GEOMETRY_KIND",
    "TableGeometry2D",
    "TableObservation2D",
]
