from __future__ import annotations

import json

import pytest

import ttflux.scene as scene


EXPECTED_PUBLIC_NAMES = {
    "Point2D",
    "SCENE_SCHEMA_VERSION",
    "SceneProvenance",
    "TABLE_GEOMETRY_KIND",
    "TableGeometry2D",
    "TableObservation2D",
}


def make_geometry() -> scene.TableGeometry2D:
    return scene.TableGeometry2D(
        image_width=1280,
        image_height=720,
        near_left=scene.Point2D(
            100,
            650,
        ),
        near_right=scene.Point2D(
            1180,
            650,
        ),
        far_right=scene.Point2D(
            900,
            200,
        ),
        far_left=scene.Point2D(
            380,
            200,
        ),
    )


def make_provenance() -> scene.SceneProvenance:
    return scene.SceneProvenance(
        source_kind="manual",
        method="four_corner_annotation",
        frame_index=125,
        model_id=None,
    )


def test_scene_package_exposes_minimal_public_api() -> None:
    assert set(scene.__all__) == (
        EXPECTED_PUBLIC_NAMES
    )

    for name in EXPECTED_PUBLIC_NAMES:
        assert getattr(scene, name) is not None


def test_point_contract_normalizes_and_validates() -> None:
    point = scene.Point2D(
        12,
        34.5,
    )

    assert point.x == 12.0
    assert point.y == 34.5
    assert scene.Point2D.from_dict(
        point.to_dict()
    ) == point

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        scene.Point2D(
            float("nan"),
            0,
        )

    with pytest.raises(TypeError):
        scene.Point2D(
            True,
            0,
        )


def test_table_geometry_round_trip_is_stable() -> None:
    geometry = make_geometry()
    payload = geometry.to_dict()

    serialized = json.loads(
        json.dumps(payload)
    )
    restored = (
        scene.TableGeometry2D.from_dict(
            serialized
        )
    )

    assert restored == geometry
    assert payload["schema_version"] == 1
    assert payload["coordinate_space"] == (
        "image_pixels"
    )
    assert geometry.polygon_area_px2 > 0


def test_table_geometry_rejects_out_of_bounds_corner() -> None:
    with pytest.raises(
        ValueError,
        match="outside",
    ):
        scene.TableGeometry2D(
            image_width=1280,
            image_height=720,
            near_left=scene.Point2D(
                -1,
                650,
            ),
            near_right=scene.Point2D(
                1180,
                650,
            ),
            far_right=scene.Point2D(
                900,
                200,
            ),
            far_left=scene.Point2D(
                380,
                200,
            ),
        )


def test_table_geometry_rejects_crossed_order() -> None:
    with pytest.raises(
        ValueError,
        match="convex",
    ):
        scene.TableGeometry2D(
            image_width=100,
            image_height=100,
            near_left=scene.Point2D(
                10,
                90,
            ),
            near_right=scene.Point2D(
                90,
                10,
            ),
            far_right=scene.Point2D(
                90,
                90,
            ),
            far_left=scene.Point2D(
                10,
                10,
            ),
        )


def test_valid_table_observation_round_trip() -> None:
    observation = scene.TableObservation2D(
        geometry=make_geometry(),
        provenance=make_provenance(),
        confidence=0.92,
        uncertainty_px=3.5,
        valid=True,
    )

    serialized = json.loads(
        json.dumps(
            observation.to_dict()
        )
    )
    restored = (
        scene.TableObservation2D.from_dict(
            serialized
        )
    )

    assert restored == observation
    assert serialized["kind"] == (
        "table_geometry_2d"
    )
    assert serialized["valid"] is True
    assert serialized["invalid_reason"] is None
    assert serialized["provenance"][
        "frame_index"
    ] == 125


def test_invalid_observation_contract_is_explicit() -> None:
    invalid = scene.TableObservation2D(
        geometry=None,
        provenance=make_provenance(),
        confidence=0.0,
        uncertainty_px=25.0,
        valid=False,
        invalid_reason="table_not_visible",
    )

    restored = (
        scene.TableObservation2D.from_dict(
            invalid.to_dict()
        )
    )

    assert restored == invalid
    assert restored.geometry is None

    with pytest.raises(
        ValueError,
        match="requires geometry",
    ):
        scene.TableObservation2D(
            geometry=None,
            provenance=make_provenance(),
            confidence=0.8,
            uncertainty_px=4.0,
            valid=True,
        )

    with pytest.raises(
        ValueError,
        match="cannot expose geometry",
    ):
        scene.TableObservation2D(
            geometry=make_geometry(),
            provenance=make_provenance(),
            confidence=0.0,
            uncertainty_px=10.0,
            valid=False,
            invalid_reason="rejected",
        )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        scene.TableObservation2D(
            geometry=make_geometry(),
            provenance=make_provenance(),
            confidence=1.1,
            uncertainty_px=1.0,
            valid=True,
        )
