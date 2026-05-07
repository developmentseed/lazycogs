"""Tests for _grid.compute_output_grid."""

from __future__ import annotations

import pytest

from lazycogs._grid import compute_output_grid


def test_basic_dimensions():
    """Width and height are derived from bbox / resolution."""
    _transform, w, h = compute_output_grid((0.0, 0.0, 10.0, 5.0), 1.0)
    assert w == 10
    assert h == 5


def test_affine_origin():
    """Transform origin is the top-left corner of the top-left pixel."""
    transform, _w, _h = compute_output_grid((10.0, 20.0, 30.0, 40.0), 1.0)
    assert transform.c == pytest.approx(10.0)
    assert transform.f == pytest.approx(40.0)
    assert transform.a == pytest.approx(1.0)
    assert transform.e == pytest.approx(-1.0)


def test_small_bbox_rounds_to_one_pixel():
    """A bbox smaller than one resolution step yields a 1x1 grid."""
    _transform, w, h = compute_output_grid((0.0, 0.0, 0.1, 0.1), 1.0)
    assert w == 1
    assert h == 1


def test_non_unit_resolution():
    """Non-unit resolution produces correct pixel count and spacing."""
    transform, w, h = compute_output_grid((0.0, 0.0, 100.0, 50.0), 10.0)
    assert w == 10
    assert h == 5
    assert transform.a == pytest.approx(10.0)
    assert transform.e == pytest.approx(-10.0)
