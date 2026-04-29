"""Tests for _grid.compute_output_grid."""

import numpy as np
import pytest

from lazycogs._grid import compute_output_grid


def test_basic_dimensions():
    """Width and height are derived from bbox / resolution."""
    _transform, w, h, _x, _y = compute_output_grid((0.0, 0.0, 10.0, 5.0), 1.0)
    assert w == 10
    assert h == 5


def test_affine_origin():
    """Transform origin is the top-left corner of the top-left pixel."""
    transform, _w, _h, _x, _y = compute_output_grid((10.0, 20.0, 30.0, 40.0), 1.0)
    assert transform.c == pytest.approx(10.0)
    assert transform.f == pytest.approx(40.0)
    assert transform.a == pytest.approx(1.0)
    assert transform.e == pytest.approx(-1.0)


def test_pixel_centres():
    """x/y coordinate arrays hold pixel centres, not edges."""
    _transform, _w, _h, x, y = compute_output_grid((0.0, 0.0, 4.0, 2.0), 1.0)
    np.testing.assert_allclose(x, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(y, [0.5, 1.5])


def test_x_coords_increase():
    """x coordinates increase left-to-right."""
    _, _, _, x, _ = compute_output_grid((0.0, 0.0, 10.0, 10.0), 1.0)
    assert np.all(np.diff(x) > 0)


def test_y_coords_increase():
    """y coordinates increase south-to-north so ascending slices work naturally."""
    _, _, _, _, y = compute_output_grid((0.0, 0.0, 10.0, 10.0), 1.0)
    assert np.all(np.diff(y) > 0)


def test_coord_array_lengths():
    """x and y arrays have lengths matching width and height."""
    _transform, w, h, x, y = compute_output_grid((0.0, 0.0, 10.0, 5.0), 1.0)
    assert len(x) == w
    assert len(y) == h


def test_small_bbox_rounds_to_one_pixel():
    """A bbox smaller than one resolution step yields a 1x1 grid."""
    _transform, w, h, _x, _y = compute_output_grid((0.0, 0.0, 0.1, 0.1), 1.0)
    assert w == 1
    assert h == 1


def test_non_unit_resolution():
    """Non-unit resolution produces correct pixel count and spacing."""
    transform, w, h, x, y = compute_output_grid((0.0, 0.0, 100.0, 50.0), 10.0)
    assert w == 10
    assert h == 5
    assert transform.a == pytest.approx(10.0)
    assert transform.e == pytest.approx(-10.0)
    np.testing.assert_allclose(x[0], 5.0)
    np.testing.assert_allclose(y[0], 5.0)


def test_first_and_last_pixel_centres_span_bbox():
    """First and last pixel centres lie half-pixel inside the bbox edges."""
    res = 2.0
    minx, miny, maxx, maxy = 0.0, 0.0, 10.0, 6.0
    _, _w, _h, x, y = compute_output_grid((minx, miny, maxx, maxy), res)
    assert x[0] == pytest.approx(minx + res / 2)
    assert x[-1] == pytest.approx(maxx - res / 2)
    assert y[0] == pytest.approx(miny + res / 2)
    assert y[-1] == pytest.approx(maxy - res / 2)
