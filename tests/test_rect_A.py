"""
Unit tests for `rect_A()` (half-open rectangles [xmin, xmax) x [ymin, ymax)).

Tests:
- Interior vs. exterior point membership
- Inclusion of lower edges and exclusion of upper edges
- Dense grids inside square/rectangle
- Fully separated rectangle selects no points
- Input validation for invalid bounds
"""

import numpy as np
import pytest

from src.box_functions import rect_A as rect_A


# Half-open: [0, 1) x [0, 1)
simple_square = rect_A(0, 1, 0, 1)
# Half-open: [0, 5) x [0, 1)
simple_rectangle = rect_A(0, 5, 0, 1)


def test_rect_A_interior_membership():
    """Points strictly inside the square are selected (no boundary points)."""
    points = np.array(
        [
            [0.5, 0.5],
            [0.01, 0.01],
            [0.99, 0.99],
            [0.01, 0.99],
            [0.99, 0.01],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.7, 0.3],
        ]
    )
    mask = simple_square(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)
    assert np.all(mask)


def test_rect_A_exterior_membership():
    """Points strictly outside the square are not selected (no boundary points)."""
    points = np.array(
        [
            [-0.20, 0.5],  # left
            [1.2, 0.5],  # right
            [0.5, -0.2],  # below
            [0.5, 1.2],  # above
            [-0.2, -0.2],  # bottom-left
            [1.2, 1.2],  # top-right
            [-0.2, 1.2],  # top-left
            [1.2, -1.2],  # bottom-right
        ]
    )
    mask = simple_square(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)
    assert np.all(~mask)


def test_rect_A_includes_lower_boundaries():
    """Left/bottom edges are included; bottom-left corner is included."""
    pts_on_lower_edges = np.array(
        [
            [0.0, 0.0],  # bottom-left (included)
            [0.0, 0.5],  # left edge
            [0.5, 0.0],  # bottom edge
            [0.99, 0.0],  # bottom edge near right
            [0.0, 0.99],  # left edge near top
            [0.0, 0.01],  # left edge near bottom
            [0.01, 0.0],  # bottom edge near left
        ]
    )
    mask = simple_square(pts_on_lower_edges)
    assert np.all(mask)


def test_rect_A_excludes_upper_boundaries():
    """Right/top edges are excluded; any point with x==xmax or y==ymax is out."""
    pts_on_upper_edges = np.array(
        [
            [1.0, 0.0],  # bottom-right corner
            [1.0, 1.0],  # top-right corner
            [0.0, 1.0],  # top-left corner
            [0.5, 1.0],  # top edge
            [1.0, 0.5],  # right edge
            [0.01, 1.0],  # top edge near left
            [0.99, 1.0],  # top edge near right
            [1.0, 0.01],  # right edge near bottom
            [1.0, 0.99],  # right edge near top
        ]
    )
    mask = simple_square(pts_on_upper_edges)
    assert np.all(~mask)


def test_rect_A_dense_grid_in_square():
    """Dense grid strictly inside the square is fully selected."""
    xs, ys = np.meshgrid(np.linspace(0, 0.99, 100), np.linspace(0, 0.99, 100))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    mask = simple_square(points)
    assert np.all(mask)


def test_rect_A_dense_grid_in_rectangle():
    """Dense grid strictly inside the rectangle is fully selected."""
    xs, ys = np.meshgrid(np.linspace(0, 4.99, 100), np.linspace(0, 0.99, 100))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    mask = simple_rectangle(points)
    assert np.all(mask)


def test_rect_A_no_points_selected():
    """Separated rectangle selects no points."""
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    A = rect_A(2.0, 3.0, 2.0, 3.0)
    mask = A(points)
    assert not mask.any()


def test_rect_A_raises_when_x_bounds_invalid():
    """Invalid x-bounds raise ValueError."""
    with pytest.raises(ValueError, match="Require xmin < xmax and ymin < ymax"):
        rect_A(1.0, 1.0, 0.0, 1.0)  # xmin == xmax
    with pytest.raises(ValueError, match="Require xmin < xmax and ymin < ymax"):
        rect_A(2.0, 1.0, 0.0, 1.0)  # xmin > xmax


def test_rect_A_raises_when_y_bounds_invalid():
    """Invalid y-bounds raise ValueError."""
    with pytest.raises(ValueError, match="Require xmin < xmax and ymin < ymax"):
        rect_A(0.0, 1.0, 1.0, 1.0)  # ymin == ymax
    with pytest.raises(ValueError, match="Require xmin < xmax and ymin < ymax"):
        rect_A(0.0, 1.0, 2.0, 1.0)  # ymin > ymax
