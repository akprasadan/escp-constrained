"""
Unit tests for `rect_A()` (half-open boxes [xmin, xmax) x [ymin, ymax) x... of any
dimension, including 1).

Tests:
- Interior vs. exterior point membership
- Inclusion of lower edges and exclusion of upper edges
- Dense grids inside square/rectangle
- Fully separated rectangle selects no points
- Input validation for invalid bounds
"""

import numpy as np
import pytest
from itertools import product


from src.box_functions import rect_A as rect_A


# [0, 1)
simple_segment = rect_A(np.array([0]), np.array([1]))
# [0, 1) x [0, 1)
simple_square = rect_A(np.array([0, 0]), np.array([1, 1]))
# [0, 5) x [0, 1)
simple_rectangle = rect_A(np.array([0, 0]), np.array([5, 1]))
# [0, 1) x [0, 1) x [0, 1)
simple_cube = rect_A(np.array([0, 0, 0]), np.array([1, 1, 1]))
# [0, 1) x [0, 5) x [0, 1)
simple_box = rect_A(np.array([0, 0, 0]), np.array([1, 5, 1]))


def test_rect_A_one_dim():
    """Simple checks for one-D inputs."""
    # Make sure (N, ) points are of shape (N, 1)
    points = np.array([0.5, -10, -0.1, 0, 0.5, 0.99, 1, 1.2, 20])[:, None]
    mask = simple_segment(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)

    expected = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0])
    assert np.allclose(expected, mask)


def test_rect_A_interior_membership_square():
    """Points strictly inside the square are selected (no boundary points)."""
    # Let's generate every possible point that is internally adjacent to a edge or corner
    # of the unit square. This is the set of tuples formed by the cartesian product
    # {0.01, 0.99}^2.

    coordinate_vals = [0.01, 0.99]
    points = np.array([p for p in product(coordinate_vals, repeat=2)])
    # Add the center point
    points = np.vstack([points, [0.5, 0.5]])

    mask = simple_square(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)
    assert np.all(mask)


def test_rect_A_interior_membership_cube():
    """Points strictly inside the cube are selected (no boundary points)."""

    # Let's generate every possible point that is internally adjacent to a face, edge, or corner
    # of the unit cube. This is the set of tuples formed by the cartesian product
    # {0.01, 0.99}^3.

    coordinate_vals = [0.01, 0.99]
    points = np.array([p for p in product(coordinate_vals, repeat=3)])

    # Add the center point
    points = np.vstack([points, [0.5, 0.5, 0.5]])

    mask = simple_cube(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)
    assert np.all(mask)


def test_rect_A_exterior_membership_square():
    """Points strictly outside the square are not selected (no boundary points)."""

    # Let's generate every possible point that is external to a edge or corner
    # of the unit square. This is the set of tuples formed by the cartesian product
    # {-0.2, 1.2, 0.5}^2, excluding the (0.5, 0.5) pair.

    coordinate_vals = [-0.2, 0.5, 1.2]
    points = np.array(
        [p for p in product(coordinate_vals, repeat=2) if p != (0.5, 0.5)]
    )

    mask = simple_square(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)
    assert np.all(~mask)


def test_rect_A_exterior_membership_cube():
    """Points strictly outside the cube are not selected (no boundary points)."""

    # Let's generate every possible point that is external to a face, edge, or corner
    # of the unit cube. This is the set of tuples formed by the cartesian product
    # {-0.2, 1.2, 0.5}^3, excluding the (0.5, 0.5, 0.5) triplet.

    coordinate_vals = [-0.2, 0.5, 1.2]
    points = np.array(
        [p for p in product(coordinate_vals, repeat=3) if p != (0.5, 0.5, 0.5)]
    )

    mask = simple_cube(points)
    assert mask.dtype == bool
    assert mask.shape == (len(points),)
    assert np.all(~mask)


def test_rect_A_includes_lower_boundaries_square():
    """Left/bottom edges are included; bottom-left corner is included."""
    pts_on_lower_edges = np.array(
        [
            [0, 0],  # bottom-left (included)
            [0, 0.5],  # left edge
            [0.5, 0],  # bottom edge
            [0.99, 0],  # bottom edge near right
            [0, 0.99],  # left edge near top
            [0, 0.01],  # left edge near bottom
            [0.01, 0],  # bottom edge near left
        ]
    )
    mask = simple_square(pts_on_lower_edges)
    assert np.all(mask)


def test_rect_A_includes_lower_boundaries_cube():
    """
    Ensure inclusion of:
    Left/front/bottom faces (3), their pair-wise shared edges (3),
    and mutually shared corner (1).
    """
    pts_on_lower_edges = np.array(
        [
            [0, 0.5, 0.5],  # left face
            [0.5, 0.5, 0],  # bottom face
            [0.5, 0, 0.5],  # front face
            [0, 0.5, 0],  # bottom-left edge
            [0, 0, 0.5],  # left-front edge
            [0.5, 0, 0],  # bottom-front edge
            [0, 0, 0],  # bottom-left-front corner
        ]
    )
    mask = simple_cube(pts_on_lower_edges)
    assert np.all(mask)


def test_rect_A_excludes_upper_boundaries_square():
    """Right/top edges are excluded; any point with x==xmax or y==ymax is out."""
    pts_on_upper_edges = np.array(
        [
            [1, 0],  # bottom-right corner
            [1, 1],  # top-right corner
            [0, 1],  # top-left corner
            [0.5, 1],  # top edge
            [1, 0.5],  # right edge
            [0.01, 1],  # top edge near left
            [0.99, 1],  # top edge near right
            [1, 0.01],  # right edge near bottom
            [1, 0.99],  # right edge near top
        ]
    )
    mask = simple_square(pts_on_upper_edges)
    assert np.all(~mask)


def test_rect_A_excludes_upper_boundaries_cube():
    """Right/top/back edges, faces, or corners are excluded; any point with x = xmax, y = ymax, z = zmax is out.
    Note this excludes every corner but the bottom/left/front one at (0, 0, 0).
    """
    pts_on_upper_edges = np.array(
        [
            [1, 0.5, 0.5],  # right face
            [1, 0, 0.5],  # right-front edge
            [1, 1, 0.5],  # right-back edge
            [0.5, 1, 0.5],  # back face
            [0.5, 1, 0],  # bottom-back edge
            [1, 0.5, 0],  # bottom-right edge
            [0.5, 0.5, 1],  # top face
            [0, 0.5, 1],  # top-left edge
            [1, 0.5, 1],  # top-right edge
            [0.5, 0, 1],  # top-front edge
            [0.5, 1, 1],  # top-back edge
            [0, 1, 0.5],  # left-back edge
            [1, 1, 1],  # top-right-back corner
            [1, 0, 1],  # top-right-front corner
            [0, 1, 1],  # top-left-back corner
            [0, 0, 1],  # top-left-front corner
            [0, 1, 0],  # bottom-left-back corner
            [1, 1, 0],  # bottom-right-back corner
            [1, 0, 0],  # bottom-right-front corner
        ]
    )
    mask = simple_cube(pts_on_upper_edges)
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


def test_rect_A_dense_grid_in_box():
    """Dense grid strictly inside a box is fully selected."""
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 0.99, 100),
        np.linspace(0, 4.99, 100),
        np.linspace(0, 0.99, 100),
    )
    points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    # [0, 1) x [0, 5) x [0, 1)
    mask = simple_box(points)
    assert np.all(mask)


def test_rect_A_no_points_selected():
    """Separated rectangle selects no points."""
    # Form points spanning square [0, 1]^2
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    # Let A be rectangle on [2, 3]^2
    A = rect_A(np.array([2, 2]), np.array([3, 3]))
    mask = A(points)
    assert not mask.any()


def test_rect_A_no_points_selected_3d():
    """Separated box selects no points."""
    # Form points spanning box [0, 1]^3
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10)
    )
    points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    # Let A be box on [2, 3]^3
    A = rect_A(np.array([2, 2, 2]), np.array([3, 3, 3]))
    mask = A(points)
    assert not mask.any()


def test_rect_A_raises_when_bounds_invalid():
    """Invalid bounds raise ValueError. Tests 1D, 2D or 3D examples."""
    # Min and max values of incorrect length
    with pytest.raises(ValueError, match="Incompatible lengths"):
        rect_A(np.array([0]), np.array([1, 1]))

    with pytest.raises(ValueError, match="Incompatible lengths"):
        rect_A(np.array([0, 0]), np.array([1, 1, 1]))

    with pytest.raises(ValueError, match="Incompatible lengths"):
        rect_A(np.array([0, 0, 0]), np.array([1, 1]))

    with pytest.raises(ValueError, match="Incompatible lengths"):
        rect_A(np.array([0]), np.array([1, 1]))

    # Bad x-value in 1D
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([1]), np.array([1]))  # xmin == xmax
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([2]), np.array([1]))  # xmin > xmax

    # Bad x-value in 2D
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([1, 0]), np.array([1, 1]))  # xmin == xmax
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([2, 0]), np.array([1, 1]))  # xmin > xmax

    # Bad y-value in 2D
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([0, 1]), np.array([1, 1]))  # ymin == ymax
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([0, 2]), np.array([1, 1]))  # ymin > ymax

    # Bad x-value in 3D
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([1, 0, 0]), np.array([1, 1, 1]))  # xmin == xmax
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([2, 0, 0]), np.array([1, 1, 1]))  # xmin > xmax

    # Bad z-value in 3D
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([0, 0, 1]), np.array([1, 1, 1]))  # zmin == zmax
    with pytest.raises(ValueError, match="min_vals >= max_vals"):
        rect_A(np.array([0, 0, 2]), np.array([1, 1, 1]))  # zmin > zmax
