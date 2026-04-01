"""
Unit tests for `src.counts_observed()`.

Tests:
- Bin counting for simple cases
- Empty bins or empty input
- Edge inclusivity: last bin includes its right edge
- When all values fall in the last bin
- Output length matches number of bins
"""

import numpy as np

from src.output_functions import counts_observed as counts_observed


def test_counts_observed_basic():
    """Values are counted into bins correctly; last bin includes right edge."""
    q_data = np.array([0.1, 0.5, 1.2, 1.8, 2.0])
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])  # bins: [0,1), [1,2), [2,3]
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [2, 2, 1]


def test_counts_observed_empty_bin():
    """Bins with no values report zero counts."""
    q_data = np.array([0.2, 0.3])  # both in first bin
    bin_edges = np.array([0.0, 0.5, 1.0])
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [2, 0]


def test_counts_observed_empty_list_returns_zeros():
    """Empty input returns zeros for all bins."""
    bin_edges = np.array([-1.0, 0.0, 0.5, 2.0])  # 3 bins
    counts = counts_observed(np.array([]), bin_edges)
    M = len(bin_edges) - 1
    assert counts.shape == (M,)
    assert counts.tolist() == [0] * M


def test_counts_observed_edges_inclusive():
    """Last bin includes its right edge; others are half-open."""
    q_data = np.array([0.0, 1.0, 2.0])
    bin_edges = np.array([0.0, 1.0, 2.0])
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [1, 2]


def test_counts_observed_all_in_last_bin():
    """All values at or above last bin's left edge go into last bin."""
    q_data = np.array([1.0, 5.0, 10.0])
    bin_edges = np.array([0.0, 1.0, 10.0])
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [0, 3]


def test_counts_observed_length_matches_bins():
    """Output length equals number of bins (len(bin_edges) - 1)."""
    q_data = np.array([0.1, 0.5, 0.9, 1.5])
    bin_edges = np.array([0.0, 0.5, 1.0, 2.0])  # 3 bins
    obs_counts = counts_observed(q_data, bin_edges)
    assert len(obs_counts) == len(bin_edges) - 1
