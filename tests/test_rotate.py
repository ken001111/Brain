import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rotate import calculate_rotation_angle, find_eye_centers


def test_find_eye_centers_returns_sorted_centers():
    mask = np.zeros((3, 5, 5), dtype=int)
    mask[1, 1:3, 1:3] = 1  # left eye blob
    mask[1, 1:3, 4:5] = 1  # right eye blob separated by a zero column

    centers = find_eye_centers(mask)

    assert len(centers) == 2
    assert centers[0][0] < centers[1][0]
    # centroids should be roughly at the center of each blob
    assert centers[0][0] == pytest.approx(1.5)
    assert centers[1][0] == pytest.approx(4.0)
    assert centers[0][1] == pytest.approx(1.5)
    assert centers[1][1] == pytest.approx(1.5)


def test_calculate_rotation_angle_with_two_centers():
    centers = [(0.0, 0.0), (1.0, 1.0)]

    angle = calculate_rotation_angle(centers)

    assert angle == pytest.approx(45.0)


def test_calculate_rotation_angle_with_single_center_returns_zero():
    angle = calculate_rotation_angle([(2.0, 3.0)])

    assert angle == 0
