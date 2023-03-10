import pytest
import numpy as np

from comvis import Pi, PiInv


def test_homogeneous_coordinates():

    inhom_coordinates_2D = np.array([10, 20]).reshape(-1, 1)
    assert all(PiInv(inhom_coordinates_2D) == np.array([10, 20, 1]).reshape(-1, 1))

    hom_coordinates_2D = np.array([10, 20, 2]).reshape(-1, 1)
    assert all(PiInv(hom_coordinates_2D) == np.array([10, 20, 2, 1]).reshape(-1, 1))
    assert all(Pi(hom_coordinates_2D) == np.array([5, 10]).reshape(-1, 1))
