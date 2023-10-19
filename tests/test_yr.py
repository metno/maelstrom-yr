#!/usr/bin/env python3

import climetlab as cml
import pandas as pd
import os
import numpy as np

from climetlab_maelstrom_yr.yr import Yr

dir = os.path.join(os.path.dirname(__file__), "../data")

def test_read():
    cmlds = cml.load_dataset(
        "maelstrom-yr",
        size="5GB",
        dates=["2020-03-01", "2020-03-02"],
        # location=f"{dir}/",
        probabilistic_target=False,
    )
    ds = cmlds.to_xarray()
    I = np.where(ds["predictor"] == "air_temperature_2m")[0][0]
    value = ds["predictors"][0, 0, 0, 0, I].values
    np.testing.assert_almost_equal(value, 0.33352637)

    value = ds["targets"][0, 0, 0, 0, 0].values
    np.testing.assert_almost_equal(value, 0.3591344)

def test_normalize_dataset():
    cmlds = cml.load_dataset(
        "maelstrom-yr",
        size="5GB",
        dates=["2020-03-01", "2020-03-02"],
        # location=f"{dir}/",
        probabilistic_target=False,
        predict_diff=True,
        normalize=True,
    )
    ds = cmlds.to_xarray()
    I = np.where(ds["predictor"] == "air_temperature_2m")[0][0]
    value = ds["predictors"][0, 0, 0, 0, I].values
    np.testing.assert_almost_equal(value, (0.33352637 - 5.388952960570653)/7.4335476655246735)

    # Check that the target has been normalized by air_temperature_2m
    value = ds["targets"][0, 0, 0, 0, 0].values
    np.testing.assert_almost_equal(value, 0.3591344 - 0.33352637)

def test_normalize_functions():
    ar = np.array([0, 1, 2], np.float32)
    Yr.normalize(ar, "air_temperature_2m")
    Yr.denormalize(ar, "air_temperature_2m")
    np.testing.assert_array_almost_equal(ar, [0, 1, 2])


if __name__ == "__main__":
    from climetlab.testing import main

    test_read()
    test_normalize_dataset()
    test_normalize_functions()
