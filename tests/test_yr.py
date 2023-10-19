#!/usr/bin/env python3

import climetlab as cml
import pandas as pd
import os

from climetlab_maelstrom_yr.yr import Yr

def test_read():
    dir = os.path.join(os.path.dirname(__file__), "files")
    ds = cml.load_dataset(
        "maelstrom-yr",
        size="5GB",
        # dates=["2020-03-01", "2020-03-02"],
        # location=f"{dir}/",
        probabilistic_target=False,
        verbose=True,
    )
    q = ds.to_xarray()


if __name__ == "__main__":
    from climetlab.testing import main

    test_read()
