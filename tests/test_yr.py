#!/usr/bin/env python3

import climetlab as cml
import pandas as pd

from climetlab_maelstrom_yr.yr import Yr


def test_read():
    return
    ds = cml.load_dataset(
        "maelstrom-yr",
        size="5GB",
        dates=["2020-06-26"],
        parameter="air_temperature",
    )
    ds.to_xarray()


def test_parse_dates():
    expected = ["20170101", "20170102", "20170103"]
    pd_dates = pd.date_range(start="2017-01-01", end="2017-01-03", freq="1D")
    assert Yr.parse_dates(pd_dates) == expected
    assert Yr.parse_dates(["2017-01-01", "2017-01-02", "2017-01-03"]) == expected


if __name__ == "__main__":
    from climetlab.testing import main

    main(globals())
