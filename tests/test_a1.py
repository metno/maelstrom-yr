#!/usr/bin/env python3

import climetlab as cml


def test_read():
    return
    ds = cml.load_dataset(
        "maelstrom-a1",
        size="5GB",
        dates=["2020-06-26"],
        parameter="air_temperature",
    )
    xds = ds.to_xarray()
    print(xds)


if __name__ == "__main__":
    test_read()
