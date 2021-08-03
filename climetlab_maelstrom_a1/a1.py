#!/usr/bin/env python3
from __future__ import annotations

import os

import climetlab as cml
import pandas as pd
import xarray as xr
from climetlab import Dataset
from climetlab.normalize import DateListNormaliser
from climetlab.sources.file import File

__version__ = "0.1.0"


class A1(Dataset):
    name = "Nordic public weather forecast dataset"
    home_page = "https://github.com/metno/maelstrom-a1"
    licence = "-"
    documentation = "-"
    citation = (
        "Nipen, T. N., Seierstad, I. A., Lussana, C., Kristiansen, J., & Hov, Ø. (2020). "
        "Adopting Citizen Observations in Operational Weather Prediction, Bulletin of the "
        "American Meteorological Society, 101(1), E43-E57."
    )
    terms_of_use = (
        "By downloading data from this dataset, you agree to the terms and conditions defined at "
        "https://github.com/metno/maelstrom_a1/LICENSE. "
        "If you do not agree with such terms, do not download the data. "
    )

    all_datelist = [
        i.strftime("%Y-%m-%d")
        for i in pd.date_range(start="2017-01-01", end="2019-03-01", freq="1D")
    ] + [
        i.strftime("%Y-%m-%d")
        for i in pd.date_range(start="2019-03-06", end="2020-12-31", freq="1D")
    ]
    default_datelist = all_datelist

    # @normalize_args(size=["300MB", "5GB"], parameter=["air_temperature", "precipitation_amount"])
    def __init__(
        self,
        size,
        parameter,
        dates=None,
        location="https://storage.ecmwf.europeanweather.cloud/MAELSTROM_AP1/",
        pattern="{parameter}_{size}/{date}T00Z.nc",
        verbose=False,
    ):
        """
        Arguments:
            size (str): Datasize, one of "5GB" and "5TB"
            parameter (str): Predictand variable, one of "air_temperature", and "precipitation_amount"
            dates (list): Only extract these dates, list of "YYYY-MM-DD" strings
            location (str): Storage location (URL, or local file directory)
            pattern (str): Pattern for filenames
            verbose (bool): Show debug statements if True
        """
        self.size = size
        self.parameter = parameter
        self.verbose = verbose

        is_url = location.find("://") >= 0
        self.debug(f"Is this a URL dataset? {is_url}")

        if dates is None:
            dates = self.default_datelist
        self.dates = self.parse_dates(dates)
        self.debug(f"Number of dates to load {len(self.dates)}")

        options = {
            "concat_characters": False,
            "data_vars": ["time", "predictors", "target"],
            "drop_variables": ["leadtime_predictor"],
        }

        if not is_url:
            # Use data stored locally
            request = dict(size=self.size, parameter=self.parameter)
            filenames = [
                location + pattern.format(date=date, **request) for date in self.dates
            ]
            filenames = [f for f in filenames if os.path.exists(f)]
            self.debug(f"Number of files found {len(filenames)}:")
            self.debug(f"{filenames}")

            files = [File(f) for f in filenames]
            if len(files) == 0:
                raise RuntimeError(
                    f"No available files matching pattern '{location}{pattern}'"
                )

            self.source = cml.load_source(
                "multi", files, merger=Merger(options=options)
            )
        else:
            # Download from the cloud
            request = dict(size=self.size, parameter=self.parameter, date=self.dates)
            self.debug(f"Request parameters {request}")
            self.source = cml.load_source(
                "url-pattern",
                location + pattern,
                request,
                merger=Merger(options=options),
            )

    @staticmethod
    def parse_dates(dates):
        """Reads dates (e.g. from pandas, or YYYY-MM-DD) and converts them to YYYYMMDD"""
        dates = DateListNormaliser("%Y-%m-%d")(dates)
        if dates is None:
            dates = A1.default_datelist
        for d in dates:
            if d not in A1.all_datelist:
                print(f"Warning: Date {d} is not available")
        dates = [d for d in dates if d in A1.all_datelist]
        dates = DateListNormaliser("%Y%m%d")(dates)
        return dates

    def debug(self, message):
        if self.verbose:
            print("DEBUG: ", message)


class Merger:
    def __init__(self, engine="netcdf4", concat_dim="record", options=None):
        self.engine = engine
        self.concat_dim = concat_dim
        self.options = options if options is not None else {}

    def to_xarray(self, paths, **kwargs):
        return xr.open_mfdataset(
            paths,
            engine=self.engine,
            concat_dim=self.concat_dim,
            combine="nested",
            **self.options,
        )
