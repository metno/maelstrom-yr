#!/usr/bin/env python3
from __future__ import annotations

import glob
import os

import climetlab as cml
import pandas as pd
import xarray as xr
from climetlab import Dataset
from climetlab.decorators import normalize
from climetlab.sources.file import File
import maelstrom


@normalize("x","date-list(%Y%m%d)")
def DateListNormaliser(x):
    return x


class Yr(Dataset):
    name = "Nordic public weather forecast dataset"
    home_page = "https://github.com/metno/maelstrom-yr"
    licence = "-"
    documentation = "-"
    citation = (
        "Nipen, T. N., Seierstad, I. A., Lussana, C., Kristiansen, J., & Hov, Ø. (2020). "
        "Adopting Citizen Observations in Operational Weather Prediction, Bulletin of the "
        "American Meteorological Society, 101(1), E43-E57."
    )
    terms_of_use = (
        "By downloading data from this dataset, you agree to the terms and conditions defined at "
        "https://github.com/metno/maelstrom_yr/LICENSE. "
        "If you do not agree with such terms, do not download the data. "
    )

    all_datelist = [
        i.strftime("%Y-%m-%d")
        for i in pd.date_range(start="2020-03-01", end="2021-02-28", freq="1D")
    ] + [
        i.strftime("%Y-%m-%d")
        for i in pd.date_range(start="2021-03-01", end="2022-02-28", freq="1D")
    ]
    all_datelist=DateListNormaliser(all_datelist)
    default_datelist=all_datelist

    def __init__(
        self,
        size,
        parameter,
        dates=None,
        location="https://object-store.os-api.cci1.ecmwf.int/maelstrom-ap1/",
        pattern="{parameter}/{size}/{date}T{hour}Z.nc",
        x_range=None,
        y_range=None,
        limit_leadtimes=None,
        limit_predictores=None,
        probabilistic_target=False,
        normalization=None,
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
        if size not in ["5GB", "5TB"]:
            raise ValueError("invalid size '{size}'")

        if parameter not in ["air_temperature", "precipitation_amount"]:
            raise ValueError("invalid parameter '{parameter}'")

        self.size = size
        self.parameter = parameter
        self.verbose = verbose

        is_url = location.find("://") >= 0
        self.debug(f"Is this a URL dataset? {is_url}")

        if dates is None:
            dates = self.default_datelist
        self.dates = self.parse_dates(dates)
        self.debug(f"Number of dates to load {len(self.dates)}")

        x_array_options = {
            # Needed to deal with char dimension in metadata variables
            "concat_characters": False,
            "data_vars": ["time", "predictors", "target_mean"],
            # Without this, fails with pandas 1.3.1
            "drop_variables": ["static_predictors", "target_std"],
        }

        tf_options = {}

        if not is_url:
            # Use data stored locally
            request = dict(size=self.size, parameter=self.parameter)
            filenames = list()
            for date in self.dates:
                hour = self.get_hour_str(date)
                filenames += glob.glob(
                    location + pattern.format(date=date, hour=hour, **request)
                )
            filenames = [f for f in filenames if os.path.exists(f)]
            self.debug(f"Number of files found {len(filenames)}:")
            self.debug(f"{filenames}")

            files = [File(f) for f in filenames]
            if len(files) == 0:
                raise RuntimeError(
                    f"No available files matching pattern '{location}{pattern}'"
                )

            self.source = cml.load_source(
                "multi",
                files,
                merger=Merger(x_array_options=x_array_options, tf_options=tf_options),
            )
        else:
            # Download from the cloud
            hours = [self.get_hour_str(date) for date in self.dates]
            request = dict(
                size=self.size, parameter=self.parameter, date=self.dates, hour=hours
            )
            self.debug(f"Request parameters {request}")
            self.source = cml.load_source(
                "url-pattern",
                location + pattern,
                request,
                merger=Merger(x_array_options=x_array_options, tf_options=tf_options),
            )

    @staticmethod
    def parse_dates(dates):
        """Reads dates (e.g. from pandas, or YYYY-MM-DD) and converts them to YYYYMMDD"""
        dates = DateListNormaliser(dates)
        if dates is None:
            dates = Yr.default_datelist
        for d in dates:
            if d not in Yr.all_datelist:
                print(f"Warning: Date {d} is not available")
        dates = [d for d in dates if d in Yr.all_datelist]
        return dates

    def get_hour_str(self, date_str):
        """Get corresponding forecast initialization hour for a given date (YYYYMMDD)"""
        date = int(date_str)
        day = date % 100
        if day % 4 == 1:
            hour = 3
        elif day % 4 == 2:
            hour = 9
        elif day % 4 == 3:
            hour = 15
        elif day % 4 == 0:
            hour = 21
        return "%02d" % hour

    def debug(self, message):
        if self.verbose:
            print("DEBUG: ", message)


class Merger:
    def __init__(
        self, engine="netcdf4", concat_dim="record", x_array_options={}, tf_options={}
    ):
        self.engine = engine
        self.concat_dim = concat_dim
        self.x_array_options = x_array_options
        self.tf_options = tf_options

    def to_xarray(self, paths, **kwargs):
        return xr.open_mfdataset(
            paths,
            engine=self.engine,
            concat_dim=self.concat_dim,
            combine="nested",
            **self.x_array_options,
        )

    def to_tfdataset(self, paths, **kwargs):
        """Returns a tensorflow dataset object"""
        loader = maelstrom.loader.FileLoader(paths)
        return loader.get_dataset()
