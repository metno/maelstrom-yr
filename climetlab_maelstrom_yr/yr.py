#!/usr/bin/env python3
from __future__ import annotations

import glob
import os

import climetlab as cml
import pandas as pd
import xarray as xr
import numpy as np

from climetlab import Dataset
from climetlab.decorators import normalize
from climetlab.sources.file import File


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
        parameter="air_temperature",
        dates=None,
        location="https://object-store.os-api.cci1.ecmwf.int/maelstrom-ap1/",
        pattern="{parameter}/{size}/{datestr}.nc",
        x_range=None,
        y_range=None,
        limit_leadtimes=None,
        limit_predictors=None,
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
            probabilistic_target (bool): If true, include target std as the second target parameter
            normalization (bool): If true, normalize the data
            verbose (bool): Show debug statements if True
        """
        if size not in ["5GB", "5TB"]:
            raise ValueError("invalid size '{size}'")

        if parameter not in ["air_temperature"]:
            raise ValueError("invalid parameter '{parameter}'")

        self.size = size
        self.parameter = parameter
        self.normalization = normalization
        self.probabilistic_target = probabilistic_target
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
            # "data_vars": ["time", "predictors", "target_mean"],
            # Without this, fails with pandas 1.3.1
            # "drop_variables": ["static_predictors", "target_std"],
            # Run preprocess steps on each netCDF file 
            "preprocess": self.preprocess,
        }
        tf_options = {}

        merger = Merger(x_array_options=x_array_options, tf_options=tf_options)

        if is_url:
            # Download from the cloud
            request = dict(
                size=self.size, parameter=self.parameter, datestr=self.datestr
            )
            self.debug(f"Request parameters {request}")
            self.source = cml.load_source(
                "url-pattern",
                location + pattern,
                request,
                merger=merger,
            )
        else:
            # Use data stored locally
            request = dict(size=self.size, parameter=self.parameter)
            filenames = list()
            for datestr in self.datestr:
                filenames += glob.glob(
                    location + pattern.format(datestr=datestr, **request)
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
                merger=merger,
            )

    @property
    def datestr(self):
        strings = list()
        for date in self.dates:
            strings += ["%sT%02dZ" % (date, self.get_hour_from_date(date))]
        return strings

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

    def get_hour_from_date(self, date_str):
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
        return hour

    def debug(self, message):
        if self.verbose:
            print("DEBUG: ", message)

    def preprocess(self, ds):
        self.debug(f"Preprocessing {ds.time.values}")

        # Set up coordinate variables
        coords = dict()
        copy_coords = ["latitude", "longitude"]
        for coord in copy_coords:
            coords[coord] = ds.variables[coord]
        coords["predictor"] = np.concatenate((ds["predictor"].values, ds["static_predictor"].values))
        if self.probabilistic_target:
            coords["target"] = np.array(["mean", "std"], np.object)
        else:
            coords["target"] = np.array(["mean"], np.object)

        # Copy variables from input
        data_vars = dict()
        copy_vars = ["time", "x", "y", "leadtime"]
        for var in copy_vars:
            data_vars[var] = (ds.variables[var].dims, ds.variables[var].values, ds.variables[var].attrs)

        # Merge predictors and static_predictors
        p = ds.variables["predictors"]
        s = np.expand_dims(ds.variables["static_predictors"], axis=0)
        s = np.repeat(s, p.shape[0], axis=0)
        data_vars["predictors"] = (("leadtime", "y", "x", "predictor"), np.concatenate((p, s), axis=-1))

        # Add target dimension to targets
        if self.probabilistic_target:
            mean = np.expand_dims(ds.variables["target_mean"], -1)
            std = np.expand_dims(ds.variables["target_std"], -1)
            targets = np.concatenate((mean, std), axis=-1)
            data_vars["targets"] = (("leadtime", "y", "x", "target"), targets)
        else:
            data_vars["targets"] = (("leadtime", "y", "x", "target"), np.expand_dims(ds.variables["target_mean"], -1))

        new_ds = xr.Dataset(data_vars, coords)
        return new_ds


class Merger:
    def __init__(
        self, engine="netcdf4", concat_dim="record", x_array_options={}, tf_options={}
    ):
        self.engine = engine
        self.concat_dim = concat_dim
        self.x_array_options = x_array_options
        self.tf_options = tf_options

    def to_xarray(self, paths, **kwargs):
        ds = xr.open_mfdataset(
            paths,
            engine=self.engine,
            concat_dim=self.concat_dim,
            combine="nested",
            **self.x_array_options,
        )

        return ds

    def to_tfdataset(self, paths, **kwargs):
        """Returns a tensorflow dataset object"""
        import tensorflow as tf
        ds = self.to_xarray(paths, **kwargs)
        def gen(pred, target):
            def _gen():
                num_records = pred.shape[0]
                for i in range(num_records):
                    yield pred[i, ...], target[i, ...]
            return _gen
        pred = ds["predictors"]
        target = ds["targets"]
        output_signature = (tf.TensorSpec(shape=(pred.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(target.shape[1:]), dtype=tf.float32))
        return tf.data.Dataset.from_generator(gen(pred, target), output_signature=output_signature)

        # dataset = tf.data.Dataset.from_tensor_slices((ds["predictors"], ds["targets"]))
        # return dataset
