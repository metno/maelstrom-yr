#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import collections

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
        normalize=False,
        predict_diff=False,
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
            normalize (bool): If true, normalize the data
            predict_diff (bool): If true, change the target to be the difference between the target
                and raw forecast
            verbose (bool): Show debug statements if True
        """
        if size not in ["5GB", "5TB"]:
            raise ValueError("invalid size '{size}'")

        if parameter not in ["air_temperature"]:
            raise ValueError("invalid parameter '{parameter}'")

        self.size = size
        self.parameter = parameter
        self.probabilistic_target = probabilistic_target
        self.do_normalize = normalize
        self.do_predict_diff = predict_diff
        self.verbose = verbose

        is_url = location.find("://") >= 0
        self.debug(f"Is this a URL dataset? {is_url}")

        self.dates = self.get_available_dates(dates)
        self.debug(f"Number of dates to load {len(self.dates)}")

        x_array_options = {
            # Needed to deal with char dimension in metadata variables
            "concat_characters": False,
            # "data_vars": ["time", "predictors", "target_mean"],
            # Without this, fails with pandas 1.3.1
            # "drop_variables": ["static_predictors", "target_std"],
            # Run preprocess steps on each netCDF file 
            # "decode_timedelta": False,
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

    @classmethod
    def get_all_dates(cls):
        """Returns all available dates (list of strings in format YYYYMMDD)"""
        all_datelist = [
            i.strftime("%Y-%m-%d")
            for i in pd.date_range(start="2020-03-01", end="2021-02-28", freq="1D")
        ] + [
            i.strftime("%Y-%m-%d")
            for i in pd.date_range(start="2021-03-01", end="2022-02-28", freq="1D")
        ]

        # Remove missing dates
        all_datelist.remove("2020-05-22")
        all_datelist.remove("2020-05-26")
        all_datelist.remove("2020-05-30")
        all_datelist.remove("2020-08-05")
        all_datelist.remove("2021-08-05")
        all_datelist.remove("2022-01-15")
        all_datelist=DateListNormaliser(all_datelist)
        return all_datelist

    @classmethod
    def get_normalization(cls):
        """Returns a dictionary with normalization coefficients (mean, std)"""
        normalization = collections.defaultdict(lambda: [0.0, 1.0])
        normalization["air_temperature_0.1_2m"] = [4.725014929970105, 7.566987847083937]
        normalization["air_temperature_0.9_2m"] = [6.112570554018021, 7.521962731122694]
        normalization["air_temperature_2m"] = [5.388952960570653, 7.4335476655246735]
        normalization["bias_yesterday"] = [-0.07066702128698428, 0.7485341580688785]
        normalization["cloud_area_fraction"] = [0.6884219621618589, 0.40055377741854187]
        normalization["precipitation_amount"] = [0.07837766820254426, 0.36662233737977684]
        normalization["x_wind_10m"] = [0.4250524006783962, 4.6850994324021595]
        normalization["y_wind_10m"] = [-0.7272756993770599, 5.11310843905376]
        normalization["altitude"] = [110.0174789428711, 218.2789258799505]
        normalization["analysis_std"] = [0.5121386324365934, 0.40215385148028904]
        normalization["bias_recent"] = [-0.007340590585954487, 0.5344001060184195]
        normalization["land_area_fraction"] = [0.45299264788627625, 0.49328521353773686]
        normalization["model_altitude"] = [109.62958717346191, 213.4803743151553]
        normalization["model_laf"] = [0.43093839287757874, 0.48621477236654775]
        return normalization

    @property
    def datestr(self):
        strings = list()
        for date in self.dates:
            strings += ["%sT%02dZ" % (date, self.get_hour_from_date(date))]
        return strings

    @staticmethod
    def get_available_dates(dates):
        """Reads dates (e.g. from pandas, or YYYY-MM-DD) and converts them to YYYYMMDD"""
        dates = DateListNormaliser(dates)
        if dates is None:
            dates = Yr.get_all_dates()
        else:
            for d in dates:
                if d not in Yr.get_all_dates():
                    print(f"Warning: Date {d} is not available")
            dates = [d for d in dates if d in Yr.get_all_dates()]
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
            coords["target"] = np.array(["mean", "std"], object)
        else:
            coords["target"] = np.array(["mean"], object)

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

        if self.do_predict_diff:
            I = np.where(coords["predictor"] == "air_temperature_2m")[0][0]
            data_vars["targets"][1][..., 0] -= data_vars["predictors"][1][..., I]

        if self.do_normalize:
            for i, name in enumerate(coords["predictor"]):
                Yr.normalize(data_vars["predictors"][1][..., i], name)

        new_ds = xr.Dataset(data_vars, coords)
        return new_ds

    @staticmethod
    def denormalize(array, name):
        array *= Yr.get_normalization()[name][1]
        array += Yr.get_normalization()[name][0]

    @staticmethod
    def normalize(array, name):
        if not isinstance(name, str):
            raise TypeError("name must be type 'str'")

        array -= Yr.get_normalization()[name][0]
        array /= Yr.get_normalization()[name][1]


class Merger:
    def __init__(
        self, engine="netcdf4", concat_dim="time", x_array_options={}, tf_options={}
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
