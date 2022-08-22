import gridpp
import netCDF4
import numpy as np
import os
import re
import scipy.stats
import datetime

import maelstrom


class VerifFile:
    """A file for producing output readable by Verif"""

    def __init__(
        self,
        filename,
        points,
        leadtimes,
        forecast_reference_hours=None,
        station_ids=None,
        units=None,
        variable_name=None,
        thresholds=[],
        quantiles=[],
        extra_attributes=dict(),
        compression=True,
        append=False,
    ):
        """Initialize a verif file

        Args:
            filename (str): Filename to write verif file to
            points (gridpp.Points): Points object with station metadata
            leadtimes (np.array): Array of forecast leadtimes [h]
            forecast_reference_hours (np.array): Array of what hours of the day forecasts are
                initialized on. If provided, then forecast reference times will be generated for all
                possible times that observations are available for. If not provided, the forecast
                reference times from forecasts are used. That is, if a pure observation verif file
                is to be produced, ensure that forecast_reference_hours is provided, otherwise the
                file be empty.
            station_ids (np.array): Array of station IDs
            units (str): Units of forecast variable
            variable_name (str): Name of the variable (used in verif legend)
            thresholds (np.array): Array of thresholds (needed if add_threshold_forecast is to be
                used.
            quantiles (np.array): Array of quantiles (needed if add_quantile_forecast is to be used.
            extra_attributes (dict): Any extra global attributes to be put in the verif file
            compression (boolean): Should the output file be compressed?
            append (boolean): Should the file be appended to (instead of created from scratch)?
        """
        if not maelstrom.util.is_list(leadtimes):
            raise ValueError("Leadtimes must be a list/array")

        if forecast_reference_hours is not None and not maelstrom.util.is_list(
            forecast_reference_hours
        ):
            raise ValueError("forecast_reference_hours must be a list/array")

        self.filename = filename
        self.points = points
        self.fcst = dict()
        self.quantile_fcst = dict()
        self.threshold_fcst = dict()
        self.thresholds = thresholds
        self.quantiles = quantiles
        self.units = units
        self.variable_name = variable_name
        self.obs = dict()
        self.std = dict()
        self.leadtimes = leadtimes
        self.forecast_reference_hours = forecast_reference_hours
        self.station_ids = station_ids
        self.append = append
        self.extra_attributes = extra_attributes
        self.compression = compression

    @staticmethod
    def from_existing(
        ifilename,
        ofilename=None,
        forecast_reference_hours=None,
        extra_attributes=dict(),
        compression=True,
    ):
        points = maelstrom.ncparser.get_gridpp_points(ifilename)
        units = variable_name = None

        append = False
        if ofilename is None:
            ofilename = ifilename
            append = True

        with netCDF4.Dataset(ofilename, "r") as file:
            thresholds = []
            quantiles = []
            if "threshold" in file.variables:
                thresholds = file.variables["threshold"]
            if "quantile" in file.variables:
                quantiles = file.variables["quantile"]
            if hasattr(file, "units"):
                units = file.units
            # Currently not needed
            # if hasattr(file, "long_name"):
            #     long_name = file.variables_name
            leadtimes = file.variables["leadtime"][:]
            station_ids = file.variables["location"][:]
            obs = file.variables["obs"][:]
            times = file.variables["time"][:]
            a, b = np.meshgrid(times, leadtimes * 3600)
            valid_times = a + b
            valid_times = np.transpose(valid_times)
        file = VerifFile(
            ofilename,
            points,
            leadtimes,
            forecast_reference_hours,
            station_ids,
            units,
            variable_name,
            thresholds,
            quantiles,
            extra_attributes,
            compression,
            append,
        )
        unique_valid_times = np.sort(np.unique(valid_times.flatten()))
        for t, valid_time in enumerate(unique_valid_times):
            I = np.where(valid_times == valid_time)
            curr_obs = obs[I[0][0], I[1][0], :]
            if np.sum(np.isnan(curr_obs) == 0) > 0:
                file.add_observations(valid_time, curr_obs)
        return file

    @staticmethod
    def from_dataset(
        dataset,
        filename,
        leadtimes,
        forecast_reference_hours=None,
        station_ids=None,
        thresholds=None,
        quantiles=None,
        extra_attributes=dict(),
        compression=True,
    ):
        points = dataset.gridpp_points
        ofile = VerifFile(
            filename,
            points,
            leadtimes,
            forecast_reference_hours,
            station_ids,
            dataset.units,
            dataset.variable,
            thresholds=thresholds,
            quantiles=quantiles,
            extra_attributes=extra_attributes,
            compression=compression,
            append=False,
        )
        for t, curr_time in enumerate(dataset.times):
            ofile.add_observations(curr_time, dataset.values[t, :])
        return ofile

    def add_forecast(self, forecast_reference_time, values):
        if len(values.shape) != 2:
            raise Exception("Forecasts must be 2D")
        if (
            values.shape[0] != len(self.leadtimes)
            or values.shape[1] != self.points.size()
        ):
            raise Exception(
                "Forecasts must be size %d,%d not %d,%d"
                % (
                    len(self.leadtimes),
                    self.points.size(),
                    values.shape[0],
                    values.shape[1],
                )
            )
        self.fcst[forecast_reference_time] = values

    def add_threshold_forecast(self, forecast_reference_time, values, thresholds=None):
        if len(self.thresholds) == 0:
            raise Exception(
                "Cannot add threshold forecasts, since verif file is initialized without any"
            )
        if len(values.shape) != 3:
            raise Exception("Quantile forecasts must be 3D")

        if thresholds is not None:
            if forecast_reference_time not in self.threshold_fcst:
                temp = np.nan * np.zeros(
                    [len(self.leadtimes), self.points.size(), len(self.thresholds)]
                )
                self.threshold_fcst[forecast_reference_time] = temp
            Iin, Iout = maelstrom.util.get_common_indices(thresholds, self.thresholds)
            self.threshold_fcst[forecast_reference_time][:, :, Iout] = values[:, :, Iin]
        else:
            if values.shape[2] != len(self.thresholds):
                raise Exception(
                    "Threshold values must be size %d,%d,%d not %d,%d,%d"
                    % (
                        len(self.leadtimes),
                        self.points.size(),
                        len(self.thresholds),
                        values.shape[0],
                        values.shape[1],
                        values.shape[2],
                    )
                )
            self.threshold_fcst[forecast_reference_time] = values

    def add_quantile_forecast(self, forecast_reference_time, values, quantiles=None):
        if len(self.quantiles) == 0:
            raise Exception(
                "Cannot add quantile forecasts, since verif file is initialized without any"
            )
        if len(values.shape) != 3:
            raise Exception("Threshold forecasts must be 3D")

        if quantiles is not None:
            if forecast_reference_time not in self.quantile_fcst:
                temp = np.nan * np.zeros(
                    [len(self.leadtimes), self.points.size(), len(self.quantiles)]
                )
                self.quantile_fcst[forecast_reference_time] = temp
            Iin, Iout = maelstrom.util.get_common_indices(quantiles, self.quantiles)
            self.quantile_fcst[forecast_reference_time][:, :, Iout] = values[:, :, Iin]
        else:
            if values.shape[2] != len(self.quantiles):
                raise Exception(
                    "Threshold values must be size %d,%d,%d not %d,%d,%d"
                    % (
                        len(self.leadtimes),
                        self.points.size(),
                        len(self.quantiles),
                        values.shape[0],
                        values.shape[1],
                        values.shape[2],
                    )
                )
            self.quantile_fcst[forecast_reference_time] = values

    def add_observations(self, curr_time, values):
        self.obs[curr_time] = values

    def add_std(self, curr_time, values):
        self.std[curr_time] = values

    def write(self):
        times_fcst = np.sort(list(self.fcst.keys()))
        times_obs = np.sort(list(self.obs.keys()))

        if self.forecast_reference_hours is None:
            times = times_fcst
        else:
            # Determine which forecast reference times the observations are available for
            # This is only needed if forecast_refernce_hours is explicitly set, otherwise only the
            # forecast_reference_times from forecasts are used
            forecast_reference_days_obs = times_obs // 86400 * 86400
            x, y = np.meshgrid(times_obs, self.leadtimes)
            init_times_from_obs = np.sort(np.unique(x.flatten() - y.flatten() * 3600))
            init_times_hours = init_times_from_obs / 3600 % 24
            I = np.where(np.in1d(init_times_hours, self.forecast_reference_hours))[0]

            times = np.sort(
                np.unique(np.concatenate([times_fcst, init_times_from_obs[I]]))
            )

        if not self.append or not os.path.exists(self.filename):
            maelstrom.util.create_directory(self.filename)
            file = netCDF4.Dataset(self.filename, "w")
            file.createDimension("time", None)
            file.createDimension("leadtime", len(self.leadtimes))
            file.createDimension("location", self.points.size())
            if len(self.thresholds) > 0:
                # debug("Adding %d thresholds" % len(self.thresholds))
                file.createDimension("threshold", len(self.thresholds))
                file.createVariable("threshold", "f4", ["threshold"])
                file.variables["threshold"][:] = self.thresholds
                file.createVariable(
                    "cdf", "f4", ("time", "leadtime", "location", "threshold")
                )
                file.variables["cdf"][:] = 1
            if len(self.quantiles) > 0:
                # debug("Adding %d quantiles" % len(self.quantiles))
                file.createDimension("quantile", len(self.quantiles))
                file.createVariable("quantile", "f4", ["quantile"])
                file.variables["quantile"][:] = self.quantiles
                file.createVariable(
                    "x", "f4", ("time", "leadtime", "location", "quantile")
                )
                file.variables["x"][:] = 1

            var = file.createVariable("time", "f8", ("time",))
            var[:] = times
            var = file.createVariable("leadtime", "f4", ("leadtime",))
            var[:] = self.leadtimes
            vLocation = file.createVariable("location", "i4", ("location",))
            vLat = file.createVariable("lat", "f4", ("location",))
            vLon = file.createVariable("lon", "f4", ("location",))
            vElev = file.createVariable("altitude", "f4", ("location",))
            file.createVariable(
                "fcst", "f4", ("time", "leadtime", "location"), zlib=self.compression
            )
            file.createVariable(
                "obs", "f4", ("time", "leadtime", "location"), zlib=self.compression
            )

            if self.variable_name is not None:
                file.long_name = self.variable_name
            if self.units is not None:
                file.units = self.units

            for key, value in self.extra_attributes.items():
                setattr(file, key, value)

            if self.station_ids is None:
                vLocation[:] = range(self.points.size())
            else:
                vLocation[:] = self.station_ids
            vLat[:] = self.points.get_lats()
            vLon[:] = self.points.get_lons()
            vElev[:] = self.points.get_elevs()
        else:
            # Check consistency with existing file
            file = netCDF4.Dataset(self.filename, "a")

        # Write forecasts, quantiles, thresholds
        # Expand the time dimension as needed
        all_unixtimes = list()
        for s in [self.fcst, self.std, self.quantile_fcst, self.threshold_fcst]:
            all_unixtimes += list(s.keys())
        all_unixtimes = np.unique(np.sort(all_unixtimes))
        for t, unixtime in enumerate(all_unixtimes):
            if unixtime in file.variables["time"][:]:
                tout = np.where(unixtime == file.variables["time"][:])[0][0]
            else:
                tout = len(file.variables["time"][:])
            if unixtime in self.fcst:
                file.variables["fcst"][tout, ...] = maelstrom.ncparser.fill(
                    self.fcst[unixtime]
                )

            if unixtime in self.std and unixtime in self.fcst:
                mean = maelstrom.ncparser.fill(self.fcst[unixtime])
                std = maelstrom.ncparser.fill(self.std[unixtime])
                if len(self.quantiles):
                    for i, quantile in enumerate(self.quantiles):
                        spread_factor = scipy.stats.norm.ppf(quantile)
                        file.variables["x"][tout, :, :, i] = mean + spread_factor * std
                if len(self.thresholds) > 0:
                    for i, threshold in enumerate(self.thresholds):
                        q = (threshold - mean) / std
                        file.variables["cdf"][tout, :, :, i] = scipy.stats.norm.cdf(q)

            if len(self.quantiles) > 0:
                if unixtime in self.quantile_fcst:
                    file.variables["x"][tout, :, :, :] = self.quantile_fcst[unixtime][:]
            if len(self.thresholds) > 0:
                if unixtime in self.threshold_fcst:
                    file.variables["cdf"][tout, :, :, :] = self.threshold_fcst[
                        unixtime
                    ][:]
            file.variables["time"][tout] = unixtime

        # Write observations
        a, b = np.meshgrid(file.variables["time"][:], np.array(self.leadtimes) * 3600)
        valid_times = a + b
        valid_times = np.transpose(valid_times)
        unique_valid_times = np.unique(np.sort(valid_times[:]))
        for t, valid_time in enumerate(unique_valid_times):
            if valid_time in self.obs.keys():
                curr_obs = maelstrom.ncparser.fill(self.obs[valid_time])
                I = np.where(valid_times == valid_time)
                for i in range(len(I[0])):
                    file.variables["obs"][I[0][i], I[1][i], :] = curr_obs

        file.Conventions = "verif_1.0.0"

        # Sort times, since they may not be in the correct order anymore
        It = np.argsort(file.variables["time"][:])
        for name in ["time", "obs", "fcst", "x", "cdf"]:
            if name in file.variables:
                file.variables[name][:] = file.variables[name][It, ...]

                # Change NaNs to netcdf missing values
                file.variables[name][:] = maelstrom.ncparser.set_missing(
                    file.variables[name][:]
                )
        file.close()

        # self.obs.clear()  # Don't clear observations, since we want to cache these in case
        # we add new forecasts with leadtimes that need these observations
        self.threshold_fcst.clear()
        self.quantile_fcst.clear()
        self.fcst.clear()


# Convert numbers to a tuple suitable as a key in a dictionary
def tokey(array, round=2):
    return tuple([np.round(float(x), round) for x in array[:]])


def get_dimension_names(file, prefix, standard_name=None):
    """Get dimension names that have a certain prefix
    Arguments:
        file (netCDF4.Dataset):
        prefix (str): Get dimensions starting with this
    Returns:
        dict(): key: dimension values, value: dimension name
    """
    dimnames = dict()
    for dim in file.dimensions:
        name = dim
        n = len(prefix)
        if len(name) > n and name[0:n] == prefix:
            var = file.variables[name]
            if standard_name is not None:
                if hasattr(var, "standard_name") and var.standard_name != standard_name:
                    continue
            dim_values = tokey(var[:])
            dimnames[dim_values] = name
    return dimnames


def is_grid(grid):
    return not isinstance(grid.size(), int)


def get_projection_varname(proj, is_grid):
    projection_varname = None
    projection_name = None
    if is_grid and proj is not None:
        match = re.match("\+proj=([\S]*) ", proj["proj4"])
        if match:
            acronym = match.group(1)
            if acronym == "longlat":
                projection_name = "latitude_longitude"
            elif acronym == "lcc":
                projection_name = "lambert_conformal_conic"
            projection_varname = "projection_%s" % acronym
        else:
            projection_varname = "projection"
    return projection_varname, projection_name
