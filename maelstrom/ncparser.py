""" NetCDF parser module

This module helps to parse contents from NetCDF files, including its grid and variables. It attempts
to read the dimensions and variables of a NetCDF file and rearrange it into a consistent output.
"""

import netCDF4
import numpy as np
import gridpp
import re
import os
import time


def has(filename, variable):
    """Determine if a variable is available (or is diagnosable) from the given file

    Args:
        filename (str|netCDF4.Dataset): Filename or NetCDF file to read from
        variable (str): Name of variable

    Returns:
        bool
    """

    if isinstance(filename, str):
        file = netCDF4.Dataset(filename, "r")
    else:
        file = filename

    value = None

    required_variables = list()
    if variable in file.variables:
        value = True
    elif variable == "precipitation_amount":
        required_variables = ["precipitation_amount_acc"]
    elif variable == "wet_bulb_temperature_2m":
        required_variables = [
            "air_temperature_2m",
            "surface_air_pressure",
            "relative_humidity_2m",
        ]
    elif variable == "dew_point_temperature_2m":
        required_variables = ["air_temperature_2m", "relative_humidity_2m"]
    elif variable == "relative_humidity_2m":
        required_variables = ["air_temperature_2m", "dew_point_temperature_2m"]
    elif variable == "wind_speed_10m":
        required_variables = ["x_wind_10m", "y_wind_10m"]
    elif variable == "wind_speed_of_gust_10m":
        required_variables = ["x_wind_gust_10m", "y_wind_gust_10m"]
    elif variable == "wind_speed_100m":
        required_variables = ["x_wind_100m", "y_wind_100m"]
    elif variable == "wind_direction_10m":
        required_variables = ["x_wind_10m", "y_wind_10m"]
    elif variable == "air_temperature_lowest_level":
        required_variables = ["air_temperature_ml"]
    else:
        value = False

    if value is None:
        for var in required_variables:
            if var not in file.variables:
                value = False

        value = True

    if isinstance(filename, str):
        file.close()

    return value


def get_lookup_variable_name(filename, variable):
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename, "r")
    else:
        file = filename

    deacc = False
    lookup_variable_name = variable
    if variable not in file.variables:
        if (
            variable == "precipitation_amount"
            and "precipitation_amount_acc" in file.variables
        ):
            lookup_variable_name = "precipitation_amount_acc"
            deacc = True
        elif (
            variable == "air_temperature_lowest_level"
            and "air_temperature_ml" in file.variables
        ):
            lookup_variable_name = "air_temperature_ml"

    if isinstance(filename, str):
        file.close()

    return lookup_variable_name, deacc


def get_gridpp_grid_from_points(points):
    """Converts a gridpp.Points object to gridpp.Grid object

    Makes a grid with size (1, L), as this is most memory efficient in C++.

    Args:
        points (gridpp.Points): Input points object

    Returns:
        gridpp.Grid
    """
    grid = gridpp.Grid(
        np.expand_dims(points.get_lats(), 0),
        np.expand_dims(points.get_lons(), 0),
        np.expand_dims(points.get_elevs(), 0),
        np.expand_dims(points.get_lafs(), 0),
    )
    assert grid.size()[0] == 1
    assert grid.size()[1] == points.size()
    return grid


def get_gridpp_grid(filename, latrange=None, lonrange=None):
    """Get a gridpp grid object based on metadata from file

    Arguments:
        filename (str|netCDF4.Dataset): Filename or NetCDF file to read from
        latrange (list of integers): latitude range (min/max)
        lonrange (list of integers): longitude range (min/max)

    Returns:
        gridpp.Grid object

    TODO: Does not read LAF and doesn't handle extra dimensions in altitude (such as time)
    """
    # if get_fileformat(filename) == "ncml":
    #     filenames = get_ncml_filenames(filename)
    #     return get_grid(filenames[0])
    #     raise NotImplementedError()

    if filename is None:
        return None

    if isinstance(filename, str):
        file = netCDF4.Dataset(filename, "r")
    else:
        file = filename
    latitudes, longitudes = get_latlon(file)
    elevs = get_altitude(file)
    lafs = get_laf(file)

    if elevs is None:
        elevs = np.full(latitudes.shape, np.nan)
    if lafs is None:
        lafs = np.full(latitudes.shape, np.nan)

    # Subset by latitudes and longitudes
    if latrange is not None or lonrange is not None:
        Ix, Iy = get_xy_indices(latitudes, longitudes, latrange, lonrange)
        latitudes = latitudes[Iy, :][:, Ix]
        longitudes = longitudes[Iy, :][:, Ix]
        elevs = elevs[Iy, :][:, Ix]
        lafs = lafs[Iy, :][:, Ix]

    if isinstance(filename, str):
        file.close()

    if len(latitudes.shape) == 2:
        return gridpp.Grid(latitudes, longitudes, elevs, lafs)
    else:
        return gridpp.Points(latitudes, longitudes, elevs, lafs)


def is_grid(filename):
    do_close_file = False
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
        do_close_file = True
    else:
        file = filename
        filename = file.filepath()

    def get(file, names, raise_on_error=True):
        for name in names:
            if name in file.variables:
                return file.variables[name]
        if raise_on_error:
            raise Exception("Could not find any of ", ",".join(names) + " in file")
        return None

    lats = get(file, ["latitude", "lat"])
    lons = get(file, ["longitude", "lon"])

    lats_shape = lats.shape
    lons_shape = lons.shape

    # Regular lat/lon grid
    is_regular = (
        len(lats_shape) == 1
        and len(lons_shape) == 1
        and lats.dimensions[0] != lons.dimensions[0]
    )
    if do_close_file:
        file.close()
    if is_regular:
        return True
    if len(lats_shape) == 2:
        return True
    return False


def get_latlon(filename):
    """Retrieve latitudes and longitudes from file

    If grid in input file is a regular lat/long grid, then mesh these into 2D arrays

    Args:
        filename (str|netCDF4.Dataset): Filename or NetCDF file to read from

    Returns:
        np.array: 2D array of latitudes
        np.array: 2D array of longitudes
    """
    do_close_file = False
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
        do_close_file = True
    else:
        file = filename
        filename = file.filepath()

    def get(file, names, raise_on_error=True):
        for name in names:
            if name in file.variables:
                return file.variables[name]
        if raise_on_error:
            raise Exception("Could not find any of ", ",".join(names) + " in file")
        return None

    lats = get(file, ["latitude", "lat"])
    lons = get(file, ["longitude", "lon"])

    # Regular lat/lon grid
    is_regular = (
        len(lats.shape) == 1
        and len(lons.shape) == 1
        and lats.dimensions[0] != lons.dimensions[0]
    )
    lats = lats[:]
    lons = lons[:]
    if is_regular:
        lons, lats = np.meshgrid(lons, lats)
    if do_close_file:
        file.close()

    return lats, lons


def get_altitude(filename):
    """Extracts altitude from file

    Args:
        filename (str|netCDF4.Dataset): Filename or NetCDF file to read from

    Returns:
        np.array: 2D array of altitudes [m] if available, None if not possible
    """
    do_close_file = False
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
        do_close_file = True
    else:
        file = filename
        filename = file.filepath()
    lats, lons = get_latlon(file)
    altitudes = None  # np.nan * np.zeros(lats.shape)
    if "altitude" in file.variables:
        altitudes = get_metadata_field(file.variables["altitude"])
    elif "surface_geopotential" in file.variables:
        altitudes = get_metadata_field(file.variables["surface_geopotential"])
        if altitudes is not None:
            altitudes /= 9.81
    else:
        debug("Cannot find altitude in %s" % filename)
    if do_close_file:
        file.close()

    return altitudes


def get_laf(filename):
    do_close_file = False
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
        do_close_file = True
    else:
        file = filename
        filename = file.filepath()
    lats, lons = get_latlon(file)
    land_area_fraction = None  # np.nan * np.zeros(lats.shape)
    if "land_area_fraction" in file.variables:
        land_area_fraction = get_metadata_field(file.variables["land_area_fraction"])
    if do_close_file:
        file.close()

    return land_area_fraction


def get_global_attributes(filename):
    do_close_file = False
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
        do_close_file = True
    else:
        file = filename
        filename = file.filepath()
    global_attributes = dict()
    for attr in file.ncattrs():
        global_attributes[attr] = getattr(file, attr)

    if do_close_file:
        file.close()

    return global_attributes


def get_gridpp_points(filename):
    """Get a gridpp grid object based on metadata from file

    Args:
        filename (str|netCDF4.Dataset): Filename or NetCDF file to read from

    Returns: gridpp.Grid object

    TODO: Does not read LAF and doesn't handle extra dimensions in altitude (such as time)
    """
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename, "r")
    else:
        file = filename
    latitudes, longitudes = get_latlon(file)

    altitudes = np.nan * np.zeros(latitudes.shape)
    if "altitude" in file.variables:
        altitudes = get_metadata_field(file.variables["altitude"])
    elif "surface_geopotential" in file.variables:
        altitudes = get_metadata_field(file.variables["surface_geopotential"])
        if altitudes is not None:
            altitudes /= 9.81
    else:
        debug("Cannot find altitude in %s" % filename)

    land_area_fraction = np.nan * np.zeros(latitudes.shape)
    if "land_area_fraction" in file.variables:
        land_area_fraction = get_metadata_field(file.variables["land_area_fraction"])

    if isinstance(filename, str):
        file.close()

    return gridpp.Points(latitudes, longitudes, altitudes, land_area_fraction)


def get_forecast_reference_time(filename):
    """Get the forecast reference time of a file

    If a file does not contain it, the first timestep is used

    Args:
        filename (str|netCDF4.Dataset): Filename or NetCDF file to read from

    Returns:
        float: forecast reference time [unixtime seconds]
    """
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
    else:
        file = filename

    if "forecast_reference_time" in file.variables:
        # When forecast_reference_time is dimensionless, then it returns a numpy array of size 0,
        # which causes problems in applications of this function. Therefore convert to an int.
        frt_temp = fill(file.variables["forecast_reference_time"][0])
        if np.isnan(frt_temp):
            frt = float(fill(file.variables["time"][0]))
            var = file.variables["time"]
        else:
            frt = float(frt_temp)
            var = file.variables["forecast_reference_time"]
    else:
        frt = float(fill(file.variables["time"][0]))
        var = file.variables["time"]

    # Convert to seconds
    if hasattr(var, "units"):
        units = var.units
        if re.match("seconds since 1970", units):
            pass
        elif re.match("days since 1970", units):
            frt *= 24 * 3600
        else:
            raise Exception(
                "Cannot parse units for forecast reference time: '%s'" % units
            )

    if isinstance(filename, str):
        file.close()

    return frt


def get_xy(filename):
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
    else:
        file = filename

    if "x" in file.dimensions:
        x = file.variables["x"][:]
        y = file.variables["y"][:]
    elif "X" in file.dimensions:
        x = file.variables["X"][:]
        y = file.variables["Y"][:]
    elif "Xc" in file.dimensions:
        x = file.variables["Xc"][:]
        y = file.variables["Yc"][:]
    elif "latitude" in file.dimensions:
        x = file.variables["longitude"][:]
        y = file.variables["latitude"][:]
    elif "lat" in file.dimensions:
        x = file.variables["lon"][:]
        y = file.variables["lat"][:]
    elif "rlat" in file.dimensions:
        x = file.variables["rlon"][:]
        y = file.variables["rlat"][:]
    else:
        raise Exception("Could not determine x and y dimensions in '%s'" % filename)

    if isinstance(filename, str):
        file.close()
    return x, y


def get_proj(filename):
    if isinstance(filename, str):
        file = netCDF4.Dataset(filename)
    else:
        file = filename
    proj = None
    for variable in file.variables:
        var = file.variables[variable]
        if hasattr(var, "proj4"):
            proj = dict()
            for attr in var.ncattrs():
                proj[attr] = getattr(var, attr)
    if proj is None:
        if "latitude" in file.dimensions and "longitude" in file.dimensions:
            proj = {"proj4": "+proj=longlat +a=6367470 w+e=0 +no_defs"}
        else:
            proj = {"proj4": None}

    if isinstance(filename, str):
        file.close()

    assert "proj4" in proj
    return proj


def get_attributes(var):
    keys = var.ncattrs()
    attributes = dict()
    for key in keys:
        attributes[key] = getattr(var, key)
    return attributes


def get_metadata_field(var):
    num_dims = len(var.shape)
    if num_dims == 5:
        I = np.where(np.isnan(var[0, 0, :, 0, 0]) == 0)[0]
        if len(I) == 0:
            return None
        if I[0] != 0:
            print("Missing metadata on first member, using member %d" % I[0])
        values = var[0, 0, I[0], :, :]
    elif num_dims == 4:
        I = np.where(np.isnan(var[:, 0, 0, 0]) == 0)[0]
        if len(I) == 0:
            return None
        if I[0] != 0:
            print("Missing metadata on first member, using member %d" % I[0])
        values = var[I[0], 0, :, :]
    elif num_dims == 3:
        I = np.where(np.isnan(var[:, 0, 0]) == 0)[0]
        if len(I) == 0:
            return None
        if I[0] != 0:
            print("Missing metadata on first member, using member %d" % I[0])
        values = var[I[0], :, :]
    else:
        values = var[:]

    values = fill(values)
    return values


def fill(values):
    if isinstance(values, np.ma.core.MaskedArray):
        values = values.filled(np.nan)
    return values


def set_missing(values, dtype="f4"):
    """Convert numpy Nans to netCDF default fill values"""
    ret = np.copy(values)
    I = np.isnan(values)
    ret[I] = netCDF4.default_fillvals[dtype]
    return ret


def is_valid(filename):
    return get_fileformat(filename) is not None


# TODO: This is a bit of a sketchy way to deduce file format
def get_fileformat(filename):
    try:
        with netCDF4.Dataset(filename):
            pass
        return "nc"
    except Exception:
        pass
    try:
        get_ncml_filenames(filename)
    except Exception:
        return None
    return "ncml"


def _check_required_variables(file, variables, lookup_variable_name):
    for variable in variables:
        if variable not in file.variables:
            raise Exception(
                f"Cannot diagnose {lookup_variable_name}. Missing {variable}"
            )


def is_real_dim(ncdim):
    """Is this a recognizable dimension?"""
    for dim_type, allowable_names in allowable_dimension_names.items():
        for r in allowable_names:
            if re.match(r, ncdim.name):
                return True
    return False


def extract_time_from_ncml_slice(filename):
    with netCDF4.Dataset(filename, "r") as file:
        curr_time = file.variables["time"][:]
    return curr_time[0]


def extract_data_from_ncml_slice(
    filename, variables, members, latrange=None, lonrange=None
):
    """Extracts data from one ncml file. Assume it only contains one time step

    Args:
        filename (str): Name of NetCDF file
        variables (list): List of variables to lookup
        start_time (int): Truncate file by removing timesteps before this unixtime
        end_time (int): Truncate file by removing timesteps after this unixtime
        members (list): List of member indices (int) to include in output. If not provided, include all.
    Returns:
        time (int): Unixtime of file
        fields (dict): Dictionary of variable->values pairs
        attributes (dict): Dictionary of variable-> attributes pairs
        product_type (int): Product type
        dmiension_values (list): Dimension values for product
    """
    with netCDF4.Dataset(filename) as file:
        # print("Opening %s" % filename)
        if len(file.variables["time"]) > 1:
            raise Exception(
                "Cannot read ncml slice from file '%s', since it contains multiple timesteps (%d)"
                % (filename, len(file.variables["time"]))
            )
        times = file.variables["time"][0].filled(np.nan)

        fields = dict()
        for variable_name in variables:
            field = extract_data_from_file(
                file, variable_name, [0], members, latrange, lonrange
            )

            fields[variable_name] = field

    return times, fields


def read_times(filename, variables):
    times = None
    with netCDF4.Dataset(filename, "r") as file:
        times = file.variables["time"][0].filled(np.nan)

    return times


def get_time_indices(times, start_time, end_time, deacc=False):
    """
    Arguments:
        times (np.array): List of times
        start_time (float): Start time, use None for no limit
        end_time (float): End time, use None for no limit
        deacc (boolean): If true, assume deaccumulation needed, therefore include an extra first
            timestep
    Returns:
        list: List of time indices

    times must be sorted
    """
    times = np.array(times)
    if start_time is not None and start_time > np.max(times):
        return []
    if end_time is not None and end_time < np.min(times):
        return []
    if start_time is None:
        Istart_time = 0
    else:
        Istart_time = np.where(start_time <= times)[0][0]
    if end_time is None:
        Iend_time = len(times)
    else:
        Iend_time = np.where(end_time >= times)[0][-1] + 1
    if deacc:
        Istart_time = max(0, Istart_time - 1)

    Itime = range(Istart_time, Iend_time)
    return Itime


def get_xy_indices(latitudes, longitudes, latrange, lonrange):
    if latrange is not None or lonrange is not None:
        is_within = np.ones(latitudes.shape)
        if lonrange is None:
            is_within = (latitudes >= latrange[0]) & (latitudes <= latrange[1])
        elif latrange is None:
            is_within = (longitudes >= lonrange[0]) & (longitudes <= lonrange[1])
        else:
            is_within = (
                (latitudes >= latrange[0])
                & (latitudes <= latrange[1])
                & (longitudes >= lonrange[0])
                & (longitudes <= lonrange[1])
            )
        Iy = np.where(np.nanmax(is_within, axis=1) > 0)[0]
        Ix = np.where(np.nanmax(is_within, axis=0) > 0)[0]
        debug(
            "Retrieving subset (%d,%d) from full domain of (%d,%d)"
            % ((len(Iy), len(Ix), latitudes.shape[0], latitudes.shape[1]))
        )
    else:
        Iy = range(latitudes.shape[0])
        Ix = range(longitudes.shape[1])
    return Ix, Iy


def is_open_dap_file(filename):
    return filename.find("http") >= 0


def get_dimension_indices(ncvar):
    """Determines what position the time, height, ensemble_member, y, and x dimensions are in the
    variable's dimensions. Will try to automatically determine the name of each dimension.

    Arguments:
        ncvar (netCDF4.Variable): NetCDF4 variable

    Returns:
        Note: Returns None for any dimension that it cannot find a suitable dimension for
        dim_time (int): What position is the time dimensions in?
        dim_level (int):
        dim_member (int):
        dim_y (int):
        dim_x (int):
    """
    # Determine these standardized dimensions:
    dims = ["time", "y", "x", "member"]
    all_names = list(ncvar.dimensions)

    # Put commonly used names for each dimension
    # TODO: Could use the attributes to determine the dimensions
    dim_to_possible_names = {
        k: v for k, v in allowable_dimension_names.items() if k in ["time", "y", "x"]
    }
    for t in ["member", "threshold", "quantile"]:
        dim_to_possible_names[t] = list()
        for v in allowable_dimension_names[t]:
            dim_to_possible_names["member"] += [v]

    dim_to_name = dict()
    dim_to_index = dict()
    for dim in dims:
        dim_to_name[dim] = None
        dim_to_index[dim] = None
        for possible_name in dim_to_possible_names[dim]:
            for i, all_name in enumerate(all_names):
                if re.match(possible_name, all_name):
                    dim_to_name[dim] = all_name
                    dim_to_index[dim] = i

    missing_dims = [d for d in dims if d not in dim_to_name.keys()]
    unknown_dims = [d for d in all_names if d not in dim_to_name.values()]
    # print("Missing dimensions:", missing_dims)
    # print("Unknown dimensions:", unknown_dims)

    # Deal with level dimension
    if len(unknown_dims) == 0:
        dim_to_index["level"] = None
    elif len(unknown_dims) == 1:
        dim_to_index["level"] = all_names.index(unknown_dims[0])
    else:
        raise ValueError("Too many dimensions %s" % unknown_dims)

    return (
        dim_to_index["time"],
        dim_to_index["level"],
        dim_to_index["member"],
        dim_to_index["y"],
        dim_to_index["x"],
    )
