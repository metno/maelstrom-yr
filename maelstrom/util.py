import calendar
import datetime
import numbers
import numpy as np
import os
import resource

import gridpp
import psutil
import tqdm


def verify(
    filename, output, times, leadtimes, grid, probabilistic, variable="air_temperature"
):
    import yrlib

    if variable == "air_temperature":
        frost_variable = "air_temperature"
        units = "C"
    else:
        raise Exception("Undefined variable %s" % variable)

    num_leadtimes = len(leadtimes)

    # Find stations within domain
    station_ids, station_lats, station_lons, station_elevs = yrlib.locations.get()
    points = yrlib.locations.get_gridpp_points()
    dist = gridpp.distance(grid, points)
    valid_stations = np.where(dist < 4000)[0]
    print(station_ids)
    valid_station_ids = ["SN%d" % id for id in station_ids[valid_stations]]
    num_valid = len(valid_stations)
    station_ids = station_ids[valid_stations]
    points = gridpp.Points(
        station_lats[valid_stations],
        station_lons[valid_stations],
        station_elevs[valid_stations],
    )
    print("Stations within domain: %d/%d" % (num_valid, len(dist)))

    # Set up verif file
    kwargs = {"units": units, "variable_name": variable}
    if probabilistic:
        kwargs["quantiles"] = [0.1, 0.9]

    file = yrlib.output.VerifFile(
        filename, points, leadtimes // 3600, station_ids, **kwargs
    )

    for t in tqdm.tqdm(range(len(times))):
        frt = times[t]
        temp = output[t, :, :, 0:num_leadtimes]
        temp = np.moveaxis(temp, [0, 1, 2], [1, 2, 0])
        temp = gridpp.nearest(grid, points, temp)
        file.add_forecast(frt, temp)

        if probabilistic:
            Q = 2
            L = points.size()
            values = np.zeros([num_leadtimes, L, Q])
            temp = output[t, :, :, num_leadtimes : (2 * num_leadtimes)]
            temp = np.moveaxis(temp, [0, 1, 2], [1, 2, 0])
            values[:, :, 0] = gridpp.nearest(grid, points, temp)
            temp = output[t, :, :, (2 * num_leadtimes) : (3 * num_leadtimes)]
            temp = np.moveaxis(temp, [0, 1, 2], [1, 2, 0])
            values[:, :, 1] = gridpp.nearest(grid, points, temp)
            file.add_quantile_forecast(frt, values)

    """ Retrive observations from obs and add to Verif file """
    frost_client_id = "92ecb11b-2a37-4b60-8f5f-74d7ea360d2a"
    x, y = np.meshgrid(times, leadtimes)
    all_valid_times = np.sort(np.unique(x.flatten() + y.flatten()))
    # obs = yrlib.obs.Frost(frost_client_id, wmo=True)
    # obs_dataset = obs.get(all_valid_times, frost_variable)
    obs_dataset = yrlib.frost.get(
        np.min(all_valid_times),
        np.max(all_valid_times),
        frost_variable,
        frost_client_id,
        wmo=True,
        station_ids=valid_station_ids,
    )
    num_hours = (np.max(all_valid_times) - np.min(all_valid_times)) // 3600
    print("Retriving %d hours from frost" % num_hours)

    Iin, Iout = yrlib.util.get_common_indices(
        [loc.id for loc in obs_dataset.locations], station_ids
    )
    for t, curr_time in enumerate(obs_dataset.times):
        temp = obs_dataset.values[t, :]
        values = np.nan * np.zeros(points.size())
        yrlib.units.convert(temp, obs_dataset.units, units, True)
        values[Iout] = temp[Iin]
        file.add_observations(curr_time, values)
    file.write()


def verify_grid(
    filename,
    output,
    targets,
    times,
    leadtimes,
    grid,
    probabilistic,
    variable="air_temperature",
    sampling_factor=1,
):
    """Writes a verif file based on a gridded truth

    Args:
        filename (str): Filename to write verif data to
        output (np.array): 4D array of predictions (time, y, x, leadtime * quantiles)
        targets (np.array): 4D array of targets (time, y, x, leadtime)
        times (np.array): 1D array of forecast reference times [unixtime]
        leadtimes (np.array): 1D array of leadtimes [s]
        grid (gridpp.Grid): Grid description of the forecast field
        probabilistic (bool): If true, treat the output as having 3 components (10, 50, 90%),
            otherwise treat it as being deterministic only
        variable (str): Name of the forecast variable
        sampling_factor (int): Reduce each grid dimension by this factor
    """
    import yrlib

    if variable == "air_temperature":
        frost_variable = "air_temperature"
        units = "C"
    else:
        raise Exception("Undefined variable %s" % variable)

    num_leadtimes = len(leadtimes)

    kwargs = {"units": units, "variable_name": variable}
    if probabilistic:
        kwargs["quantiles"] = [0.1, 0.9]
    if sampling_factor == 1:
        points = grid.to_points()
        Iy = slice(grid.size()[0])
        Ix = slice(grid.size()[1])
    else:
        Iy = slice(0, grid.size()[0], sampling_factor)
        Ix = slice(0, grid.size()[1], sampling_factor)
        points = gridpp.Points(
            grid.get_lats()[Iy, Ix].flatten(),
            grid.get_lons()[Iy, Ix].flatten(),
            grid.get_elevs()[Iy, Ix].flatten(),
        )
    file = yrlib.output.VerifFile(filename, points, leadtimes // 3600, **kwargs)

    for t in tqdm.tqdm(range(len(times))):
        frt = times[t]
        temp = output[t, Iy, Ix, 0:num_leadtimes]  # y, x, leadtime
        temp = np.moveaxis(temp, [0, 1, 2], [1, 2, 0])  # leadtime, y, x
        temp = np.reshape(temp, [temp.shape[0], temp.shape[1] * temp.shape[2]])
        file.add_forecast(frt, temp)

        if probabilistic:
            Q = 2
            L = points.size()
            values = np.zeros([num_leadtimes, L, Q])
            temp = output[t, Iy, Ix, num_leadtimes : (2 * num_leadtimes)]
            temp = np.moveaxis(temp, [0, 1, 2], [1, 2, 0])  # leadtime, y, x
            temp = np.reshape(temp, [temp.shape[0], temp.shape[1] * temp.shape[2]])
            values[:, :, 0] = temp
            temp = output[t, Iy, Ix, (2 * num_leadtimes) : (3 * num_leadtimes)]
            temp = np.moveaxis(temp, [0, 1, 2], [1, 2, 0])  # leadtime, y, x
            temp = np.reshape(temp, [temp.shape[0], temp.shape[1] * temp.shape[2]])
            values[:, :, 1] = temp
            file.add_quantile_forecast(frt, values)

    x, y = np.meshgrid(times, leadtimes)
    valid_times = x + y
    all_valid_times = np.sort(np.unique(x.flatten() + y.flatten()))
    for t, curr_time in enumerate(all_valid_times):
        I = np.where(valid_times == curr_time)
        temp = targets[I[1][0], Iy, Ix, I[0][0]]
        values = temp.flatten()
        file.add_observations(curr_time, values)
    file.write()


def create_directory(filename):
    """Creates all sub directories necessary to be able to write filename"""
    dir = os.path.dirname(filename)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def get_memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss
    for child in p.children(recursive=True):
        mem += child.memory_info().rss
    return mem


def get_max_memory_usage():
    """In bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000


def print_memory_usage(message=None, show_line=False):
    """Prints the current and maximum memory useage of this process
    Args:
        message (str): Prepend with this message
        show_line (bool): Add the file and line number making this call at the end of message
    """

    output = "Memory usage (max): %.2f MB (%.2f MB)" % (
        get_memory_usage() / 1024 / 1024,
        get_max_memory_usage() / 1024 / 1024,
    )
    if message is not None:
        output = message + " " + output
    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)


def list_to_str(lst):
    lst_str = list()
    for i in lst:
        if i is None:
            lst_str += ["None"]
        elif isinstance(i, int):
            lst_str += ["%d" % i]
        else:
            lst_str += ["%g" % i]
    return ", ".join(lst_str)


def date_to_unixtime(date, hour=0, min=0, sec=0):
    """Convert YYYYMMDD(HHMMSS) to unixtime

    Arguments:
       date (int): YYYYMMDD
       hour (int): HH
        min (int): MM
        sec (int): SS

    Returns:
       int: unixtime [s]
    """
    if not isinstance(date, int):
        raise ValueError("Date must be an integer")
    if not isinstance(hour, numbers.Number):
        raise ValueError("Hour must be a number")
    if hour < 0 or hour >= 24:
        raise ValueError("Hour must be between 0 and 24")
    if min < 0 or hour >= 60:
        raise ValueError("Minute must be between 0 and 60")
    if sec < 0 or hour >= 60:
        raise ValueError("Second must be between 0 and 60")

    year = date // 10000
    month = date // 100 % 100
    day = date % 100
    ut = calendar.timegm(datetime.datetime(year, month, day).timetuple())
    return ut + (hour * 3600) + (min * 60) + sec


def unixtime_to_date(unixtime):
    """Convert unixtime to YYYYMMDD

    Arguments:
       unixtime (int): unixtime [s]

    Returns:
       int: date in YYYYMMDD
       int: hour in HH
    """
    if not isinstance(unixtime, numbers.Number):
        raise ValueError("unixtime must be a number")

    dt = datetime.datetime.utcfromtimestamp(int(unixtime))
    date = dt.year * 10000 + dt.month * 100 + dt.day
    hour = dt.hour
    return date, hour


def is_list(ar):
    try:
        len(ar)
        return True
    except Exception as e:
        return False


def get_common_indices(x, y):
    Ix = list()
    Iy = list()
    for i in range(len(x)):
        if x[i] in y:
            Ix += [i]
            if isinstance(y, list):
                Iy += [y.index(x[i])]
            else:
                Iy += [np.where(x[i] == y)[0][0]]
    return Ix, Iy
