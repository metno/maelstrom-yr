import os
import glob
import time
import netCDF4
import tqdm
import numpy as np
import tensorflow as tf
import yaml
import json
import multiprocessing
import gridpp
import maelstrom


def load(
    variable="air_temperature",
    size="5GB",
    num_days=None,
    subset_leadtimes=None,
    basedir="data/",
):
    s_time = time.time()
    datadir = "%s/%s/%s/" % (basedir, variable, size)
    filenames = glob.glob("%s/*.nc" % (datadir))
    filenames.sort()
    if num_days is not None:
        filenames = filenames[slice(num_days)]
    print("Loading files: %d" % len(filenames))

    if len(filenames) == 0:
        raise Exception("No data found in '%s'" % datadir)

    predictors = targets = times = leadtimes = None
    for t in tqdm.tqdm(range(len(filenames))):
        filename = filenames[t]
        with netCDF4.Dataset(filename, "r") as ifile:
            # Ix = range(0, 200)
            # Iy = range(0, 200)
            # Ix = range(128)
            # Iy = range(128)
            Ix = range(len(ifile.dimensions["x"]))
            Iy = range(len(ifile.dimensions["y"]))
            name_predictor = ifile.variables["name_predictor"][:].filled("")
            name_predictor = np.array(
                [
                    "".join([qq.decode("utf-8") for qq in name_predictor[p, :]])
                    for p in range(name_predictor.shape[0])
                ]
            )
            if predictors is None:
                leadtimes = ifile.variables["leadtime"][:].filled()
                predictor_leadtimes = ifile.variables["leadtime_predictor"][:].filled()
                Ip = range(len(predictor_leadtimes))
                Ilt = range(len(leadtimes))
                if subset_leadtimes is not None:
                    for lt in subset_leadtimes:
                        if lt not in leadtimes:
                            raise Exception(
                                "Cannot subset leadtime %s, because file does not contain it. Possible values are %s"
                                % (lt, leadtimes)
                            )
                    Ip = [
                        p
                        for p in range(len(predictor_leadtimes))
                        if predictor_leadtimes[p] in subset_leadtimes
                    ]
                    Ilt = [np.where(leadtimes == lt)[0][0] for lt in subset_leadtimes]
                    leadtimes = subset_leadtimes
                    print(Ip, Ilt)
                predictors = np.zeros(
                    [len(filenames)]
                    + list(ifile.variables["predictors"][Iy, Ix, Ip].shape),
                    np.float32,
                )
                if "target" in ifile.variables:
                    targets = np.zeros(
                        [len(filenames)]
                        + list(ifile.variables["target"][Iy, Ix, Ilt].shape),
                        np.float32,
                    )
                else:
                    targets = np.zeros(
                        [len(filenames)]
                        + list(ifile.variables["target_mean"][Iy, Ix, Ilt].shape)
                        + [2],
                        np.float32,
                    )
                times = np.zeros(len(filenames))
                lats, lons = maelstrom.ncparser.get_latlon(ifile)
                elevs = maelstrom.ncparser.get_altitude(ifile)
                lats = lats[Iy, :][:, Ix]
                lons = lons[Iy, :][:, Ix]
                elevs = elevs[Iy, :][:, Ix]
                grid = gridpp.Grid(lats, lons, elevs)
            # print(ifile.variables["predictors"].shape)
            predictors[t, ...] = ifile.variables["predictors"][Iy, Ix, Ip]
            # print("%.1f s: Done loading predictors" % (time.time() - s_time))
            if "target" in ifile.variables:
                targets[t, ...] = ifile.variables["target"][Iy, Ix, Ilt]
            else:
                targets[t, ..., 0] = ifile.variables["target_mean"][Iy, Ix, Ilt]
                targets[t, ..., 1] = ifile.variables["target_std"][Iy, Ix, Ilt]
            # print("%.1f s: Done loading targets" % (time.time() - s_time))
            times[t] = float(ifile.variables["time"][0])
    Itime = np.where(
        [
            (np.max(targets[i, ...]) < 100000) and (np.max(predictors[i, ...]) < 100000)
            for i in range(targets.shape[0])
        ]
    )[0]
    return (
        predictors[Itime, ...],
        targets[Itime, ...],
        times[Itime],
        leadtimes,
        grid,
        name_predictor,
    )


def load_patch_dataset(
    variable="air_temperature",
    size="5GB",
    num_days=None,
    subset_leadtimes=None,
    basedir="data/",
    num_cores=4,
):
    s_time = time.time()
    datadir = "%s/%s/%s/" % (basedir, variable, size)
    filenames = glob.glob("%s/*.nc" % (datadir))
    filenames.sort()
    if num_days is not None:
        filenames = filenames[slice(num_days)]
    print("Loading files: %d" % len(filenames))

    if len(filenames) == 0:
        raise Exception("No data found in '%s'" % datadir)

    pool = multiprocessing.Pool(num_cores)
    F = len(filenames)
    s_time = time.time()
    results = pool.starmap(load_from_file, [(f, subset_leadtimes) for f in filenames])
    S, Y, X, P = results[0][0].shape
    LT = results[0][1].shape[3]
    T = results[0][1].shape[4]
    predictors = np.zeros([F * S, Y, X, P], np.float32)
    targets = np.zeros([F * S, Y, X, LT, T], np.float32)
    times = np.zeros([F * S], np.float32)
    for i in range(len(results)):
        predictors[slice(i * S, (i + 1) * S), ...] = results[i][0]
        targets[slice(i * S, (i + 1) * S), ...] = results[i][1]
        times[slice(i * S, (i + 1) * S)] = results[i][2]
    name_predictor = results[0][4]
    leadtimes = results[0][3]
    xx, yy = np.meshgrid(np.arange(Y), np.arange(X))
    grid = gridpp.Grid(xx, yy)
    return predictors, targets, times, leadtimes, grid, name_predictor


class PatchDataset(tf.data.Dataset):
    def __init__(self, filenames):
        self.filenames = fileanames

    def _generator(filenames):
        for i in range(len(filenames)):
            yield np.zeros([3, 2], np.float32), np.zeros([3, 2], np.float32)

    def __new__(cls, filenames):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(3, 2), dtype=tf.int64),
                tf.TensorSpec(shape=(3, 2), dtype=tf.int64),
            ),
            args=(filenames,),
        )


def load_from_file(filename, subset_leadtimes):
    print(filename)
    s_time = time.time()
    with netCDF4.Dataset(filename, "r") as ifile:
        name_predictor = ifile.variables["name_predictor"][:].filled("")
        name_predictor = np.array(
            [
                "".join([qq.decode("utf-8") for qq in name_predictor[p, :]])
                for p in range(name_predictor.shape[0])
            ]
        )
        S = len(ifile.dimensions["sample"])
        leadtimes = ifile.variables["leadtime"][:].filled()
        predictor_leadtimes = ifile.variables["leadtime_predictor"][:].filled()
        Ip = range(len(predictor_leadtimes))
        Ilt = range(len(leadtimes))
        if subset_leadtimes is not None:
            for lt in subset_leadtimes:
                if lt not in leadtimes:
                    raise Exception(
                        "Cannot subset leadtime %s, because file does not contain it. Possible values are %s"
                        % (lt, leadtimes)
                    )
            Ip = [
                p
                for p in range(len(predictor_leadtimes))
                if predictor_leadtimes[p] in subset_leadtimes
            ]
            Ilt = [np.where(leadtimes == lt)[0][0] for lt in subset_leadtimes]
            leadtimes = subset_leadtimes
        name_predictor = name_predictor[Ip]
        predictors = np.zeros(
            [S] + list(ifile.variables["predictors"][0, :, :, Ip].shape), np.float32
        )
        if "target" in ifile.variables:
            targets = np.zeros(
                [S] + list(ifile.variables["target"][0, :, :, Ilt].shape), np.float32
            )
        else:
            targets = np.zeros(
                [S] + list(ifile.variables["target_mean"][0, :, :, Ilt].shape) + [2],
                np.float32,
            )
        times = np.zeros(S)

        # Faster to load all first
        q = ifile.variables["predictors"][:]
        predictors[:] = q[:, :, :, Ip]

        if "target" in ifile.variables:
            targets[:] = ifile.variables["target"][:, :, :, Ilt]
        else:
            targets[..., 0] = ifile.variables["target_mean"][:, :, :, Ilt]
            targets[..., 1] = ifile.variables["target_std"][:, :, :, Ilt]
        times = float(ifile.variables["time"][0])
        return predictors, targets, times, leadtimes, name_predictor


class Dataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        predictors,
        targets,
        batch_size,
        patch_size,
        patches_per_file,
        predictor_names,
        start=0,
        end=1,
        num_targets=1,
    ):
        self.num_y = predictors.shape[1]
        self.num_x = predictors.shape[2]
        self.num_predictors = predictors.shape[-1]
        self.predictors = predictors
        self.targets = targets
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patches_per_file = patches_per_file
        self.num_files = predictors.shape[0]
        self.predictor_names = None
        self.num_leadtimes = targets.shape[3] // num_targets
        if predictor_names is not None:
            self.predictor_names = predictor_names

        if self.patches_per_file is None:
            self.patches_per_file = 1
        if self.num_x < patch_size:
            raise Exception(
                f"Patch size ({patch_size}) must be less than x ({self.num_x})"
            )
        if self.num_y < patch_size:
            raise Exception(
                f"Patch size ({patch_size}) must be less than y ({self.num_y})"
            )

        num_missing_predictor_values = np.sum(np.isnan(self.predictors))
        num_missing_target_values = np.sum(np.isnan(self.targets))
        if num_missing_predictor_values > 0:
            raise Exception(
                f"Num missing predictor values: {num_missing_predictor_values}"
            )
        if num_missing_target_values > 0:
            raise Exception(f"Num missing target values: {num_missing_target_values}")

        self.num_samples = self.num_files * self.patches_per_file

        self.indices = get_indices(
            self.num_files,
            self.patches_per_file,
            self.batch_size,
            self.num_x,
            self.num_y,
            self.patch_size,
        )
        length = len(self.indices)

        """
        # Shuffle indices randomly, such that validation gets slices from all files
        # Note, not needed after get_inidices ensures each batch gets all files
        np.random.seed(1000)
        I = np.argsort(np.random.rand(length))
        self.indices = [self.indices[i] for i in I]
        """
        # Extract training or validation subset
        I_start = int(length * start)
        I_end = int(length * end)
        self.indices = self.indices[slice(I_start, I_end)]
        # print(start, end, self.indices)

    def description(self):
        d = dict()

        d["Predictor shape"] = ", ".join(["%d" % i for i in self.predictors.shape])
        d["Target shape"] = ", ".join(["%d" % i for i in self.targets.shape])
        d["Num files"] = self.num_files
        d["Patches per file"] = self.patches_per_file
        d["Batch size"] = self.batch_size
        d["Num samples"] = self.num_samples
        d["Patch size"] = self.patch_size
        d["Num leadtimes"] = self.num_leadtimes
        d["Num predictors"] = self.num_predictors
        d["Predictors"] = list()
        total_size = np.product(self.predictors.shape) + np.product(self.targets.shape)
        d["Total size (GB)"] = total_size * 4 / 1024 ** 3
        if self.predictor_names is not None:
            for q in self.predictor_names:
                d["Predictors"] += [str(q)]
        return d

    def __str__(self):
        return json.dumps(self.description(), indent=4)

    def get_subset(self, start, end):
        """Creates a new dataset object, with a subset of the files. No new memory is created."""
        assert start >= 0 and start <= 1
        assert end >= 0 and end <= 1
        num_files = self.predictors.shape[0]
        If = slice(int(start * num_files), int(end * num_files))
        data = Dataset(
            self.predictors,
            self.targets,
            self.batch_size,
            self.patch_size,
            self.patches_per_file,
            self.predictor_names,
            start,
            end,
        )
        return data

    def __getitem__(self, index):
        debug = False
        I, Iy, Ix = self.indices[index]
        pred = self.predictors[:, Iy, Ix, :][I, ...]
        targ = self.targets[:, Iy, Ix, :][I, ...]
        if debug:
            print(f"Getting item {index}: {pred.shape} {targ.shape} {I}")
        return pred, targ

    def __len__(self):
        # int(np.ceil(self.num_files * self.patches_per_file / self.batch_size * (self.end - self.start)))
        return len(self.indices)


# def get_synthetic_data(batch_size, patch_size, num_leadtimes=60, num_x=1796, num_y=2321, num_cases=1400):
def get_synthetic_data(
    batch_size, patch_size, num_predictors, num_leadtimes, num_samples, num_files
):
    predictors = np.random.rand(
        num_files, patch_size, patch_size, num_predictors * num_leadtimes
    ).astype(np.float32)
    targets = np.random.rand(num_files, patch_size, patch_size, num_leadtimes).astype(
        np.float32
    )
    data = Dataset(
        predictors, targets, batch_size, patch_size, num_samples // num_files, None
    )
    return data


def get_real_data(
    filenames,
    batch_size,
    patch_size,
    patches_per_file=None,
    limit_leadtimes=None,
    extra_predictors=[],
    groups=False,
    limit_predictors=None,
    normalization=None,
    include_target_std=False,
):
    s_time = time.time()

    predictors = targets = None

    # Load all data
    for t in tqdm.tqdm(range(len(filenames)), desc="Loading data", leave=False):
        with netCDF4.Dataset(filenames[t], "r") as file:
            # Get the leadtime (in hours) corresponding to each predictor
            all_predictor_leadtimes = (
                file.variables["leadtime_predictor"][:]
                .astype(np.float32)
                .filled(np.nan)
            )
            I = np.where(~np.isnan(all_predictor_leadtimes))
            all_predictor_leadtimes[I] = all_predictor_leadtimes[I] // 3600

            # Get the target leadtimes (in hours)
            target_leadtimes = file.variables["leadtime"][:] // 3600

            Ipredictors = range(len(all_predictor_leadtimes))
            Itarget = range(len(target_leadtimes))

            # Limit the predictors to the chosen leadtimes
            if limit_leadtimes is not None:
                Ipredictors = [
                    i
                    for i in range(len(all_predictor_leadtimes))
                    if all_predictor_leadtimes[i] in limit_leadtimes
                    or np.isnan(all_predictor_leadtimes[i])
                ]
                Itarget = [
                    i
                    for i in range(len(target_leadtimes))
                    if target_leadtimes[i] in limit_leadtimes
                ]
                target_leadtimes = [i for i in target_leadtimes if i in limit_leadtimes]
            assert len(Itarget) > 0
            assert len(Ipredictors) > 0

            # Limit the predictors to the chosen predictor names
            all_predictor_names = file.variables["name_predictor"][:].filled("")
            all_predictor_names = np.array(
                [
                    "".join([qq.decode("utf-8") for qq in all_predictor_names[p, :]])
                    for p in range(all_predictor_names.shape[0])
                ]
            )
            if limit_predictors is not None:
                Icurr = [
                    i
                    for i in range(len(all_predictor_names))
                    if all_predictor_names[i] in limit_predictors
                ]
                Ipredictors = [I for I in Icurr if I in Ipredictors]
            assert len(Ipredictors) > 0
            predictor_names = all_predictor_names[Ipredictors]
            predictor_leadtimes = all_predictor_leadtimes[Ipredictors]

            # print(time.time() - s_time)
            curr_predictors = file.variables["predictors"][:][:, :, Ipredictors]
            # print(time.time() - s_time)
            if include_target_std:
                c0 = file.variables["target_mean"][:][:, :, Itarget]
                c1 = file.variables["target_std"][:][:, :, Itarget]
                # c0 = np.expand_dims(c0, axis=3)
                # c1 = np.expand_dims(c1, axis=3)
                curr_targets = np.concatenate((c0, c1), axis=2)
            else:
                curr_targets = file.variables["target_mean"][:][:, :, Itarget]
            # print(time.time() - s_time)
            if predictors is None:
                predictors = np.nan * np.zeros(
                    [len(filenames)] + list(curr_predictors.shape), np.float32
                )
            if targets is None:
                targets = np.nan * np.zeros(
                    [len(filenames)] + list(curr_targets.shape), np.float32
                )
            predictors[t, ...] = curr_predictors
            targets[t, ...] = curr_targets
            # print(time.time() - s_time)

    num_predictors = predictors.shape[3]

    num_leadtimes = targets.shape[3]
    if include_target_std:
        num_leadtimes = num_leadtimes // 2
    assert num_leadtimes == len(target_leadtimes)
    print("Done loading data:", time.time() - s_time)

    # Add extra static fields
    if extra_predictors is not None:
        with netCDF4.Dataset(filenames[0], "r") as file:
            extra = np.zeros(
                [
                    predictors.shape[0],
                    predictors.shape[1],
                    predictors.shape[2],
                    len(extra_predictors),
                ],
                np.float32,
            )
            for i in tqdm.tqdm(
                range(len(extra_predictors)),
                desc="Adding extra predictors",
                leave=False,
            ):
                extra_predictor = extra_predictors[i]
                if extra_predictor == "altitude":
                    values = file.variables["altitude"][:]
                elif extra_predictor == "land_area_fraction":
                    values = file.variables["land_area_fraction"][:]
                elif extra_predictor == "x":
                    x, y = np.meshgrid(file.variables["x"][:], file.variables["y"][:])
                    values = x
                elif extra_predictor == "y":
                    x, y = np.meshgrid(file.variables["x"][:], file.variables["y"][:])
                    values = y
                else:
                    raise NotImplementedError()
                for p in range(predictors.shape[0]):
                    extra[p, :, :, i] = values
                predictor_leadtimes = np.append(predictor_leadtimes, np.nan)
                predictor_names = np.append(predictor_names, extra_predictor)
            predictors = np.concatenate((predictors, extra), axis=3)

    print("Done adding extra static fields: ", time.time() - s_time)
    num_predictors = predictors.shape[3]

    if groups:
        unique_predictor_leadtime = np.unique(
            predictor_leadtimes[~np.isnan(predictor_leadtimes)]
        )
        # print(unique_predictor_leadtime)
        assert num_leadtimes == len(unique_predictor_leadtime)

        # Duplicate all predictors with nan leadtimes
        Ip_static = np.where(np.isnan(predictor_leadtimes))[0]
        if len(Ip_static) > 0:
            Ip_non_static = np.where(~np.isnan(predictor_leadtimes))[0]
            num_copies = num_leadtimes - 1
            extra = np.zeros(
                [
                    predictors.shape[0],
                    predictors.shape[1],
                    predictors.shape[2],
                    len(Ip_static) * num_leadtimes,
                ],
                np.float32,
            )
            extra_predictor_leadtimes = list()
            extra_predictor_names = list()
            for i in range(len(Ip_static)):
                for j in range(num_leadtimes):
                    extra[:, :, :, i * num_leadtimes + j] = predictors[
                        :, :, :, Ip_static[i]
                    ]
                    extra_predictor_leadtimes = np.append(
                        extra_predictor_leadtimes, target_leadtimes[j]
                    )
                    extra_predictor_names = np.append(
                        extra_predictor_names, predictor_names[Ip_static[i]]
                    )
            predictors = predictors[:, :, :, Ip_non_static]
            predictors = np.concatenate((predictors, extra), axis=3)
            predictor_leadtimes = predictor_leadtimes[Ip_non_static]
            predictor_leadtimes = np.concatenate(
                (predictor_leadtimes, extra_predictor_leadtimes)
            )
            predictor_names = predictor_names[Ip_non_static]
            predictor_names = np.concatenate((predictor_names, extra_predictor_names))

        # Reshuffle so that we have predictors ordered by leadtime
        num_samples = predictors.shape[0]
        num_predictors = predictors.shape[3]
        num_predictors_per_leadtime = num_predictors // num_leadtimes
        predictor_leadtimes_copy = np.copy(predictor_leadtimes)
        predictor_names_copy = np.copy(predictor_names)
        assert predictors.shape[-1] == len(predictor_leadtimes)
        assert num_leadtimes == len(target_leadtimes)
        for i in tqdm.tqdm(range(num_samples), desc="Grouping predictors", leave=False):
            temp_predictors = np.zeros(predictors[i, ...].shape, np.float32)
            for j in range(num_leadtimes):
                Iout = slice(
                    j * num_predictors_per_leadtime,
                    (j + 1) * num_predictors_per_leadtime,
                )
                Iin = slice(j, num_predictors, num_leadtimes)
                temp_predictors[..., Iout] = predictors[i, ..., Iin]
                if i == 0:
                    predictor_leadtimes_copy[Iout] = predictor_leadtimes[Iin]
                    predictor_names_copy[Iout] = predictor_names[Iin]
            predictors[i, ...] = temp_predictors

        predictor_leadtimes = predictor_leadtimes_copy
        predictor_names = predictor_names_copy

        # Shuffling of targets is only needed if target has uncertainty
        num_targets = targets.shape[3]
        num_targets_per_leadtime = num_targets // num_leadtimes
        if num_targets_per_leadtime > 1:
            for i in range(num_samples):
                temp_targets = np.zeros(targets[i, ...].shape, np.float32)
                for j in range(num_leadtimes):
                    Iout = slice(
                        j * num_targets_per_leadtime, (j + 1) * num_targets_per_leadtime
                    )
                    Iin = slice(j, num_targets, num_leadtimes)
                    temp_targets[..., Iout] = targets[i, ..., Iin]
                targets[i, ...] = temp_targets
        num_predictors = predictors.shape[3]
        assert num_predictors == num_leadtimes * len(np.unique(predictor_names))

        # Check that grouping was done correctly
        for i, leadtime in enumerate(target_leadtimes):
            I = slice(
                i * num_predictors // num_leadtimes,
                (i + 1) * num_predictors // num_leadtimes,
            )
            assert (predictor_leadtimes[I] == leadtime).all()
        print("Done grouping:", time.time() - s_time)

    # print("HERE", np.nanmean(np.abs(predictors[:, :, :, 0] - targets[:, :, :, 0])))
    print("Raw MAE corr")
    mae = 0
    corr = 0
    for i, leadtime in enumerate(target_leadtimes):
        Ip = np.where(
            (predictor_names == "air_temperature_2m")
            & (predictor_leadtimes == leadtime)
        )[0][0]
        a = predictors[:, :, :, Ip].flatten()
        b = targets[:, :, :, i].flatten()
        curr_mae = np.mean(np.abs(a - b))
        curr_corr = np.corrcoef(a, b)[0, 1]
        mae += curr_mae
        corr += curr_corr
        print(f"{leadtime}, {curr_mae:.3f}, {curr_corr:.3f}")
    print(f"All, {mae / len(target_leadtimes):.3f}, {corr / len(target_leadtimes):.3f}")

    # Normalize
    num_predictors = predictors.shape[3]
    if normalization is None:
        for p in tqdm.tqdm(
            range(num_predictors), desc="Normalizing predictors", leave=False
        ):
            m = np.nanmean(predictors[:, :, :, p])
            s = np.nanstd(predictors[:, :, :, p])
            # print(p, predictor_names[p], m, s)
            predictors[:, :, :, p] -= m
            predictors[:, :, :, p] /= s
            # predictors[:, :, :, p] -= np.nanmin(predictors[:, :, :, p])
            # predictors[:, :, :, p] /= np.nanmax(predictors[:, :, :, p])
    else:
        with open(normalization) as file:
            coefficients = yaml.load(file, Loader=yaml.SafeLoader)
        for p in tqdm.tqdm(
            range(num_predictors), desc="Normalizing predictors", leave=False
        ):
            name = predictor_names[p]
            if name not in coefficients:
                print(list(coefficients.keys()))
                raise ValueError(f"Could not find normalization information for {name}")
            m = coefficients[name][0]
            s = coefficients[name][0]
            predictors[:, :, :, p] -= m
            predictors[:, :, :, p] /= s
        # targets -= coefficients["air_temperature_2m"]
        # targets /= coefficients["air_temperature_2m"]

    print("Done normalizing:", time.time() - s_time)

    num_targets = 1 + include_target_std
    data = Dataset(
        predictors,
        targets,
        batch_size,
        patch_size,
        patches_per_file,
        np.sort(np.unique(predictor_names)),
        num_targets=num_targets,
    )
    return data


def get_indices(num_files, patches_per_file, batch_size, num_x, num_y, patch_size):
    num_cases = num_files * patches_per_file
    length = int(np.ceil(num_files * patches_per_file / batch_size))
    indices = list()
    f = np.tile(range(num_files), patches_per_file)
    for i in range(length):
        I_start = i * batch_size
        I_end = min(num_cases, (i + 1) * batch_size)
        ind = range(I_start, I_end)
        # Select files
        I = [int(i // patches_per_file) for i in ind]
        I = [f[i] for i in ind]

        if num_x == patch_size:
            x = 0
        else:
            x = np.random.randint(0, num_x - patch_size)
        if num_y == patch_size:
            y = 0
        else:
            y = np.random.randint(0, num_y - patch_size)
        Ix = slice(x, x + patch_size)
        Iy = slice(y, y + patch_size)
        indices += [(I, Iy, Ix)]
    return indices


def load_patch_dataset2(
    variable="air_temperature",
    size="5GB",
    num_days=None,
    subset_leadtimes=None,
    basedir="data/",
    num_cores=4,
):
    def parse_file(filename):
        print(filename)
        # filename = filename.numpy()
        # print(filename)
        return np.zeros([3, 2, 1]), np.zeros([3, 2, 1])
        return load_from_file(filename, [])

    def prepare_for_training(ds, cache=True, augment=True, shuffle_buffer_size=1000):
        # If a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets
        # that don't fit in memory.
        if cache:
            # store in local storage
            if isinstance(cache, str):
                ds = ds.cache(cache)
            # store in memory
            else:
                ds = ds.cache()

        # augment after cache, and should only be applied to a training dataset
        if augment:
            ds = ds.map(
                lambda image, label: (tf.image.random_flip_left_right(image), label)
            ).map(lambda image, label: (tf.image.random_flip_up_down(image), label))

        batch_size = 10
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    datadir = "%s/%s/%s/" % (basedir, variable, size)
    print(datadir + "*.nc")
    filenames = glob.glob("%s/*.nc" % (datadir))
    filenames.sort()
    if num_days is not None:
        filenames = filenames[slice(num_days)]

    print("Loading files: %d" % len(filenames))
    # cache = True, False, './file_name'
    # If the dataset doesn't fit in memory use a cache file,
    # eg. cache='./train.tfcache'
    # Take note that your training cache should not be mixed with your validation cache
    cache = "data/" + ".tfcache"
    list_ds = tf.data.Dataset.list_files([datadir + "*.nc"])
    print(list_ds)
    for q in list_ds:
        print(q)
    # print(filenames)
    # list_ds = tf.data.Dataset.from_tensor_slices(filenames)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Set `num_parallel_calls` so that multiple images are
    # processed in parallel
    labeled_ds = list_ds.map(parse_file, num_parallel_calls=AUTOTUNE)
    data_generator = prepare_for_training(labeled_ds, cache=cache)

    return data_generator
