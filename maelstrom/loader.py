import collections
import copy
import glob
import json
import multiprocessing
import numbers
import os
import re
import sys
import time

import gridpp
import netCDF4
import numpy as np
import tensorflow as tf
import tqdm
import yaml

import maelstrom

# import horovod.tensorflow as hvd



def get(args, sample=None):
    """Initialize loader with configuration"""
    name = args["type"]
    args = {k: v for k, v in args.items() if k not in ["type"]}
    range_variables = ["x_range", "y_range", "limit_leadtimes"]

    # Process value arguments
    for range_variable in range_variables:
        if range_variable in args:
            curr = args[range_variable]
            if isinstance(curr, str):
                if curr.find(":") == -1:
                    raise ValueError(
                        f"Cannot interpret range string {curr}. Should be in the form start:end"
                    )
                start, end = curr.split(":")
                args[range_variable] = range(int(start), int(end))
    if "filenames" in args:
        if isinstance(args["filenames"], dict):
            if sample not in args["filenames"]:
                raise ValueError(f"Sample type {sample} not defined")
            filenames_str = args["filenames"][sample]
        else:
            filenames_str = args["filenames"]

        if not isinstance(filenames_str, list):
            filenames_str = [filenames_str]
        filenames = list()
        for f in filenames_str:
            filenames += glob.glob(f)
        filenames = list(set(filenames))
        filenames.sort()

        # Remove filenames matching skip_filenames patterns
        skip_filenames_str = list()
        if "skip_filenames" in args:
            if isinstance(args["skip_filenames"], dict):
                if sample not in args["skip_filenames"]:
                    raise ValueError(f"Sample type {sample} not defined")
                skip_filenames_str = args["skip_filenames"][sample]
            else:
                skip_filenames_str = args["skip_filenames"]
            del args["skip_filenames"]  # Don't pass this argument to the initializer

            if not isinstance(skip_filenames_str, list):
                skip_filenames_str = [skip_filenames_str]
            skip_filenames = list()
            for f in skip_filenames_str:
                skip_filenames += glob.glob(f)

            filenames = [f for f in filenames if f not in skip_filenames]

        filenames.sort()
        args["filenames"] = filenames

    if name == "file":
        loader = FileLoader(**args)
    elif name == "fake":
        loader = FakeLoader(**args)
    else:
        raise NotImplementedError(f"Undefined loader {name}")

    if "compute_normalization" in args and args["compute_normalization"]:
        loader.compute_normalization_coefficients()
    return loader


class DataLoader:
    """Class for loading data from a remote store

    This class handles caching, parallel loading, and stores timing statistics

    Data is assumed to be of the following form:
        predictors: (time, leadtime, y, x, predictor)
        targets: (time, leadtime, y, x, target)

    Example usage:
        loader = DataLoader(...)
        dataset = loader.get_dataset()
        model = tf.model.Model(...)
        model.fit(dataset, ...)
    """

    def __init__(
        self,
        times,
        leadtimes,
        predictor_names,
        num_targets,
        num_files,
        num_samples_per_file,
        grid,
        cache_size=None,
        batch_size=1,
        prefetch=1,
        num_parallel_calls=1,
        predict_diff=False,
        debug=False,
        patch_size=None,
        num_random_patches=None,
    ):
        """Initialize loader

        Args:
            times (list): Array of forecast reference times (first dimension of the dataset)
            leadtimes (list): Array of forecast leadtimes (second dimension)
            predictor_names (list): List of names (fifth dimension)
            num_targets (int): Number of target fields (typically 1)
            num_files (int): Number of files in the dataset
            num_samples_per_file (int): Number of samples in each file
            cache_size (int): Number of files to store in memory. If None, then unlimited.
            batch_size (int): Number of indices to concatenate into each batch. Batching is done
                here, and not in the model.fit call.
            prefetch (int): Number of files to prefetch
            num_parallel_calls (int): Number of dataset indices to be read in parallel
            grid (gridpp.Grid): Object describing the grid (third and fourth dimension of the dataset)
            predict_diff (bool): Convert target into the difference between the raw forecast and the
                target
            debug (bool): Turn on debug information
            patch_size (int): Rearange data into square patches with this width. Note that in this
                case, grid is ignored and replaced by a fake patched one. Note that any remaining
                gridpoints that are not a part of a while patch on the right and top edges of the domain will be ignored.
            num_random_patches (int): NUmber of random patches to pick from full file. If None, use all patches.

        """
        if predictor_names is not None and not isinstance(predictor_names, list):
            raise ValueError("predictor_names must be a list of strings")

        if batch_size != 1:
            raise ValueError(
                "batch size other than 1 is not currently tested well enough..."
            )

        self.num_files = num_files
        self.times = times
        self.leadtimes = leadtimes
        self.predictor_names = predictor_names
        self.predict_diff = predict_diff
        self.num_targets = num_targets
        self.num_samples_per_file = num_samples_per_file
        self.patch_size = patch_size
        self.num_random_patches = num_random_patches
        if patch_size is not None:
            full_num_y, full_num_x = grid.size()

            if patch_size > full_num_x:
                raise ValueError(
                    f"Patch size ({patch_size}) must be <= size of domain ({full_num_y},{full_num_x})"
                )

            if self.num_random_patches is not None:
                self.num_x_patches_per_file = self.num_random_patches
                self.num_y_patches_per_file = 1
            else:
                # Ignore points on the edges
                self.num_x_patches_per_file = int(full_num_x // patch_size)
                self.num_y_patches_per_file = int(full_num_y // patch_size)

            # Create a fake grid
            x, y = np.meshgrid(
                np.linspace(0, 1, patch_size), np.linspace(0, 1, patch_size)
            )
            self.grid = gridpp.Grid(y, x)
        else:
            if self.num_random_patches is not None:
                raise ValueError(
                    "patch_size must be specified if num_random_patches is specified"
                )

            self.grid = grid
            self.num_x_patches_per_file = 1
            self.num_y_patches_per_file = 1

        self.target_shape = [
            self.num_leadtimes,
            self.num_y,
            self.num_x,
            self.num_targets,
        ]
        self.predictor_shape = [
            self.num_leadtimes,
            self.num_y,
            self.num_x,
            self.num_predictors,
        ]

        self.cache_size = cache_size
        self.cache = dict()
        self.debug = debug

        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.prefetch = prefetch

        self.sample_indices = np.argsort(np.random.rand(self.num_samples))
        self.sample_indices = range(self.num_samples)

        self.count_reads = 0
        self.timing = collections.defaultdict(lambda: 0)

    def load_data(self, index):
        """Loads data from archive
        Args:
            index (int): Index from archive to load from

        Returns:
            predictors (np.array): Array of predictors (sample, leadtime, y, x, predictor)
            targets (np.array): Array of targets (sample, leadtime, y, x)
        """
        raise NotImplementedError()

    @property
    def num_leadtimes(self):
        return len(self.leadtimes)

    @property
    def num_predictors(self):
        return len(self.predictor_names)

    def write_debug(self, message):
        if self.debug:
            print(message)

    @property
    def num_samples(self):
        return self.num_files * self.num_samples_per_file

    @property
    def num_patches(self):
        return self.num_files * self.num_samples_per_file * self.num_patches_per_sample

    @property
    def num_patches_per_sample(self):
        return self.num_x_patches_per_file * self.num_y_patches_per_file

    @property
    def num_patches_per_file(self):
        return self.num_patches_per_sample * self.num_samples_per_file

    @property
    def num_x(self):
        """Number of x coordinates in grid"""
        return self.grid.size()[1]

    @property
    def num_y(self):
        """Number of y coordinates in grid"""
        return self.grid.size()[0]

    def get_time_from_batch(self, batch):
        """Which forecast reference time does this batch represent?"""
        raise NotImplementedError()

        batches_per_file = (
            self.num_patches_per_sample * self.num_samples_per_file // self.batch_size
        )
        file_index = batch // batches_per_file
        return self.times[batch]

    def __call__(self):
        """
        A call to the loader returns a generator that provides the indicies (integers) of the
        dataset. Normally, a generator would be returned that actually produces the data, but this
        can then not be parallelized in tensorflow. Instead, the data loading is done by a map
        operation
        """
        for i in range(self.__len__()):
            yield tf.convert_to_tensor([i])

    def __getitem__(self, idx):
        """Retrive data for index idx

        Args:
            idx (int): Index to retrieve data for. Must be between 0 and __len__()

        Returns:
            p (tf.tensor): Tensor with predictor data (sample, leadtime, y, x, predictor)
            t (tf.tensor): Tensor with target data (sample, leadtime, y, x, target)
        """

        # print(idx, self.sample_indices)
        s_time = time.time()
        ids = self.sample_indices[idx]
        f = ids // self.num_samples_per_file
        # print(f"#2 idx={idx}, ids={ids}, f={f}, in={f in self.cache}")
        # if f in self.cache:
        #     raise Exception()
        # print(f, idx, self.num_samples_per_file, self.num_files)
        s = idx % self.num_samples_per_file
        assert f < self.num_files
        # print(f"Getting {f} {s}")

        if self.cache_size is not None and self.cache_size > 0 and f in self.cache:
            # self.write_debug("Cache hit")
            p, t = self.cache[f]
        else:
            # self.write_debug("Cache miss")
            ss_time = time.time()
            predictors, targets = self.load_data(f)
            self.count_reads += 1
            self.timing["reading"] += time.time() - ss_time

            # Perform all processing steps here
            predictors, targets = self.process(predictors, targets)

            if self.cache_size is not None and len(self.cache) >= self.cache_size:
                # self.write_debug("Clearing cache")
                self.cache.clear()

            ss_time = time.time()
            # p = tf.convert_to_tensor(predictors[[s], ...])
            # t = tf.convert_to_tensor(targets[[s], ...])
            p = tf.convert_to_tensor(predictors)
            t = tf.convert_to_tensor(targets)
            self.timing["convert"] += time.time() - ss_time
            self.cache[f] = p, t
            self.write_debug(f"Loaded and processed {idx} {time.time() - s_time}")
            self.timing["total"] += time.time() - s_time

        return p, t

    def __len__(self):
        """Returns the number of total samples in the dataset"""
        return (
            self.num_files
        )  # self.num_patches # self.num_files * self.num_samples_per_file

    def patch(self, predictors, targets):
        """Reorganize predictors and targets into patches"""
        if self.patch_size is not None:
            s_time = time.time()
            S, L, Y, X, P = predictors.shape
            assert self.num_patches_per_sample > 0
            new_predictors = np.zeros(
                [
                    S * self.num_patches_per_sample,
                    L,
                    self.patch_size,
                    self.patch_size,
                    P,
                ],
                np.float32,
            )
            new_targets = np.zeros(
                [
                    S * self.num_patches_per_sample,
                    L,
                    self.patch_size,
                    self.patch_size,
                    self.num_targets,
                ],
                np.float32,
            )
            count = 0
            for s in range(S):
                if self.num_random_patches is None:
                    for x in range(self.num_x_patches_per_file):
                        Ix = slice(x * self.patch_size, (x + 1) * self.patch_size)
                        for y in range(self.num_y_patches_per_file):
                            Iy = slice(y * self.patch_size, (y + 1) * self.patch_size)
                            new_predictors[count, ...] = predictors[s, :, Iy, Ix, :]
                            new_targets[count, ...] = targets[s, :, Iy, Ix, :]
                            count += 1
                else:
                    full_num_y = predictors.shape[2]
                    full_num_x = predictors.shape[3]
                    for x in range(self.num_random_patches):
                        xi = np.random.randint(0, full_num_x - self.patch_size)
                        yi = np.random.randint(0, full_num_y - self.patch_size)
                        Ix = slice(xi, xi + self.patch_size)
                        Iy = slice(yi, yi + self.patch_size)
                        new_predictors[count, ...] = predictors[s, :, Iy, Ix, :]
                        new_targets[count, ...] = targets[s, :, Iy, Ix, :]
                        count += 1

            e_time = time.time()
            self.timing["patching"] += e_time - s_time

            return new_predictors, new_targets
        else:
            return predictors, targets

    def process(self, predictors, targets):
        """Any processing step that will be performed on data after it has been loaded

        This can be for example, normalizing data, feature extraction, augmentation, etc.  Subclases
        can override this
        """
        # print("#1", predictors.shape)
        # print("Before")
        # for i in range(predictors.shape[-1]):
        #     print(np.nanmean(predictors[..., i]), np.nanstd(predictors[..., i]))
        predictors, targets = self._process(predictors, targets)
        # print("#2", predictors.shape)
        predictors, targets = self.patch(predictors, targets)
        # print("After")
        # for i in range(predictors.shape[-1]):
        #     print(np.nanmean(predictors[..., i]), np.nanstd(predictors[..., i]))
        # sys.exit()

        return predictors, targets

    def _process(self, predictors, targets):
        return predictors, targets

    def get_dataset(self, randomize_order=False):
        """Returns a tensorflow dataset that reads data in parallel

        Args:
            randomize_order (bool): Randomize which files are read

        Returns:
            tf.data.Dataset: Tensorflow dataset
        """

        # The generator just returns an index, and then the data loading actually happens in map
        # TODO: Randomize on each call to the dataset to avoid the training repeating itself
        if randomize_order:
            z = np.argsort(np.random.rand(self.num_files)).tolist()
        else:
            z = list(range(self.num_files))
        dataset = tf.data.Dataset.from_generator(lambda: z, tf.uint32)

        def getitem(idx):
            # A tensor is passed in, so decode it here
            idx = idx.numpy()[0]
            return self[idx]

        f = lambda i: tf.py_function(
            func=getitem, inp=[i], Tout=[tf.float32, tf.float32]
        )

        # If we add batch here, then in getitem we need idx.numpy()[0]
        dataset = dataset.batch(1)
        dataset = dataset.map(f, num_parallel_calls=self.num_parallel_calls)

        # Since a file can contain multiple samples/patches, we need to rebatch the dataset
        dataset = dataset.unbatch()
        dataset = dataset.batch(self.batch_size)
        if self.prefetch is not None:
            dataset = dataset.prefetch(self.prefetch)  # .cache()

        return dataset

    def get_data_size(self):
        """Returns the number of bytes needed to store the full dataset"""
        size = (
            4
            * self.__len__()
            * (np.product(self.predictor_shape) + np.product(self.target_shape))
        )
        return size * self.num_patches_per_sample * self.num_samples_per_file

    def description(self):
        d = dict()

        d["Predictor shape"] = ", ".join(["%d" % i for i in self.predictor_shape])
        d["Target shape"] = ", ".join(["%d" % i for i in self.target_shape])
        d["Num files"] = self.num_files
        d["Samples per file"] = self.num_samples_per_file
        if self.patch_size is None:
            d["Num samples"] = len(self) * self.num_samples_per_file
        else:
            d["Patches per sample"] = self.num_patches_per_sample
            d["Num patches"] = self.num_patches
            d["Patch size"] = self.patch_size
        d["Num leadtimes"] = self.num_leadtimes
        d["Batch size"] = self.batch_size
        d["Num predictors"] = self.num_predictors
        d["Num targets"] = self.num_targets
        d["Predictors"] = list()
        if self.patch_size is None:
            d["Sample size (MB)"] = self.get_data_size() / 1024**2 / self.num_samples
        else:
            d["Patch size (MB)"] = self.get_data_size() / 1024**2 / self.num_patches
        d["Total size (GB)"] = self.get_data_size() / 1024**3
        if self.predictor_names is not None:
            for q in self.predictor_names:
                d["Predictors"] += [str(q)]
        return d

    def __str__(self):
        """Returns a string representation of the dataset"""
        return json.dumps(self.description(), indent=4)

    def verify(self, ofilename, model, quantiles, sampling=1):
        """Verifies a trained model against the dataset provided by this loader

        Args:
            ofilename (str): Filename to write verif file to
            model (keras.Model): A trained model object to verify. The predict function will be run
                on this model.
            quantiles (np.array): Quantiles used when model was trained
            sampling (int): Resample the output grid to this resolution (grid points)
        """

        # Check that we are verifying with a non-patched dataset
        if self.patch_size is not None:
            raise ValueError(
                f"Cannot verify with a loader here patch size ({self.patch_size}) != 1"
            )
        assert self.num_patches_per_sample == 1

        write_quantiles = False
        if len(quantiles) > 1 or quantiles[0] != 0.5:
            write_quantiles = True

        # Set up verif file
        points = self.get_grid_resampled(sampling).to_points()
        attributes = self.description()
        attributes.update(model.description())
        for k, v in attributes.items():
            attributes[k] = str(v)

        verif_kwargs = {}
        if write_quantiles:
            verif_kwargs["quantiles"] = quantiles
        ofile = maelstrom.output.VerifFile(
            ofilename,
            points,
            [i // 3600 for i in self.leadtimes],
            extra_attributes=attributes,
            **verif_kwargs,
        )

        if 0.5 not in quantiles:
            print(
                "Note: quantile=0.5 not in output. Determinsitic forecast not written to verif file"
            )

        # A cache for the observations: valid_time -> observations
        obs = set()

        # Loop over all files in the dataset and add data to verif file
        for i in range(len(self)):
            fcst, targets = self[i]
            targets = np.copy(targets)
            output = model.predict(fcst)
            assert output.shape[0] == 1
            output = output[0, ...]
            num_outputs = output.shape[-1]

            curr_time = self.times[i]

            # Add observations
            for j in range(len(self.leadtimes)):
                curr_valid_time = curr_time + self.leadtimes[j]
                if curr_valid_time not in obs:
                    curr_obs = np.reshape(
                        targets[0, j, ::sampling, ::sampling, 0], [points.size()]
                    )
                    ofile.add_observations(curr_valid_time, curr_obs)
                    obs.add(curr_valid_time)

            # Add determinsitic forecast
            if 0.5 in quantiles:
                I50 = quantiles.index(0.5)

                curr_fcst = output[..., I50]
                assert len(curr_fcst.shape) == 3

                curr_fcst = np.reshape(
                    curr_fcst[:, ::sampling, ::sampling],
                    [len(self.leadtimes), points.size()],
                )
                ofile.add_forecast(curr_time, curr_fcst)
                # print("Fcst", i, np.nanmean(curr_fcst))

            # Add probabilistic forecast
            if write_quantiles:
                curr_fcst = np.zeros(fcst.shape[1:-1] + [num_outputs], np.float32)
                for q in range(num_outputs):
                    curr_fcst[..., q] = output[..., q]
                    assert len(curr_fcst.shape) == 4

                curr_fcst = np.reshape(
                    curr_fcst[:, ::sampling, ::sampling, :],
                    [len(self.leadtimes), points.size(), num_outputs],
                )
                ofile.add_quantile_forecast(curr_time, curr_fcst)

        ofile.write()
        return ofile

    def get_grid_resampled(self, sampling=1):
        """Returns a gridpp.Grid that is resampled to a lower resolution"""
        lats = self.grid.get_lats()[::sampling, ::sampling]
        lons = self.grid.get_lons()[::sampling, ::sampling]
        elevs = self.grid.get_elevs()[::sampling, ::sampling]
        lafs = self.grid.get_lafs()[::sampling, ::sampling]

        return gridpp.Grid(lats, lons, elevs, lafs)


class FileLoader(DataLoader):
    def __init__(
        self,
        filenames,
        limit_leadtimes=None,
        limit_predictors=None,
        x_range=None,
        y_range=None,
        probabilistic_target=False,
        normalization=None,
        extra_features=[],
        debug=False,  # Debug is needed here, since it must be available before call to super()
        quick_metadata=False,
        **kwargs,
    ):
        """
        Args:
            filenames (list): List of filename strings
            limit_leadtimes (list): Only use these leadtimes
            limit_predictors (list): Only use these predictors
            normalization (str): Filename where normalization data is stored
            extra_featuers (list): List of dictionaries, where each element in the list is an extra
                feature that should be diagnosed on the fly. The dictionary has the following keys:
                type (str): One of 'altitude_diff', 'laf_diff', 'leadtime', 'x', 'y', name of a predictor
                halfwidth (int): Apply a neighbourhood mean to the field with this half width
                tpi_halfwidth (int): Compute topographical index using this halfwidth
                min (float): Don't let variable be less than this value
                max (float): Don't let variable be more than this value
            quick_metadata (bool): Determine the valid time based on the filename. This means that
                files are not checked if they have valid data.
        """
        if len(filenames) == 0:
            raise Exception("No forecast files")

        if not isinstance(extra_features, list):
            raise Exception(
                "Extra features must be a list, not a", type(extra_features)
            )

        self.debug = debug
        self.quick_metadata = quick_metadata

        self.x_range = x_range
        self.y_range = y_range
        self.probabilistic_target = probabilistic_target
        self.extra_features = extra_features

        (
            times,
            leadtimes,
            filenames,
            predictor_names,
            static_predictor_names,
            num_targets,
            num_samples_per_file,
            grid,
        ) = self.load_metadata(filenames)

        self.filenames = filenames
        num_files = len(filenames)

        self.leadtime_indices = None
        if limit_leadtimes is not None:
            limit_leadtimes_seconds = [i * 3600 for i in limit_leadtimes]
            self.leadtime_indices = [
                np.where(leadtimes == i)[0][0] for i in limit_leadtimes_seconds
            ]
            leadtimes = limit_leadtimes_seconds

        # What indicies in the predictor variable should be loaded?
        self.predictor_indices_to_load = None
        self.static_predictor_indices_to_load = None

        # What indices in the predictor variable will be kept at the end?
        self.predictor_indices_to_keep = None
        self.static_predictor_indices_to_keep = None

        # What predictor names will be loaded?
        self.predictor_names_to_load = list()
        self.static_predictor_names_to_load = list()
        if limit_predictors is not None:
            self.predictor_indices_to_load = list()
            self.static_predictor_indices_to_load = list()

            self.predictor_indices_to_keep = list()

            predictor_names_orig = [i for i in predictor_names]
            static_predictor_names_orig = [i for i in static_predictor_names]
            predictor_names = list()
            static_predictor_names = list()
            for i in limit_predictors:
                if i in predictor_names_orig:
                    self.predictor_indices_to_load += [predictor_names_orig.index(i)]
                    predictor_names += [i]
                elif i in static_predictor_names_orig:
                    self.static_predictor_indices_to_load += [
                        static_predictor_names_orig.index(i)
                    ]
                    static_predictor_names += [i]
                else:
                    raise ValueError(
                        f"Predictor {i} does not exist in dataset and is not a diagnosable feature"
                    )

            for feature in self.extra_features:
                required_predictors = self.get_required_predictors(feature)
                for required_predictor in required_predictors:
                    if required_predictor in predictor_names_orig:
                        self.predictor_indices_to_load += [
                            predictor_names_orig.index(required_predictor)
                        ]
                    elif required_predictor in static_predictor_names_orig:
                        self.static_predictor_indices_to_load += [
                            static_predictor_names_orig.index(required_predictor)
                        ]
                    else:
                        raise ValueError(
                            f"Predictor {required_predictor} does not exist in dataset"
                        )

            self.predictor_indices_to_load = list(set(self.predictor_indices_to_load))
            self.predictor_indices_to_load.sort()
            self.static_predictor_indices_to_load = list(
                set(self.static_predictor_indices_to_load)
            )
            self.static_predictor_indices_to_load.sort()

            for i in range(len(self.predictor_indices_to_load)):
                index = self.predictor_indices_to_load[i]
                self.predictor_names_to_load += [predictor_names_orig[index]]

            for i in range(len(self.static_predictor_indices_to_load)):
                index = self.static_predictor_indices_to_load[i]
                self.static_predictor_names_to_load += [
                    static_predictor_names_orig[index]
                ]

            for predictor in predictor_names:
                index = self.predictor_names_to_load.index(predictor)
                self.predictor_indices_to_keep += [index]

            for predictor in static_predictor_names:
                offset = len(self.predictor_names_to_load)
                index = self.static_predictor_names_to_load.index(predictor) + offset
                self.predictor_indices_to_keep += [index]
        else:
            for i in predictor_names:
                self.predictor_names_to_load += [i]
            for i in static_predictor_names:
                self.static_predictor_names_to_load += [i]

        all_predictor_names = predictor_names + static_predictor_names
        all_predictor_names += [
            self.get_feature_name(feature) for feature in self.extra_features
        ]

        super().__init__(
            times,
            leadtimes,
            all_predictor_names,
            num_targets,
            num_files,
            num_samples_per_file,
            grid,
            debug=debug,
            **kwargs,
        )

        self.coefficients = None
        if normalization is not None:
            with open(normalization) as file:
                self.coefficients = yaml.load(file, Loader=yaml.SafeLoader)

    def load_data(self, index):
        """This function needs to know what predictors/static predictors to load"""
        s_time = time.time()
        filename = self.filenames[index]
        mem_usage = maelstrom.util.get_memory_usage() / 1024**3
        self.write_debug(f"Loading {filename}: {mem_usage:.1f} GB memory")
        with netCDF4.Dataset(filename, "r") as ifile:

            def get(
                var,
                leadtime_indices=None,
                predictor_indices=None,
                x_range=None,
                y_range=None,
            ):
                s_time = time.time()
                if (
                    leadtime_indices is None
                    and predictor_indices is None
                    and x_range is None
                    and y_range is None
                ):
                    output = var[:]
                else:
                    has_sample = "sample" in var.dimensions
                    has_leadtime = "leadtime" in var.dimensions
                    if x_range is None:
                        x_range = slice(var.shape[has_sample + has_leadtime + 1])
                    if y_range is None:
                        y_range = slice(var.shape[has_sample + has_leadtime])
                    if has_leadtime:
                        if leadtime_indices is None:
                            leadtime_indices = slice(var.shape[has_sample])
                        if "predictor" in var.dimensions:
                            if predictor_indices is None:
                                predictor_indices = slice(var.shape[-1])
                            if "sample" in var.dimensions:
                                output = var[
                                    :,
                                    leadtime_indices,
                                    y_range,
                                    x_range,
                                    predictor_indices,
                                ]
                            else:
                                # print(var.shape, leadtime_indices, y_range, x_range, predictor_indices)
                                output = var[
                                    leadtime_indices,
                                    y_range,
                                    x_range,
                                    predictor_indices,
                                ]
                        else:
                            # This can be target fields
                            if "sample" in var.dimensions:
                                output = var[:, leadtime_indices, y_range, x_range]
                            else:
                                # print(var.shape, leadtime_indices, y_range, x_range)
                                output = var[leadtime_indices, y_range, x_range]
                    else:
                        if predictor_indices is None:
                            predictor_indices = slice(var.shape[-1])
                        if "sample" in var.dimensions:
                            output = var[:, y_range, x_range, predictor_indices]
                        else:
                            output = var[y_range, x_range, predictor_indices]

                e_time = time.time()

                return output.filled(np.nan)

            predictors = get(
                ifile.variables["predictors"],
                self.leadtime_indices,
                self.predictor_indices_to_load,
                self.x_range,
                self.y_range,
            )
            if "static_predictors" in ifile.variables:
                ss_time = time.time()
                if (
                    self.static_predictor_indices_to_load is None
                    or len(self.static_predictor_indices_to_load) > 0
                ):
                    temp = get(
                        # TODO: Leadtime indices
                        ifile.variables["static_predictors"],
                        None,
                        self.static_predictor_indices_to_load,
                        self.x_range,
                        self.y_range,
                    )

                    # Add leadtime dimension to static_predictors
                    has_sample = (
                        "sample" in ifile.variables["static_predictors"].dimensions
                    )
                    if has_sample:
                        # Add leadtime dimension after sample dimension
                        new_shape = (
                            (temp.shape[0],) + (predictors.shape[1],) + temp.shape[1:]
                        )
                        static_predictors = np.zeros(new_shape, np.float32)
                        for i in range(new_shape[1]):
                            static_predictors[:, i, ...] = temp
                    else:
                        new_shape = (predictors.shape[0],) + temp.shape
                        static_predictors = np.zeros(new_shape, np.float32)
                        for i in range(new_shape[0]):
                            static_predictors[i, ...] = temp
                    predictors = np.concatenate(
                        (predictors, static_predictors), axis=-1
                    )
                    ee_time = time.time()
                    self.timing["reshaping_static_predictors"] += ee_time - ss_time

            targets = get(
                ifile.variables["target_mean"],
                self.leadtime_indices,
                None,
                self.x_range,
                self.y_range,
            )
            targets = np.expand_dims(targets, -1)
            if self.probabilistic_target:
                # num_targets = 1 + self.probabilistic_target
                # target_shape = ifile.variables["target_mean"].shape + (num_targets,)
                # targets = np.zeros(target_shape, np.float32)
                std = get(
                    ifile.variables["target_std"],
                    self.leadtime_indices,
                    None,
                    self.x_range,
                    self.y_range,
                )
                std = np.expand_dims(std, -1)
                targets = np.concatenate((targets, std), axis=-1)

            if "sample" not in ifile.dimensions:
                predictors = np.expand_dims(predictors, 0)
                targets = np.expand_dims(targets, 0)
            # self.write_debug("time: %.1f s" % (time.time() - s_time))
            return predictors, targets

    def load_metadata(self, filenames):
        filename = filenames[0]
        s_time = time.time()
        with netCDF4.Dataset(filename, "r") as ifile:
            predictor_names = ifile.variables["predictor"][:]
            predictor_names = [i for i in predictor_names]
            leadtimes = ifile.variables["leadtime"][:].filled()

            static_predictor_names = list()
            if "static_predictor" in ifile.variables:
                static_predictor_names = ifile.variables["static_predictor"][:]
                static_predictor_names = [i for i in static_predictor_names]

            num_samples_per_file = 1
            if "sample" in ifile.dimensions:
                num_samples_per_file = len(ifile.dimensions["sample"])
            num_x = len(ifile.dimensions["x"])
            num_y = len(ifile.dimensions["y"])

            lats = ifile.variables["latitude"][:]
            lons = ifile.variables["longitude"][:]
            elevs = ifile.variables["altitude"][:]
            lafs = ifile.variables["land_area_fraction"][:]

            if self.x_range is not None:
                if np.max(self.x_range) > lats.shape[1]:
                    raise RuntimeError(f"x_range exceeds grid size ({lats.shape[1]})")

                lats = lats[:, self.x_range]
                lons = lons[:, self.x_range]
                elevs = elevs[:, self.x_range]
                lafs = lafs[:, self.x_range]
            if self.y_range is not None:
                if np.max(self.y_range) > lats.shape[0]:
                    raise RuntimeError(f"x_range exceeds grid size ({lats.shape[0]})")

                lats = lats[self.y_range, :]
                lons = lons[self.y_range, :]
                elevs = elevs[self.y_range, :]
                lafs = lafs[self.y_range, :]

            grid = gridpp.Grid(lats, lons, elevs, lafs)

        times = []
        required_vars = ["target_mean", "predictors", "time"]
        if "static_predictor" in ifile.dimensions:
            required_vars += ["static_predictors"]

        valid_filenames = list()

        for i in tqdm.tqdm(
            range(len(filenames)), desc="Loading metadata", disable=not self.debug
        ):
            filename = filenames[i]
            do_quick = False
            if self.quick_metadata:
                if (
                    re.match("^\d\d\d\d\d\d\d\dT\d\dZ.nc$", os.path.basename(filename))
                    is not None
                ):
                    do_quick = True
                else:
                    print(f"Cannot do quick metadata for {filename}")
            if do_quick:
                # Determine the valid time based on the filename. This does
                # not check if the file has valid data.
                datestamp = os.path.basename(filename).split(".")[0]
                yyyymmdd = int(datestamp[0:8])
                hour = int(datestamp[9:11])
                times += [maelstrom.util.date_to_unixtime(yyyymmdd, hour)]
                valid_filenames += [filename]
            else:
                with netCDF4.Dataset(filename, "r") as ifile:
                    has_all = True
                    for required_var in required_vars:
                        if required_var not in ifile.variables:
                            print(f"{filename} does not contain {required_var}")
                            has_all = False
                            break
                    if not has_all:
                        continue
                    curr_time = float(ifile.variables["time"][0])
                    times += [curr_time]
                    valid_filenames += [filename]

        num_targets = 1 + self.probabilistic_target

        return (
            times,
            leadtimes,
            valid_filenames,
            predictor_names,
            static_predictor_names,
            num_targets,
            num_samples_per_file,
            grid,
        )

    def get_required_predictors(self, feature):
        feature_type = feature["type"]
        if feature_type == "altitude_diff":
            return ["altitude", "model_altitude"]
        elif feature_type == "laf_diff":
            return ["land_area_fraction", "model_laf"]
        elif feature_type in ["leadtime", "x", "y"]:
            return []
        elif feature_type in ["diff", "multiply"]:
            output = list()
            if not isinstance(feature["left"], numbers.Number):
                output += [feature["left"]]
            if not isinstance(feature["right"], numbers.Number):
                output += [feature["right"]]
            return output
        else:
            return [feature_type]

    def compute_extra_features(self, predictors):
        def neighbourhood_static_field(ar, half_width, operator=gridpp.Mean):
            assert len(ar.shape) == 4
            q = ar[0, 0, ...]
            q = gridpp.neighbourhood(q, half_width, operator)
            out = np.zeros(ar.shape, np.float32)
            for t in range(ar.shape[0]):
                for i in range(ar.shape[1]):
                    out[t, i, ...] = q
            return out

        # Add new features
        if len(self.extra_features) > 0:
            num_leadtimes = predictors.shape[1]
            extra_values = np.zeros(
                predictors.shape[0:-1] + (len(self.extra_features),), np.float32
            )
            predictor_names_loaded = (
                self.predictor_names_to_load + self.static_predictor_names_to_load
            )
            for f, feature in enumerate(self.extra_features):
                feature_type = feature["type"]
                curr = np.zeros(predictors.shape[0:-1], np.float32)
                if feature_type == "altitude_diff":
                    I0 = predictor_names_loaded.index("altitude")
                    I1 = predictor_names_loaded.index("model_altitude")
                    curr = predictors[..., I1] - predictors[..., I0]
                elif feature_type == "laf_diff":
                    I0 = predictor_names_loaded.index("land_area_fraction")
                    I1 = predictor_names_loaded.index("model_laf")
                    curr = predictors[..., I1] - predictors[..., I0]
                elif feature_type in ["diff", "multiply"]:

                    def _get(predictors, predictor_names_loaded, arg):
                        if isinstance(arg, numbers.Number):
                            return arg
                        else:
                            I = predictor_names_loaded.index(arg)
                            return predictors[..., I]

                    right = _get(predictors, predictor_names_loaded, feature["right"])
                    left = _get(predictors, predictor_names_loaded, feature["left"])
                    if feature_type == "diff":
                        curr = left - right
                    elif feature_type == "multiply":
                        curr = left * right
                elif feature_type == "leadtime":
                    for i in range(num_leadtimes):
                        curr[:, i, ...] = i
                elif feature_type in ["x", "y"]:
                    Y = curr.shape[2]
                    X = curr.shape[3]
                    x, y = np.meshgrid(range(X), range(Y))
                    if feature_type == "x":
                        curr = np.tile(
                            x[None, None, ...], [curr.shape[0], curr.shape[1], 1, 1]
                        )
                    else:
                        curr = np.tile(
                            y[None, None, ...], [curr.shape[0], curr.shape[1], 1, 1]
                        )
                elif feature_type == "model_laf":
                    I = predictor_names_loaded.index("model_laf")
                    curr = predictors[..., I]
                else:
                    I = predictor_names_loaded.index(feature_type)
                    curr = predictors[..., I]

                if "tpi_halfwidth" in feature:
                    q = neighbourhood_static_field(
                        curr, feature["tpi_halfwidth"], gridpp.Mean
                    )
                    curr = curr - q
                if "halfwidth" in feature:
                    curr = neighbourhood_static_field(
                        curr, feature["halfwidth"], gridpp.Mean
                    )
                if "min" in feature:
                    curr[curr < feature["min"]] = feature["min"]
                if "max" in feature:
                    curr[curr > feature["max"]] = feature["max"]
                if "time_window" in feature:
                    w = feature["time_window"]
                    curr = np.cumsum(curr, axis=-1)
                    curr[..., w:] = curr[..., w:] - curr[..., 0:-w]
                extra_values[..., f] = curr

            if self.predictor_indices_to_keep is not None:
                predictors = predictors[..., self.predictor_indices_to_keep]
            predictors = np.concatenate((predictors, extra_values), axis=-1)
        return predictors

    def _process(self, predictors, targets):
        """This function needs to know what predictors to keep"""

        ss_time = time.time()
        predictors = self.compute_extra_features(predictors)
        self.timing["feature_extraction"] += time.time() - ss_time

        if self.predict_diff:
            ss_time = time.time()
            Ip = self.predictor_names.index("air_temperature_2m")

            # Code for target dimension:
            # The first target (target_mean) is also adjusted
            targets[..., 0] = targets[..., 0] - predictors[..., Ip]

            # Code for without a target dimension:
            # targets = predictors[..., Ip] - targets
            self.timing["predicting_difference"] += time.time() - ss_time

        # Normalization should be after predict diff, otherwise we are not truly predicting the
        # difference (unless targets are also normalized using the same normalization as for the
        # Ip predictor)
        ss_time = time.time()
        self.normalize(predictors, self.predictor_names)
        self.timing["normalization"] += time.time() - ss_time

        return predictors, targets

    def compute_normalization_coefficients(self):
        self.coeffients = dict()
        m = dict()
        m2 = dict()
        count = dict()
        for i in range(len(self)):
            predictors, targets = self[i]
            for k, name in enumerate(self.predictor_names):
                if name not in m:
                    m[name] = 0
                    m2[name] = 0
                    count[name] = 0
                m[name] += np.mean(predictors[..., k])
                m2[name] += np.mean(predictors[..., k] ** 2)
                count[name] += 1
        for k, name in enumerate(self.predictor_names):
            m[name] /= count[name]
            m2[name] /= count[name]
            curr_mean = m[name]
            curr_std = np.sqrt(m2[name] - m[name] ** 2)
            self.coefficients[self.predictor_names[k]] = [curr_mean, curr_std]
            print(name, self.coefficients[self.predictor_names[k]])

    def normalize(self, predictors, names):
        assert predictors.shape[-1] == len(names)

        if self.coefficients is None:
            return
        for p in range(predictors.shape[-1]):
            name = names[p]
            if name not in self.coefficients:
                # print(list(self.coefficients.keys()))
                # raise ValueError(f"Could not find normalization information for {name}")
                continue
            mean = self.coefficients[name][0]
            std = self.coefficients[name][1]
            predictors[..., p] -= mean
            predictors[..., p] /= std

    def denormalize(self, predictors, names):
        if self.coefficients is None:
            return
        if isinstance(names, list):
            for p in range(predictors.shape[-1]):
                name = names[p]
                if name not in self.coefficients:
                    # print(list(self.coefficients.keys()))
                    # raise ValueError(
                    #     f"Could not find normalization information for {name}"
                    # )
                    continue
                mean = self.coefficients[name][0]
                std = self.coefficients[name][1]
                predictors[..., p] *= std
                predictors[..., p] += mean
        else:
            name = names
            if name not in self.coefficients:
                # print(list(self.coefficients.keys()))
                # raise ValueError(f"Could not find normalization information for {name}")
                return
            mean = self.coefficients[name][0]
            std = self.coefficients[name][1]
            predictors[:] *= std
            predictors[:] += mean

    def get_feature_name(self, feature):
        if "name" in feature:
            return feature["name"]
        else:
            return feature["type"]


class FakeLoader(DataLoader):
    def generate_data(self):
        # predictors = np.random.randn(self.num_samples_per_file, *self.predictor_shape).astype(np.float32)
        # targets = np.random.randn(self.num_samples_per_file, *self.target_shape).astype(np.float32)
        predictors = np.zeros(
            [self.num_samples_per_file] + self.predictor_shape, np.float32
        )
        targets = np.zeros([self.num_samples_per_file] + self.target_shape, np.float32)
        return predictors, targets

    def __init__(
        self,
        num_files,
        num_predictors,
        num_targets,
        num_x,
        num_y,
        num_leadtimes,
        num_samples_per_file=1,
        read_delay=None,
        throughput=None,
        data_generator_func=None,
        **kwargs,
    ):
        """
        Args:
            read_delay (float): Delay in reading file in seconds
            throughput (float): Read throughput in bytes/s (typically 150MB for spinning disks,
                measured to be 147.8MB/s on the PPI)
        """
        self.read_delay = read_delay
        self.throughput = throughput
        times = np.arange(0, num_files * 24 * 3600, 24 * 3600)
        leadtimes = np.arange(0, num_leadtimes * 3600, 3600)
        predictor_names = ["predictor%d" % i for i in range(num_predictors)]
        lons, lats = np.meshgrid(np.linspace(0, 10, num_x), np.linspace(0, 10, num_y))
        grid = gridpp.Grid(lats, lons)
        if data_generator_func is None:
            self.data_generator_func = self.generate_data
        else:
            self.data_generator_func = data_generator_func
        super().__init__(
            times,
            leadtimes,
            predictor_names,
            num_targets,
            num_files,
            num_samples_per_file,
            grid,
            **kwargs,
        )

        self._orig_num_x = num_x
        self._orig_num_y = num_y

    def load_data(self, index):
        s_time = time.time()
        predictors, targets = self.data_generator_func()
        e_time = time.time()
        gen_time = e_time - s_time

        # Simulate a delay in loading the data, but account for the fact that the
        # generator already takes some time to complete
        delay = self.get_delay()
        if delay is not None:
            delay -= gen_time
            if delay > 0:
                time.sleep(delay)
        return predictors, targets

    def get_delay(self):
        delay = None
        if self.read_delay is not None:
            delay = self.read_delay
        elif self.throughput is not None:
            delay = self.get_data_size() / len(self) / self.throughput
        return delay


class Cache:
    def __init__(self, memory_size, loader_func, local_storage, local_storage_size):
        # self.local_storage = local_storage
        # self.local_storage_size = local_storage_size
        self.memory_size = memory_size
        self.memory_cache = dict()
        self.local_cache = dict()
        self.loader_func = loader_func

    def load(self, filename):
        if filename in self.memory_cache:
            return self.memory_cache[filename]
        # elif filename in self.local_cache:
        #     # Decide if we need to store in memory cache
        #     return self.local_cache[filename]
        else:
            data = self.loader_func(filename)
            # Decide if we need to store in memory cache
            # Decide if we need to store in local cache
            size = get_size(data)
            self.memory_cache[filename] = data
            if self.get_memory_cache_size() + size > self.memory_size:
                self.memory_cache.pop()
            return data

    @staticmethod
    def get_size(obj):
        size = 0
        for i in obj:
            size += sys.getsizeof(i)
        return size

    @property
    def get_memory_cache_size(self):
        return get_size(self.memory_cache)
