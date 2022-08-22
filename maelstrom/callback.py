import copy
import json
import netCDF4
import numpy as np
import queue
import threading
import time
from tensorflow import keras

import maelstrom

"""This is a collection of callbacks that can be passed to keras model fit, or maelstrom.trainer.fit

    Example usage:

    callbacks = [WeightsCallback(model, filename)]
    model.fit(dataset, callbacks=callbacks)
"""


class WeightsCallback(keras.callbacks.Callback):
    """Writes parameter weights and metrics to netCDF file. The callback records values as training
    progresses and writes the file when training is finished
    """

    def __init__(self, model, filename):
        """
        Args:
            model: a tensorflow model
            filename (str): Filename to write weights and metrics
        """
        if not isinstance(filename, str):
            raise ValueError(f"filename={filename} must be a string")

        self.model = model
        self.filename = filename

        self.num_epochs = 0
        self.num_batches = 0

        # Metrics that are recorded for every batch
        self.batch_metrics = dict()

        # Batch metrics for the current epoch
        self.curr_batch_metrics = dict()

        # Parameter weights recorded for every batch
        self.batch_weights = list()

        # Parameter weights for the current epoch
        self.curr_batch_weights = list()

        # Metrics that are recorded for each epoch only
        self.epoch_metrics = dict()

    def on_train_end(self, logs=None):
        self.write(self.filename)

    def on_train_batch_end(self, batch, logs=None):
        if len(self.model.layers) == 0:
            return

        for k, v in logs.items():
            if k not in self.curr_batch_metrics:
                self.curr_batch_metrics[k] = list()
            self.curr_batch_metrics[k] += [v]

        # Get weights from the last layer
        curr = list()
        for layer in self.model.layers:
            for s in layer.get_weights():
                curr += [np.array(s).flatten()]
        if len(curr) > 0:
            # curr can have length 0 if there are no traininable parameters
            self.curr_batch_weights += [np.concatenate(curr)]

        # print(self.model.layers[0].get_weights())
        # weights = np.squeeze(self.model.layers[0].get_weights())
        self.num_batches += 1

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs += 1
        self.batch_weights += [self.curr_batch_weights]
        for k, v in self.curr_batch_metrics.items():
            if k not in self.batch_metrics:
                self.batch_metrics[k] = list()
            self.batch_metrics[k] += [v]

        for k, v in logs.items():
            if k not in self.batch_metrics:
                if k not in self.epoch_metrics:
                    self.epoch_metrics[k] = list()
                self.epoch_metrics[k] += [v]

        self.curr_batch_weights = list()
        self.curr_batch_metrics = dict()

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    @property
    def batches_per_epoch(self):
        return int(self.num_epochs / self.num_batches)

    def write(self, filename):
        weights = np.array(self.batch_weights)
        if 0 in weights.shape:
            print(
                f"Not writing weights file {filename} since there are no training parameters"
            )
            return
        num_weights = weights.shape[2]
        maelstrom.util.create_directory(filename)
        with netCDF4.Dataset(filename, "w") as file:
            file.createDimension("epoch", self.num_epochs)
            file.createDimension("batch", self.batches_per_epoch)
            file.createDimension("weight", num_weights)
            var = file.createVariable("weights", "f4", ("epoch", "batch", "weight"))
            var[:] = weights
            file.num_epochs = self.num_epochs
            for k, v in self.batch_metrics.items():
                var = file.createVariable(k, "f4", ("epoch", "batch"))
                var[:] = np.array(v)
            for k, v in self.epoch_metrics.items():
                var = file.createVariable(k, "f4", ("epoch"))
                var[:] = np.array(v)


class Timing(keras.callbacks.Callback):
    """Class for reporting maelstrom timing results for benchmarking"""

    def __init__(self, logger):
        self.times = []
        self.start_time = time.time()
        self.epochs = list()
        self.results = dict()
        self.logger = logger
        self.num_batches = 0
        self.num_epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        end_time = time.time()
        total_time = end_time - self.start_time
        self.epochs += [total_time]
        self.num_epochs = epoch + 1

    def on_train_batch_end(self, batch, logs):
        self.num_batches += 1

    def on_train_end(self, logs={}):
        end_time = time.time()
        epoch_times = np.diff([0] + self.epochs)
        total_time = end_time - self.start_time
        average_time = total_time / len(epoch_times)
        self.results["total_time"] = total_time
        self.results["average_epoch_time"] = average_time
        self.results["first_epoch_time"] = epoch_times[0]
        self.results["min_epoch_time"] = np.min(epoch_times)
        self.results["max_epoch_time"] = np.max(epoch_times)

        for k, v in self.results.items():
            self.logger.add("Timing", "Training", k, v)

        batch_size = self.num_batches / self.num_epochs

        self.logger.add(
            "Timing",
            "Training",
            "time_per_batch",
            self.results["average_epoch_time"] / batch_size,
        )

    def get_results(self):
        return self.results


class Convergence(keras.callbacks.Callback):
    """Write (training and test) loss information to text file

       Deprecated. Use Validation instead
    """

    def __init__(
        self,
        filename,
        verif_style=False,
        flush_each_epoch=True,
        include_batch_logs=False,
        leadtime_is_batch=False,
    ):
        """Initialize object

        Args:
            filename (str): Where to write output file to
            verif_style (bool): If True, write file using verif text format
            flush_each_epoch (bool): Flush data to file after each epoch
            include_batch_logs (bool): Write scores for each batch in addition to each epoch
            leadtime_is_batch (bool): Use the batch number (not epoch) as the leadtime
        """
        self.filename = filename
        self.verif_style = verif_style
        self.results = list()
        self.batch_results = list()
        self.num_batches = 0

        maelstrom.util.create_directory(self.filename)
        self.file = open(self.filename, "w")
        self.header = None
        self.flush_each_epoch = flush_each_epoch
        self.include_batch_logs = include_batch_logs
        self.leadtime_is_batch = leadtime_is_batch
        self.curr_leadtime = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.include_batch_logs:
            self.batch_results += [(batch, logs)]
        self.num_batches += 1

    def on_epoch_begin(self, epochs, logs=None):
        self.batch_results.clear()
        self.num_batches = 0

    def on_epoch_end(self, epoch, logs={}):
        # Step size of each batch
        if self.leadtime_is_batch:
            dx = 1
        else:
            dx = 1.0 / self.num_batches

        if self.include_batch_logs:
            # Skip the last batch, since this would be identical to the epoch logs
            for i in range(len(self.batch_results) - 1):
                batch, batch_logs = self.batch_results[i]
                x = self.curr_leadtime + float(batch + 1) * dx
                self.results += [(x, batch_logs)]

        self.curr_leadtime += self.num_batches * dx
        x = self.curr_leadtime
        self.results += [(x, logs)]
        if self.flush_each_epoch:
            self.write()

    def write(self):
        if self.header is None:
            keys = list()
            for epoch, line in self.results:
                print(epoch, line)
                keys += line.keys()
            keys = list(set(keys))  # ["epoch"] + [k for k in line.keys()]
            self.header = keys
            keys = ["epoch"] + keys
            if self.verif_style:
                # Rename certain headings to verif-compatible fields
                rename = dict()
                rename["epoch"] = "leadtime"
                rename["loss"] = "fcst"
                rename["val_loss"] = "obs"

                keys = [rename[k] if k in rename else k for k in keys]
            self.file.write(" ".join([f"{k}" for k in keys]))
            self.file.write("\n")

        for epoch, line in self.results:
            self.file.write(f"{epoch} ")
            for key in self.header:
                if key in line.keys():
                    self.file.write("%s " % line[key])
                else:
                    self.file.write("-999 ")
            self.file.write("\n")
        self.file.flush()
        self.results.clear()

    def on_train_end(self, logs={}):
        self.write()
        self.file.close()

    def get_results(self):
        return self.results


class Validation(keras.callbacks.Callback):
    """This callback runs validation after a certain number of batches and stores training scores

    Args:
        filename (str): Write validation to this filename
        model (keras.Model): Model to run validation on
        dataset (tf.Dataset): Validation dataset
        frequency (int): Run validation after this many batches. If None, run at the end of epoch
        logger (maelstorm.logger.Logger): Write timing info to this logger
        verif_style (bool): If true, format output file to be compatible with github.com/WFRT/Verif
    """

    def __init__(
        self,
        filename,
        model,
        dataset,
        frequency=None,
        logger=None,
        verif_style=True,
    ):
        self.model = model
        self.dataset = dataset
        self.frequency = frequency

        self.filename = filename
        self.verif_style = verif_style
        self.logger = logger

        self.results = list()
        self.batch_results = list()

        maelstrom.util.create_directory(self.filename)
        self.file = open(self.filename, "w")
        self.header = None
        self.count = 0
        self.num_validation = 0
        self.final_validation_scores = dict()

        # self.queue = queue.Queue()
        # threading.Thread(target=self._process, daemon=True).start()
        self.do_async = False

        self.total_time = 0
        self.start_time = 0
        self.acc_size = 0

    def _process(self):
        while True:
            func, args = self.queue.get()
            func(*args)
            self.queue.task_done()

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.acc_size = 0

    def on_train_batch_end(self, batch, logs=None):
        # We need batch + 1, so that we validate after each file (not before)
        if self.frequency is not None and (batch + 1) % self.frequency == 0:
            if self.do_async:
                self.queue.put((self.run, (self.epoch, batch, logs)))
            else:
                self.run_validation(self.epoch, batch, logs)
            self.write()

    def run_validation(self, epoch, batch, logs):
        s_time = time.time()
        count = 0
        val_logs = dict()
        for x, y in self.dataset:
            ss_time = time.time()
            curr_logs = self.model.test_on_batch(x=x, y=y, return_dict=True)
            for k, v in curr_logs.items():
                new_key = "val_%s" % k
                if new_key not in val_logs:
                    val_logs[new_key] = 0
                val_logs[new_key] += v
            count += 1
            self.acc_size += np.product(x.shape) * 4

        for k, v in val_logs.items():
            val_logs[k] = v / count

        # Add validation to logs, so that it shows up on the tensorflow bar
        for k, v in val_logs.items():
            logs[k] = v

        # Make a copy, so that we don't add the other metrics to the tensorflow bar
        results_logs = copy.copy(logs)
        curr_acc_time = time.time() - self.start_time
        results_logs["step"] = self.count
        results_logs["epoch"] = epoch
        results_logs["batch"] = batch
        results_logs["acc_time"] = curr_acc_time
        results_logs["size_gb"] = self.acc_size / 1024 ** 3
        self.results += [results_logs]

        self.count += 1
        self.num_batches += 1
        e_time = time.time()
        self.total_time = e_time - s_time
        self.num_validation += 1
        self.final_validation_scores = val_logs
        print("Validation time: ", e_time - s_time)

    def on_epoch_begin(self, epochs, logs=None):
        self.batch_results.clear()
        self.num_batches = 0
        self.epoch = epochs

    def on_epoch_end(self, epoch, logs={}):
        if self.frequency is None:
            if self.do_async:
                self.queue.put((run, self.epoch, batch))
            else:
                self.run_validation(self.epoch, self.num_batches, logs)
            self.write()

    def write(self):
        if self.header is None:
            keys = list()
            for line in self.results:
                keys += line.keys()
            keys = list(set(keys))
            preferred_order = [
                "step",
                "epoch",
                "batch",
                "acc_time",
                "size_gb",
                "val_loss",
                "loss",
            ]

            in_keys = [k for k in preferred_order if k in keys]
            out_keys = [k for k in keys if k not in preferred_order]
            keys = in_keys + out_keys
            self.header = keys

            if self.verif_style:
                # Rename certain headings to verif-compatible fields
                rename = dict()
                rename["step"] = "leadtime"
                rename["loss"] = "fcst"
                rename["val_loss"] = "obs"

                keys = [rename[k] if k in rename else k for k in keys]

            self.file.write(" ".join([f"{k}" for k in keys]))
            self.file.write("\n")

        for line in self.results:
            for key in self.header:
                if key in line.keys():
                    self.file.write("%s " % line[key])
                else:
                    self.file.write("-999 ")
            self.file.write("\n")
        self.file.flush()
        self.results.clear()

    def on_train_end(self, logs={}):
        self.write()
        self.file.close()
        if self.do_async:
            self.queue.join()

        if self.logger is not None:
            self.logger.add("Timing", "Training", "validation_time", self.total_time)
            self.logger.add("Timing", "Training", "num_validation", self.num_validation)
            for k, v in self.final_validation_scores.items():
                self.logger.add("Scores", k, v)


class Testing(keras.callbacks.Callback):
    def __init__(self, filename, leadtimes, loss):
        self.filename = filename
        self.leadtimes = leadtimes
        self.loss = loss
        with open(self.filename, "w") as file:
            file.write("unixtime leadtime obs fcst loss\n")

    def on_predict_batch_end(self, batch, logs=None):
        with open(self.filename, "a") as file:
            for i in range(fcst.shape[1]):
                leadtime = self.leadtimes[i] // 3600
                curr_fcst = fcst[:, [i], ...]
                curr_targets = targets[:, [i], ...]
                curr_loss = self.loss(curr_targets, curr_fcst)
                file.write(
                    "%d %d %.2f %.2f %.2f\n"
                    % (
                        forecast_reference_time,
                        leadtime,
                        np.nanmean(curr_targets),
                        np.nanmean(curr_fcst),
                        curr_loss,
                    )
                )

    def on_predict_end(self, logs=None):
        print("#")
