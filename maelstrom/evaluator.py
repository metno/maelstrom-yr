import numpy as np

import maelstrom


class Evaluator:
    def evaluate(self, forecast_reference_time, fcst, targets):
        raise NotImplementedError()

    def close(self):
        pass


class Verif(Evaluator):
    def __init__(
        self, filename, leadtimes, points, quantiles=None, attributes=dict(), sampling=1
    ):
        self.filename = filename

        if 0.5 not in quantiles:
            print(
                "Note: quantile=0.5 not in output. Determinsitic forecast not written to verif file"
            )

        self.sampling = sampling
        self.leadtimes = leadtimes
        self.points = points
        self.quantiles = quantiles

        kwargs = dict()
        if len(quantiles) > 1 or quantiles[0] != 0.5:
            kwargs["quantiles"] = quantiles
            self.write_quantiles = True
        else:
            self.write_quantiles = False

        self.file = maelstrom.output.VerifFile(
            filename,
            points,
            [i // 3600 for i in self.leadtimes],
            extra_attributes=attributes,
            **kwargs,
        )

        # A cache for the observations: valid_time -> observations
        self.obs_cache = set()

    def evaluate(self, forecast_reference_time, fcst, targets):
        assert len(fcst.shape) == 5

        if fcst.shape[0] > 1:
            raise ValueError(
                f"Cannot create Verif file for datasets with multiple samples/patches ({fcst.shape[0]}) per forecast_reference_time"
            )
        assert fcst.shape[0] == 1
        assert targets.shape[0] == 1

        # Add observations
        for j in range(len(self.leadtimes)):
            curr_valid_time = forecast_reference_time + self.leadtimes[j]
            if curr_valid_time not in self.obs_cache:
                curr_obs = np.reshape(
                    targets[0, j, :: self.sampling, :: self.sampling, 0],
                    [self.points.size()],
                )
                self.file.add_observations(curr_valid_time, curr_obs)
                self.obs_cache.add(curr_valid_time)
                # print("obs:", curr_obs)

        # Add determinsitic forecast
        if 0.5 in self.quantiles:
            I50 = self.quantiles.index(0.5)

            curr_fcst = fcst[0, ..., I50]

            curr_fcst = np.reshape(
                curr_fcst[:, :: self.sampling, :: self.sampling],
                [len(self.leadtimes), self.points.size()],
            )
            self.file.add_forecast(forecast_reference_time, curr_fcst)
            # print("Fcst", i, np.nanmean(curr_fcst))

        # Add probabilistic forecast
        if self.write_quantiles:
            num_outputs = len(self.quantiles)
            curr_fcst = np.reshape(
                fcst[0, :, :: self.sampling, :: self.sampling, :],
                [len(self.leadtimes), self.points.size(), num_outputs],
            )
            self.file.add_quantile_forecast(forecast_reference_time, curr_fcst)
        self.file.sync()

    def close(self):
        self.file.write()


class Aggregator(Evaluator):
    def __init__(self, filename, leadtimes, loss):
        self.filename = filename
        self.leadtimes = leadtimes
        self.loss = loss
        with open(self.filename, "w") as file:
            file.write("unixtime leadtime obs fcst loss\n")

    def evaluate(self, forecast_reference_time, fcst, targets):
        """
        Args:
            forecast_reference_time (float): Forecast reference time of forecasts
            fcst (np.array): 5D array of forecasts (sample, leadtime, y, x, output_variable)
            targets (np.array): 5D array of targets (sample, leadtime, y, x, target_variable)
        """
        assert len(fcst.shape) == 5
        assert len(targets.shape) == 5

        with open(self.filename, "a") as file:
            for i in range(fcst.shape[1]):
                leadtime = self.leadtimes[i] // 3600
                curr_fcst = fcst[:, [i], ...]
                curr_targets = targets[:, [i], ...]
                curr_loss = self.loss(curr_targets, curr_fcst)
                file.write(
                    "%d %d %.5f %.5f %.5f\n"
                    % (
                        forecast_reference_time,
                        leadtime,
                        np.nanmean(curr_targets),
                        np.nanmean(curr_fcst),
                        curr_loss,
                    )
                )
