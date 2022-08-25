import copy
import inspect
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import maelstrom

"""This module contains various MAELSTROM models"""


def get(input_shape, num_outputs, **model_args):
    """Model factory. Initialize model with configuration.
    Args:
        input_shape (tuple): Shape of input (not including the sample dimension)
        num_outputs (int): Number of outputs in the network
        model_args (dict): Arguments to be passed to the model initialization method

    Returns:
        maelstrom.models.Model object
    """
    candidate_models = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    name = model_args["type"].lower()
    leadtime_dependent = (
        "leadtime_dependent" in model_args and model_args["leadtime_dependent"]
    )
    # leadtime_dependent = "leadtime_dependent" in model_args and model_args["leadtime_dependent"]

    kwargs = {
        k: v
        for k, v in model_args.items()
        if k not in ["type", "leadtime_dependent", "name"]
    }

    model = None
    for candidate_model in candidate_models:
        if name == candidate_model[0].lower():
            model = candidate_model[1](input_shape, num_outputs, **kwargs)
            break
    if model is None:
        raise NotImplementedError(f"Model {name} not recognized")

    if leadtime_dependent:
        model = LeadtimeModel(input_shape, num_outputs, model)

    return model


class Model(keras.Model):
    """Abstract model class

    Example use:
        model = maelstrom.models.TestModel(...)
        model.fit(...)

    Subclasses have access to the following:
        _input_shape (tuple): Shape of the input tensor (does not include the sample dimension)
        _num_outputs (int): Number of output fields that the network must produce

    """

    def __init__(self, input_shape, num_outputs):
        """Initializes the abstract model. Must be called by subclasses.

        Args:
            _input_shape (tuple): Shape of the input tensor (does not include the sample dimension)
            _num_outputs (int): Number of output fields that the network must produce
        """
        self._input_shape = input_shape
        self._num_outputs = num_outputs

        # Build the model
        inputs = keras.layers.Input(input_shape)
        outputs = self.get_outputs(inputs)

        super().__init__(inputs, outputs)

    def get_outputs(self, inputs):
        outputs = inputs
        for layer in self.get_layers():
            outputs = layer(outputs)
        return outputs

    def get_layers(self):
        """Returns the layers in the model

        Must be implemented by subclasses. Alternatively, override get_outputs.

        Returns:
            layers (list): A list of layers in the model. Each layer is uninitialized and the input
                layer is not included in this list.
        """
        raise NotImplementedError()

    def description(self):
        """Returns a description of the model

        Returns:
            dict: Dictionary of descriptive fields for the model
        """
        d = dict()
        d["type"] = str(type(self))
        d["input_shape"] = maelstrom.util.list_to_str(self._input_shape)
        d["num_outputs"] = self._num_outputs
        d.update(self._description())
        return d

    def _description(self):
        """Subclass specific description. Can be overridden by subclasses."""
        return {}


class SelectPredictor(Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        indices,
    ):
        if len(indices) == 0:
            raise ValueError("Needs at least one index")
        max_index = np.max(indices)
        if max_index > input_shape[-1]:
            raise ValueError(
                f"Max index ({max_index}) is larger than the number of predictors ({input_shape[-1]})"
            )
        self._indices = indices

        new_input_shape = get_input_size(input_shape, False, False)
        super().__init__(new_input_shape, num_outputs)

    def get_outputs(self, inputs):
        outputs = tf.expand_dims(inputs[..., self._indices[0]], axis=-1)
        for i in range(1, len(self._indices)):
            outputs = keras.layers.concatenate(
                (outputs, tf.expand_dims(inputs[..., self._indices[i]], axis=-1)),
                axis=-1,
            )
        return outputs


class LeadtimeModel(keras.Model):
    """Class that turns a 5D model into a set of separate models for each leadtime
    Not sure it is possible to derive from Model, because we can't define the model in terms
    of a list of layers, since we need to split the input tensor.
    """

    def __init__(self, input_shape, num_outputs, model, **kwargs):
        # TODO: Deal with possibility that model is spatially dependent
        spatial_dependent = False
        new_input_shape = get_input_size(input_shape, True, spatial_dependent)
        inputs = keras.layers.Input(shape=new_input_shape)

        # Wrap all layers of the model in a LeadtimeLayer
        outputs = inputs
        for layer in model.get_layers():
            layer = maelstrom.layers.LeadtimeLayer(layer)
            outputs = layer(outputs)

        super().__init__(inputs, outputs)
        self._description = copy.copy(model.description())

        # NOTE: Don't save model as a member variable, otherwise keras will try to train
        # its weights

    def description(self):
        return self._description


class BasicBenchmark(Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        filter_sizes=[1024, 256, 256],
        neighbourhood_size=1,
        time_size=1,
        hidden_layer_activation="relu",
        last_layer_activation="linear",
        leadtime_dependent=False,
        dropout=None,
        batch_normalization=False,
    ):
        self._filter_sizes = filter_sizes
        self._neighbourhood_size = neighbourhood_size
        self._time_size = time_size
        self._hidden_layer_activation = hidden_layer_activation
        self._last_layer_activation = last_layer_activation
        self._leadtime_dependent = leadtime_dependent
        self._dropout = dropout
        self._batch_normalization = batch_normalization

        new_input_shape = get_input_size(input_shape, self._leadtime_dependent, False)

        super().__init__(new_input_shape, num_outputs)

    def get_layers(self):
        layers = list()
        """
        if self.leadtime_dependent:
            # Implemented by reshaping and then using grouped convolution
            # Not checked that the grouping puts predictors for one leadtime together, and not the other
            # way around. This only works on GPUs.
            num_leadtimes = input_shape[0]
            filter_sizes = [size // num_leadtimes * num_leadtimes for size in filter_sizes]
            conv_size = [neighbourhood_size, neighbourhood_size]
            reshape = (input_shape[1], input_shape[2], input_shape[0] * input_shape[3])
            layers += [keras.layers.Reshape(reshape, input_shape=input_shape)]
            for size in filter_sizes:
                layers += [keras.layers.Conv2D(size, conv_size, padding='same',
                    activation=hidden_layer_activation, groups=num_leadtimes)]
            layers += [keras.layers.Conv2D(num_outputs * num_leadtimes, conv_size, padding='same',
                activation=last_layer_activation, groups=num_leadtimes)]
            output_shape = [input_shape[0], input_shape[1], input_shape[2], num_outputs]
            layers += [keras.layers.Reshape(output_shape, input_shape=self.output_shape)]
        else:
        """
        if len(self._input_shape) == 4:
            func = keras.layers.Conv3D
            conv_size = [
                self._time_size,
                self._neighbourhood_size,
                self._neighbourhood_size,
            ]
        else:
            func = keras.layers.Conv2D
            conv_size = [self._neighbourhood_size, self._neighbourhood_size]
        for size in self._filter_sizes:
            layers += [
                func(
                    size,
                    conv_size,
                    padding="same",
                    activation=self._hidden_layer_activation,
                )
            ]
            if self._batch_normalization:
                layers += [keras.layers.BatchNormalization()]
            if self._dropout is not None:
                layers += [keras.layers.Dropout(dropout)]

        layers += [
            func(
                self._num_outputs,
                conv_size,
                padding="same",
                activation=self._last_layer_activation,
            )
        ]

        return layers


class Regression(Model):
    """Simple linear regression model"""

    def __init__(self, input_shape, num_outputs):
        new_input_shape = get_input_size(input_shape, False, False)
        super().__init__(new_input_shape, num_outputs)

    def get_layers(self):
        layers = list()
        layers += [keras.layers.Dense(self._num_outputs, activation="linear")]
        return layers


class LocallyConnected(Model):
    def __init__(self, input_shape, num_outputs):
        super().__init__(input_shape, num_outputs)

    def get_layers(self):
        layers = list()
        # TODO:
        # layers += [keras.layers.Reshape(shape, input_shape=self._input_shape)]
        layer = keras.layers.LocallyConnected2D(1, (1, 1))
        layers += [
            maelstrom.layers.LeadtimeIndependentLayer(
                layer, remove_leadtime_dimension=True
            )
        ]
        layers += [keras.layers.Dense(self._num_outputs)]
        return layers


class Lstm(Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        filter_sizes=[1024, 256, 256],
        neighbourhood_size=1,
        hidden_layer_activation="relu",
        last_layer_activation="linear",
    ):
        self._filter_sizes = filter_sizes
        self._neighbourhood_size = neighbourhood_size
        self._hidden_layer_activation = hidden_layer_activation
        self._last_layer_activation = last_layer_activation

        new_input_shape = get_input_size(input_shape, False, True)

        super().__init__(new_input_shape, num_outputs)

    def get_layers(self):
        layers = list()
        conv_size = [self._neighbourhood_size, self._neighbourhood_size]
        for size in self._filter_sizes:
            layers += [
                keras.layers.ConvLSTM2D(
                    size,
                    conv_size,
                    padding="same",
                    activation=self._hidden_layer_activation,
                    return_sequences=True,
                )
            ]
        layers += [
            keras.layers.Dense(
                self._num_outputs, activation=self._last_layer_activation
            )
        ]
        return layers


class TestModel(Model):
    """Model for quick testing of layers"""

    def __init__(
        self,
        input_shape,
        num_outputs,
        size=1,
        num=1,
        with_leadtime=False,
        with_combinatorial=True,
    ):
        self._size = size
        self._num = num
        self._num_outputs = num_outputs
        self._with_leadtime = with_leadtime
        self._with_combinatorial = with_combinatorial

        new_input_shape = get_input_size(input_shape, self._with_leadtime, False)

        super().__init__(new_input_shape, num_outputs)

    def get_outputs(self, inputs):
        outputs = inputs
        if 1:
            # Insert a combinatorial layer with 3D convolution
            outputs0 = outputs
            # outputs0 = keras.layers.Dense(20, activation="linear")(outputs0)
            # outputs0 = maelstrom.layers.NormalizationLayer(5)(outputs0)
            if self._with_combinatorial:
                outputs0 = maelstrom.layers.CombinatorialLayer("multiply")(outputs0)
            outputs0 = keras.layers.TimeDistributed(
                keras.layers.SeparableConv2D(
                    5, (5, 5), activation="relu", padding="same"
                )
            )(outputs0)
            # outputs0 = keras.layers.Conv3D(5, (1, 5, 5), activation="relu", padding="same")(outputs0)
            # outputs1 = keras.layers.Conv3D(5, (1, 5, 5), activation="relu", padding="same")(outputs)
            outputs = outputs0
            # outputs1 = outputs
            # outputs = keras.layers.Concatenate(axis=-1)((outputs0, outputs1))
        else:
            outputs = inputs
        for i in range(self._num):
            # outputs = keras.layers.BatchNormalization()(outputs)
            outputs = keras.layers.Dense(self._size, activation="relu")(outputs)

        # Last layer
        if self._with_leadtime:
            smoothed_inputs = keras.layers.TimeDistributed(
                keras.layers.AveragePooling2D((5, 5), strides=(1, 1), padding="same")
            )(inputs)
            outputs = keras.layers.Concatenate(axis=-1)(
                (smoothed_inputs, inputs, outputs)
            )
            layer = keras.layers.Dense(self._num_outputs, activation="linear")
            outputs = maelstrom.layers.LeadtimeLayer(layer, "dependent")(outputs)
            # outputs = keras.layers.Dense(self._num_outputs, activation="linear")(outputs)

            # outputs = keras.layers.TimeDistributed(layer)(outputs)
        else:
            outputs = keras.layers.Dense(self._num_outputs, activation="linear")(
                outputs
            )

        return outputs


class Regression2(Model):
    """Model for quick testing of layers"""

    def __init__(self, input_shape, num_outputs, size=1, num=1, with_leadtime=False):
        self._size = size
        self._num = num
        self._with_leadtime = with_leadtime

        new_input_shape = get_input_size(input_shape, self._with_leadtime, False)

        super().__init__(new_input_shape, num_outputs)

    def get_outputs(self, inputs):
        outputs = inputs

        if 0:
            # Inputs + smoothed_inputs
            smoothed_inputs = keras.layers.TimeDistributed(
                keras.layers.AveragePooling2D((5, 5), strides=(1, 1), padding="same")
            )(inputs)
            outputs = keras.layers.Concatenate(axis=-1)((smoothed_inputs, inputs))
        else:
            outputs = maelstrom.layers.CombinatorialLayer("multiply")(outputs)
            outputs = keras.layers.TimeDistributed(
                keras.layers.SeparableConv2D(
                    3, (5, 5), activation="relu", padding="same"
                )
            )(outputs)
            outputs = keras.layers.TimeDistributed(
                keras.layers.SeparableConv2D(
                    3, (5, 5), activation="relu", padding="same"
                )
            )(outputs)
            outputs = keras.layers.Concatenate(axis=-1)((outputs, inputs))

        layer = keras.layers.Dense(self._num_outputs, activation="linear")
        outputs = maelstrom.layers.LeadtimeLayer(layer, "dependent")(outputs)

        return outputs


class Unet(Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        features=16,
        levels=3,
        pool_size=2,
        conv_size=3,
        with_leadtime=False,
    ):
        """U-net

        Args:
            features (int): Number of features in the first layer
            levels (int): Depth of the U-net
            pool_size (int): Pooling ratio (> 0)
            conv_size (int): Convolution size (> 0)
            with_leadtime (bool): Should the last layer be leadtime dependent?
        """
        self._features = features
        self._levels = levels
        self._pool_size = pool_size
        self._conv_size = conv_size
        self._with_leadtime = with_leadtime

        new_input_shape = get_input_size(input_shape, self._with_leadtime, False)

        super().__init__(input_shape, num_outputs)

    def get_outputs(self, inputs):
        outputs = inputs
        levels = list()

        features = self._features

        pool_size = [1, self._pool_size, self._pool_size]
        if 1:
            Conv = keras.layers.Conv3D
            hood_size = [1, self._conv_size, self._conv_size]
        elif 0:
            Conv = maelstrom.layers.TimeDistributedConv2D
            hood_size = [self._conv_size, self._conv_size]
        else:
            Conv = maelstrom.layers.SeparableConv2D
            hood_size = [self._conv_size, self._conv_size]

        levels += [outputs]
        # Downsampling
        for i in range(self._levels):
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = keras.layers.MaxPooling3D(pool_size=pool_size)(outputs)
            # print(i, outputs.shape)
            levels += [outputs]
            features *= 2

        features /= 2

        # Upsampling
        for i in range(self._levels - 1, -1, -1):
            features /= 2
            outputs = keras.layers.UpSampling3D(pool_size)(outputs)
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            # print(i, outputs.shape, levels[i].shape)
            outputs = keras.layers.concatenate((levels[i], outputs), axis=-1)

        if self._with_leadtime:
            layer = keras.layers.Dense(self._num_outputs, activation="linear")
            outputs = maelstrom.layers.LeadtimeLayer(layer, "dependent")(outputs)
        else:
            outputs = keras.layers.Dense(self._num_outputs, activation="linear")(
                outputs
            )

        return outputs


class Dense(Model):
    """Fully connected dense model"""

    def __init__(
        self,
        input_shape,
        num_outputs,
        layers,
        nodes,
        activation="relu",
        final_activation="linear",
    ):
        """
        Args:
            layers (int): Number of layers
            nodes (int): Number of nodes per layer
            activation (str): Activation function between layers
            final_activation (str): Activation function for output layer
        """
        new_input_shape = get_input_size(input_shape, False, False)
        self._num_layers = layers
        self._num_nodes = nodes
        self._activation = activation
        self._final_activation = final_activation
        super().__init__(new_input_shape, num_outputs)

    def get_layers(self):
        layers = list()
        for i in range(self._num_layers - 1):
            layers += [keras.layers.Dense(self._num_nodes, activation=self._activation)]
        layers += [
            keras.layers.Dense(self._num_outputs, activation=self._final_activation)
        ]
        return layers


class Custom(Model):
    """Customizable model"""

    def __init__(self, input_shape, num_outputs, layers, final_activation="linear"):
        """
        Args:
            layers (list): List of layer configurations (dict)
            final_activation (str): Activation function for the final layer
        """
        new_input_shape = get_input_size(input_shape, False, False)
        self._layers = layers
        self._final_activation = final_activation
        super().__init__(new_input_shape, num_outputs)

    def get_layers(self):
        layers = list()
        for i, config in enumerate(self._layers):
            layer = maelstrom.layers.get(**config)
        layers += [
            keras.layers.Dense(self._num_outputs, activation=self._final_activation)
        ]
        return layers


class FakeModel(keras.models.Model):
    def __init__(self, input_shape, output_shape, delay=None, throughput=None):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._delay = delay
        self._throughput = throughput
        super().__init__()
        super().build(input_shape)
        pass

    def description(self):
        return {}

    def call(self, inputs):
        def slow(x):
            if self._delay is not None:
                print(
                    "...delaying",
                )
                for i in inputs:
                    print(i.shape)
                time.sleep(self._delay)
            return x

        return slow


def get_input_size(input_shape, leadtime_dependent, spatial_dependent):
    """Convert input sizes to None when temporaly or spatially invariant"""
    # Spatial dimensions need not be specified
    new_input_shape = [None] * len(input_shape)
    new_input_shape[-1] = input_shape[-1]
    if spatial_dependent:
        new_input_shape[1] = input_shape[1]
        new_input_shape[2] = input_shape[2]
    if leadtime_dependent:
        new_input_shape[0] = input_shape[0]
    return new_input_shape
