from tensorflow import keras
import copy
import inspect
import numpy as np
import sys
import tensorflow as tf

"""This module contains custom layers that can be used together with Keras and tensorflow"""


def get(**kwargs):
    curr_type = kwargs["type"]
    args = {k: v for k, v in kwargs.items() if k not in ["type"]}

    if curr_type == "dense":
        layer = keras.layers.Dense(**args)
    elif curr_type == "conv2d":
        layer = keras.layers.Conv2D(**args)
    else:
        candidate_layers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        for candidate_layer in candidate_layers:
            if curr_type.lower() == candidate_layer[0].lower():
                layer = candidate_layer[1](**args)
                break
        raise NotImplementedError(f"Unknown layer: {curr_type}")
    return layer


class LeadtimeLayer(keras.layers.Layer):
    """A layer that wraps an existing layer and runs it separately for each leadtime"""

    def __init__(self, layer, mode="dependent", remove_leadtime_dimension=False):
        super().__init__()
        # self._name = self._name + "_Yangula_bangula"

        if mode not in ["dependent", "independent"]:
            raise ValueError(f"Invalid mode {mode}")

        self.mode = mode
        self.layer = layer
        self.layer_weights = list()
        self.layers = list()
        self.remove_leadtime_dimension = remove_leadtime_dimension

    def build(self, input_shape):
        num_leadtimes = input_shape[1]
        # TODO: Why are we not inserting a singleton leadtime dimension here?
        # Don't some of the models require a leadtime dimension (except if
        # remove_leadtime_dimension=True)?
        temp_shape = [input_shape[0]] + input_shape[2:]

        if self.mode == "independent":
            layer = copy.copy(self.layer)
            layer.build(temp_shape)

        for i in range(num_leadtimes):
            if self.mode == "dependent":
                layer = copy.copy(self.layer)
                layer.build(temp_shape)
            # print(layer.weights)
            self.layers += [layer]
            for weights in layer.weights:
                self.layer_weights += [weights]

    def call(self, inputs):
        # print(inputs.shape)
        num_leadtimes = inputs.shape[1]
        outputs = list()
        for i in range(num_leadtimes):
            if self.remove_leadtime_dimension:
                layer = self.layers[i].call(inputs[:, i, ...])
                layer = tf.expand_dims(layer, 1)
            else:
                layer = self.layers[i].call(inputs[:, i : i + 1, ...])
            outputs += [layer]

        output = keras.layers.concatenate(outputs, axis=1)
        return output


class LeadtimeDependentLayer(LeadtimeLayer):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, "dependent", **kwargs)


class LeadtimeIndependentLayer(LeadtimeLayer):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, "independent", **kwargs)


class CombinatorialLayer(keras.layers.Layer):
    def __init__(
        self, operator, include_identity_terms=False, include_cross_terms=True
    ):
        if operator == "multiply":
            self._operator = tf.multiply
        elif operator == "add":
            self._operator = tf.add
        else:
            raise ValueError(f"{operator} not a recognized operator ")

        self._include_identity_terms = include_identity_terms
        self._include_cross_terms = include_cross_terms

        super().__init__(trainable=False)

    def call(self, inputs):
        num_predictors = inputs.shape[-1]
        q = list()
        for i in range(num_predictors):
            start = i if self._include_identity_terms else i + 1
            end = num_predictors if self._include_cross_terms else i + 1
            J = range(start, end)
            for j in J:
                out = self._operator(inputs[..., i], inputs[..., j])
                out = tf.expand_dims(out, -1)
                q += [out]

        q = keras.layers.concatenate(q, axis=-1)
        return q


class NormalizationLayer(keras.layers.Layer):
    def __init__(self, width):
        self._width = width
        super().__init__()

    def call(self, inputs):
        m = keras.layers.AveragePooling2D((self._width, self._width), padding="same")(
            inputs
        )
        m = keras.layers.UpSampling2D((self._width, self._width))(m)
        m = tf.subtract(inputs, m)

        s2 = keras.layers.AveragePooling2D((self._width, self._width), padding="same")(
            m * m
        )
        s = tf.add(tf.sqrt(s2), 1)
        s = keras.layers.UpSampling2D((self._width, self._width))(s)
        return tf.divide(m, s)

        # m = keras.layers.Conv2D(1, (self._width, self._width), padding="same")(inputs)
        m = keras.layers.AveragePooling2D((self._width, self._width), padding="same")(
            inputs
        )
        m = keras.layers.UpSampling2D((self._width, self._width))(m)
        m = tf.subtract(inputs, m)

        s2 = keras.layers.AveragePooling2D((self._width, self._width), padding="same")(
            m * m
        )
        s2 = keras.layers.UpSampling2D((self._width, self._width))(s2)
        s = tf.sqrt(s2)
        m = tf.divide(m, s)
        return m
        return tf.subtract(inputs, m)


class GroupedLayer(keras.layers.Layer):
    def __init__(self, operator, num_groups=None, inv=False):
        """Initialize layer
        Args:
            operator (str): one of 'multiply', 'add'
            num_groups (int): Number of groups to operate on
            inv (bool): If True...
        """
        if operator == "multiply":
            self._operator = tf.multiply
        elif operator == "add":
            self._operator = tf.add
        else:
            raise ValueError(f"{operator} not a recognized operator ")
        self._num_groups = num_groups
        self._inv = inv

        super().__init__(trainable=False)

    def build(self, input_shape):
        num_predictors = input_shape[-1]
        if num_predictors % self._num_groups != 0:
            raise ValueError(
                "num_groups must be a multiple of the number of predictors"
            )

    def call(self, inputs):
        num_predictors = inputs.shape[-1]
        if self._inv:
            raise NotImplementedError()
            step = num_predictors // self._num_groups
            out = inputs[..., 0:step]
            for i in range(1, self._num_groups):
                start = i * step
                end = start + step
                out = self.operator(out, inputs[..., start:end])
        else:
            num_predictors = inputs.shape[-1]
            step = num_predictors // self._num_groups
            out = inputs[..., 0:step]
            for i in range(1, self._num_groups):
                start = i * step
                end = start + step
                out = self._operator(out, inputs[..., start:end])
        return out


class SeparableConv2D(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.layer = keras.layers.TimeDistributed(
            keras.layers.SeparableConv2D(*args, **kwargs)
        )
        super().__init__()

    def call(self, inputs):
        return self.layer(inputs)


class TimeDistributedConv2D(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.layer = keras.layers.TimeDistributed(keras.layers.Conv2D(*args, **kwargs))
        super().__init__()

    def call(self, inputs):
        return self.layer(inputs)
