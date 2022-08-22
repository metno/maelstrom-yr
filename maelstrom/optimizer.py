from tensorflow import keras
import tensorflow as tf
import maelstrom


def get(**kwargs):
    if "type" not in kwargs:
        raise ValueError("'type' is a required keyword of optimizer")

    name = kwargs["type"]
    name = name.lower()
    args = {k: v for k, v in kwargs.items() if k not in ["type", "learning_rate"]}
    if "learning_rate" in kwargs:
        args["learning_rate"] = get_learning_rate(kwargs["learning_rate"])
    if name == "adam":
        optimizer = keras.optimizers.Adam(**args)
    elif name == "nadam":
        optimizer = keras.optimizers.Nadam(**args)
    elif name == "ftrl":
        optimizer = keras.optimizers.Ftrl(**args)
    elif name == "sgd":
        optimizer = keras.optimizers.SGD(**args)
    elif name == "lars":
        optimizer = maelstrom.lars_optimizer.LARS(**args)
    elif name == "lamb":
        optimizer = maelstrom.lamb_optimizer.LAMB(**args)
        # optimizer = tf.tfa.optimizers.LAMB(**args)
    else:
        raise ValueError(f"Unknown optimizer {name}")

    return optimizer


class LinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self):
        pass

    def __call__(self, step):
        lr = 1e-3 * 10 ** (step / 1000)
        return lr


def get_learning_rate(args):
    """Get learning rate from configuration

    Example usage:
        lr = get_learning_rate(0.01)
        optimizer = keras.optimizers.Adam(learning_rate=lr)

    Args:
        args (float|dict): one of these:
            learning_rate: 0.01
            learning_rate:
                - type: exponential_decay
                  initial_learning_rate: 0.01
                - type: piecewise
                  boundaries: [10]
                  values: [1.0e-2, 1.0e-3]

    Returns:
        If args is numeric, a numerical value is returned
        in args is a dict, an object of type keras.optimizers.schedule is returned
    """
    if isinstance(args, dict):
        name = args["type"]
        name = name.lower()
        curr_args = {k: v for k, v in args.items() if k not in ["type"]}
        if name == "exponential_decay":
            learning_rate = keras.optimizers.schedules.ExponentialDecay(**curr_args)
        elif name == "linear":
            learning_rate = LinearSchedule()
        elif name == "piecewise":
            learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
                **curr_args
            )
        else:
            raise ValueError(f"Unknown learning rate schedule {name}")
    else:
        learning_rate = args
        if learning_rate is None:
            raise ValueError("learning_rate cannot be None")
    return learning_rate
