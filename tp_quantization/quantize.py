import numpy as np


def quantize(float_value, min_range, max_range, zero=0):
    b = 8
    S = (max_range - min_range) / (2 ** b - 1)
    quantized = (float_value / S).astype(np.int8)
    return quantized


def to_float(uint_values, min_range, max_range):
    b = 8
    S = (max_range - min_range) / (2 ** b - 1)
    float_values = uint_values * S
    return float_values



