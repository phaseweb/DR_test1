import numpy as np
import math

class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return math.exp(-(sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y)))/2*sigma**2)