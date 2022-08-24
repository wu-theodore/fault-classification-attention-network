import numpy as np

class MinMaxScale(object):
    """
    Rescales tensor to be within bounds defined by [max, min].
    """
    def __init__(self, max=1.0, min=-1.0):
        self.max = max
        self.min = min

    def __call__(self, sample):
        sample_min, sample_max = sample.min(), sample.max()
        scaled_sample = (sample - sample_min)/(sample_max - sample_min)*(self.max - self.min) + self.min
        assert(scaled_sample.min() == self.min and scaled_sample.max() == self.max)
        return scaled_sample

class Sample(object):
    """
    Samples tensor at defined sampling interval. E.g. sample_freq=2 means take every other data point.
    """
    def __init__(self, sample_freq=2):
        self.freq = sample_freq

    def __call__(self, sample):
        sample_indices = np.array([i for i in range(sample.shape[0]) if i % self.freq == 0])
        return sample[sample_indices, :]

class Compose(object):
    """
    Composes multiple transform objects.
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        for transform in self.transform_list:
            sample = transform(sample)

        return sample