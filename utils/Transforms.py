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