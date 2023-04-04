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

class ExtractTimeDomainFeatures(object):
    """
    Computes a set of 10 time domain features over the sequence dimension for each vehicle.
    Features:
        1. Root Mean Square (RMS)
        2. Square Root of Amplitude (SRA)
        3. Kurtosis Value (KV)
        4. Skewness Value (SV)
        5. Peak-to-Peak Value (PPV)
        6. Crest Factor (CF)
        7. Impulse Factor (IF)
        8. Marginal Factor (MF)
        9. Shape Factor (SF)
        10. Kurtosis Factor (KF)
    """
    def __init__(self):
        self.num_features = 10

    def __call__(self, sample):
        seq_len, num_vehicles = sample.shape

        features = np.zeros((num_vehicles * self.num_features))
        for i in range(num_vehicles):
            features[i * self.num_features + 0] = self.compute_RMS(sample[:, i])
            features[i * self.num_features + 1] = self.compute_SRA(sample[:, i])
            features[i * self.num_features + 2] = self.compute_KV(sample[:, i])
            features[i * self.num_features + 3] = self.compute_SV(sample[:, i])
            features[i * self.num_features + 4] = self.compute_PPV(sample[:, i])
            features[i * self.num_features + 5] = self.compute_CF(sample[:, i])
            features[i * self.num_features + 6] = self.compute_IF(sample[:, i])
            features[i * self.num_features + 7] = self.compute_MF(sample[:, i])
            features[i * self.num_features + 8] = self.compute_SF(sample[:, i])
            features[i * self.num_features + 9] = self.compute_KF(sample[:, i])
        
        return features

    def compute_RMS(self, x):
        return np.sqrt(np.mean(x**2))

    def compute_SRA(self, x):
        sum_of_root_amplitudes = np.mean(np.sqrt(np.abs(x)))
        return sum_of_root_amplitudes ** 2

    def compute_KV(self, x):
        mean = np.mean(x)
        variance = np.var(x)

        normalized_x = (x - mean) / variance
        return np.mean(normalized_x ** 4)

    def compute_SV(self, x):
        mean = np.mean(x)
        variance = np.var(x)

        normalized_x = (x - mean) / variance
        return np.mean(normalized_x ** 3)
    
    def compute_PPV(self, x):
        return np.max(x) - np.min(x)

    def compute_CF(self, x):
        rms = self.compute_RMS(x)
        return np.max(np.abs(x)) / rms

    def compute_IF(self, x):
        return np.max(np.abs(x)) / np.mean(np.abs(x))

    def compute_MF(self, x):
        sra = self.compute_SRA(x)
        return np.max(np.abs(x)) / sra

    def compute_SF(self, x):
        rms = self.compute_RMS(x)
        return rms / np.mean(np.abs(x))

    def compute_KF(self, x):
        kurtosis = self.compute_KV(x)
        return kurtosis / (np.mean(x ** 2) ** 2)

class Truncate(object):
    """
    Truncate the sequence. 
    """
    def __init__(self, truncate_size):
        self.truncate_size = truncate_size

    def __call__(self, sample):
        return sample[:self.truncate_size]

class GaussianNoise(object):
    """
    Add Gaussian noise to the signal.
    """
    def __init__(self, mean=0, variance=0.1):
        self.mean = mean
        self.variance = variance

    def __call__(self, sample):
        noise = np.random.normal(self.mean, self.variance, size=sample.shape)
        return sample + noise

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
