"""Creates a current class operator to couple with data."""
import numpy as np
import gpytorch as gpy
import torch
import copy
from scipy import interpolate
from fumes.model.utils import normalize_data, unnormalize_data
from fumes.model.gpsampler import GPSampler


class CurrMag(GPSampler):
    """Creates a current magnitude operator which inherits from GPSampler."""

    def magnitude(self, z, t):
        """Returns MLE functional for use in model class."""
        # Convert time to fractional hours from UTC 00:00:00
        t = t / 3600.
        # t = t % 24
        return self.cache_model(t)

    def sample_magnitude(self, num_samples):
        # trainx = copy.copy(self.trainx)
        # trainx = trainx.cpu().numpy()
        # return [interpolate.interp1d(self.trainx.cpu().numpy(), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        # return [interpolate.interp1d(self.trainx.cpu().numpy(), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        try:
            return [interpolate.interp1d(unnormalize_data(self.trainx.cpu().numpy(), self.x_minmax), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        except:
            return [interpolate.interp1d(unnormalize_data(self.trainx, self.x_minmax), self.sample(self.trainx, 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]


class CurrHead(GPSampler):
    """Creates a current heading operator which inherits from GPSampler."""

    def heading(self, t):
        """Returns MLE functional for use in model class."""
        # Convert time to fractional hours from UTC 00:00:00
        t = t / 3600
        # t = t % 24
        return self.cache_model(t) * np.pi / 180.

    def sample_heading(self, num_samples):
        # return [interpolate.interp1d(self.trainx, self.sample(self.trainx, 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        try:
            return [interpolate.interp1d(unnormalize_data(self.trainx.cpu().numpy(), self.x_minmax), self.sample(self.trainx.cpu().numpy(), 1) * np.pi / 180., bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        except:
            return [interpolate.interp1d(unnormalize_data(self.trainx, self.x_minmax), self.sample(self.trainx, 1) * np.pi / 180., bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
