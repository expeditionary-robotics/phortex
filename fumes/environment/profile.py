"""Creates a profile class."""

import torch
import gpytorch as gpy
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from fumes.model.gpsampler import GPSampler
from fumes.model.utils import normalize_data, unnormalize_data
from fumes.environment.utils import pacific_sp_T, pacific_sp_S, curfunc


class Profile(GPSampler):
    """Instantiates a profile class for salinity and temperature."""

    def profile(self, z):
        """Returns profile functional."""
        return self.cache_model(z)

if __name__ == "__main__":
    t = np.linspace(0, 12*3600, 50)
    prof = Profile(t, curfunc(None, t))
    plt.plot(t, curfunc(None, t))
    plt.plot(t, prof.profile(t))
    plt.show()