"""Creates a parameter class."""
import numpy as np
import scipy as sp
from distfit import distfit
from sklearn.neighbors import KernelDensity


class Parameter(object):
    """Instantiates a parameter object."""

    def __init__(self, prior, proposal, limits):
        """Creates a parameter object.

        Args:
            prior (distfit object or float): prior
            proposal (scipy stats object or float): proposal
            limits (list of floats): whether there are hard extremes
        """
        self.dist = prior
        self.bandwidth = prior.bandwidth
        self.prop = proposal
        self.limits = limits

    def _get_best_distribution(self, data):
        """Fits a distribution over input data.

        Args:
            data (list[floats]): data to fit a distribution to

        Returns:
            dist (a scipy distribution object)
        """
        dist = distfit(smooth=10, distr='norm')  # force norm for now
        dist.fit_transform(data, verbose=0.)
        return dist

    def sample_proposal(self, num_samples):
        """Generates samples from proposal distribution.

        Args:
            num_samples (int): number of samples

        Returns:
            samples from proposal distribution
        """
        if np.isscalar(self.prop):
            return [self.prop for m in range(num_samples)]
        else:
            return self.prop.rvs(size=num_samples)

    def sample(self, num_samples):
        """Generates samples from distribution.

        Args:
            num_samples (int): number of samples

        Returns:
            samples from the distribution
        """
        if np.isscalar(self.dist):
            return np.asarray([self.dist for m in range(num_samples)])
        else:
            dist = getattr(sp.stats, self.dist.model['name'])
            loc = self.dist.model['loc']
            scale = self.dist.model['scale']
            args = self.dist.model['arg']
            if args:
                samples = dist.rvs(size=num_samples, *args,
                                   loc=loc, scale=scale)
            else:
                samples = dist.rvs(size=num_samples, loc=loc, scale=scale)
            return samples

    def update(self, data):
        """From obs of the empirical distribution, update.

        Args:
            data (list[float]): obs from empirical distribution
        """
        if np.isscalar(self.dist):
            pass
        else:
            self.dist = self._get_best_distribution(data)

    def predict(self, X):
        """Generates PDF probability.

        Args:
            X (list[float]): points to perform inference over

        Returns:
            probability of observation
        """
        if np.isscalar(self.dist):
            return 1.0
        else:
            dist = getattr(sp.stats, self.dist.model['name'])
            loc = self.dist.model['loc']
            scale = self.dist.model['scale']
            args = self.dist.model['arg']
            if args:
                pdf_fitted = dist.pdf(X, *args, loc=loc, scale=scale)
            else:
                pdf_fitted = dist.pdf(X, loc=loc, scale=scale)
            return pdf_fitted

    def get_attributes(self):
        """Provides meta data about a parameter."""
        attr_dict = {}

        if np.isscalar(self.dist):
            attr_dict['dist_name'] = 'is_scalar'
            attr_dict['dist_loc'] = self.dist
            attr_dict['dist_scale'] = None
            attr_dict['dist_arg'] = None
        else:
            attr_dict['dist_name'] = self.dist.model['name']
            attr_dict['dist_loc'] = self.dist.model['loc']
            attr_dict['dist_scale'] = self.dist.model['scale']
            attr_dict['dist_arg'] = self.dist.model['arg']

        if np.isscalar(self.prop):
            attr_dict['prop_name'] = 'is_scalar'
            attr_dict['prop_loc'] = self.prop
            attr_dict['prop_scale'] = None
        else:
            attr_dict['prop_name'] = str(self.prop.dist)
            attr_dict['prop_loc'] = self.prop.mean()
            attr_dict['prop_scale'] = self.prop.std()

        return attr_dict


class ParameterKDE(Parameter):
    """Creates a parameter class using sklearn instead of distfit"""

    def _get_best_distribution(self, data):
        """Fits a distribution over input data.

        Args:
            data (list[floats]): data to fit a distribution to

        Returns:
            dist (a scipy distribution object)
        """
        dist = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(data[:, np.newaxis])
        return dist

    def sample(self, num_samples):
        """Generates samples from distribution.

        Args:
            num_samples (int): number of samples

        Returns:
            samples from the distribution
        """
        if np.isscalar(self.dist):
            return np.asarray([self.dist for m in range(num_samples)])
        else:
            samples = self.dist.sample(num_samples)
            return samples

    def predict(self, X):
        """Generates PDF probability.

        Args:
            X (list[float]): points to perform inference over

        Returns:
            probability of observation
        """
        if np.isscalar(self.dist):
            return 1.0
        elif np.isscalar(X):
            if X >= self.limits[0] and X <= self.limits[1]:
                pdf_fitted = np.exp(self.dist.score_samples(np.asarray([X])[:, np.newaxis]))
                return pdf_fitted
            else:
                return 0.01
        else:
            X = np.asarray(X)
            mask = (X >= self.limits[0]) & (X <= self.limits[1])
            pdf_fitted = np.exp(self.dist.score_samples(X[:, np.newaxis]))
            pdf_fitted[~mask] = 0.01
            return pdf_fitted

    def get_attributes(self):
        """Provides meta data about a parameter."""
        attr_dict = {}

        if np.isscalar(self.dist):
            attr_dict['dist_name'] = 'is_scalar'
            attr_dict['dist_loc'] = self.dist
            attr_dict['dist_scale'] = None
            attr_dict['dist_arg'] = None
        else:
            attr_dict['dist_name'] = "KDE"
            attr_dict['dist_params'] = self.dist.get_params()
            attr_dict['dist_loc'] = self.dist.sample(1000).mean()
            attr_dict['dist_scale'] = self.dist.sample(1000).std()

        if np.isscalar(self.prop):
            attr_dict['prop_name'] = 'is_scalar'
            attr_dict['prop_loc'] = self.prop
            attr_dict['prop_scale'] = None
        else:
            attr_dict['prop_name'] = str(self.prop.dist)
            attr_dict['prop_loc'] = self.prop.mean()
            attr_dict['prop_scale'] = self.prop.std()

        return attr_dict
