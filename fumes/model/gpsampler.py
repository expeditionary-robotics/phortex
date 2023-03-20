"""Creates a GP Sampler"""

import numpy as np
import gpytorch as gpy
import torch
from scipy import interpolate

from fumes.environment.utils import ExactGPModel, ExactGPModelPeriodic
from fumes.model.utils import normalize_data, unnormalize_data


class GPSampler(object):
    """Instantiates a GP sampler object."""

    def __init__(self, datax, datay, training_iter=100, learning_rate=0.1, type="RBF"):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu" # manually select cpu
        self.x_minmax = (np.nanmin(datax), np.nanmax(datax))
        self.y_minmax = (np.nanmin(datay), np.nanmax(datay))

        self.trainx = torch.Tensor(normalize_data(datax, self.x_minmax)).to(device=self.device)
        self.trainy = torch.Tensor(normalize_data(datay, self.y_minmax)).to(device=self.device)

        self.likelihood = gpy.likelihoods.GaussianLikelihood().to(device=self.device)
        if type == "RBF":
            self.model = ExactGPModel(self.trainx, self.trainy, self.likelihood).to(device=self.device)
        elif type == "Periodic":
            self.model = ExactGPModelPeriodic(self.trainx, self.trainy, self.likelihood).to(device=self.device)

        self.training_iter = training_iter
        self.learning_rate = learning_rate

        self.train_model()
        self.interp_model()

    def train_model(self):
        """Trains the GP model."""
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.learning_rate)
        mll = gpy.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.training_iter):
            # zero gradients
            optimizer.zero_grad()
            # output from model
            output = self.model(self.trainx)
            # calc loss
            loss = -mll(output, self.trainy)
            loss.backward()
            optimizer.step()

    def interp_model(self):
        """Returns scipy approximator."""
        testx = np.linspace(self.x_minmax[0], self.x_minmax[1], 1000)
        mean, _ = self.get_mean_var(testx)

        self.cache_model = interpolate.interp1d(testx, mean, fill_value="extrapolate", bounds_error=False)

    def sample(self, X, num_samples=100):
        """Returns samples of model at X queries.

        Args:
            X (array[float]): queries
            num_samples (int): samples to draw

        Returns:
            samples of function at queries
        """
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            # TODO: remove this garbage and figure out devices :) 
            try:
                testx = torch.Tensor(normalize_data(X.cpu().numpy(), self.x_minmax)).to(device=self.device)
            except:
                testx = torch.Tensor(normalize_data(X, self.x_minmax)).to(device=self.device)
            samples = self.likelihood(self.model(testx)).rsample(torch.Size([num_samples]))
        return unnormalize_data(samples.cpu().numpy(), self.y_minmax)

    def get_mean_var(self, X):
        """Get the mean."""
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            testx = torch.Tensor(normalize_data(X, self.x_minmax)).to(device=self.device)
            pred = self.model(testx)
        return unnormalize_data(pred.mean.cpu().numpy(), self.y_minmax), pred.variance.cpu().detach().numpy()

    def get_attributes(self):
        """Provides meta data about Sampler."""
        sampler_dict = {}
        sampler_dict['datax'] = unnormalize_data(self.trainx.to(device="cpu"), self.x_minmax).tolist()
        sampler_dict['datay'] = unnormalize_data(self.trainy.to(device="cpu"), self.y_minmax).tolist()
        sampler_dict['training_iter'] = self.training_iter
        sampler_dict['learning_rate'] = self.learning_rate
        return sampler_dict
