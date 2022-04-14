"""Fully observeable 2D model."""

from .model import Model
import torch
import gpytorch

class GPC(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        """Create a standard gp classifier in pytorch"""
        super(GPC, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)))
    
    def forward(self, x):
        """Specify the forward pass"""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GPClassifier(Model):
    def __init__(self, environment, pre_trained=None):
        """Initialize gp classifier.

        Args:
            environment (Environment): an Environment object
            pre_trained (GPytorch GP Model): a pre-trained GP model
        """
        self.env = environment
        if pre_trained is None:
            self.model = None
        else:
            self.model = pre_trained
    
    def train(self, train_x, train_y, training_iter=50):
        """Train the classifier from input data
        
        Args:
            train_x (torch Tensor): input observations
            train_y (torch Tensor): labels
        """
        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(target=train_y, learn_additional_noise=True)
        model = GPC(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, likelihood.transformed_targets).sum()
            loss.backward()
            optimizer.step()
        
        self.model = model
        self.likelihood = likelihood


    def get_value(self, t, loc):
        """Get a deterministic prediction of the model output t and loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xy or xyz space

                Returns (float): the concentration value
                """
        model = self.model
        likelihood = self.likelihood
        model.eval()
        likelihood.eval()

        test_x = torch.Tensor([t, loc]) #TODO

        with gpytorch.settings.fast_pred_var), torch.no_grad():
            pred = model(test_x).loc
        
        return pred

    def get_prediction(self, t, loc):
        """Get a (mean, variance) prediction of the model output t and loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xyz or xy space

                Returns:
                        tuple(float, float): mean and variance of concentration value
                """
        model = self.model
        likelihood = self.likelihood
        model.eval()
        likelihood.eval()

        test_x = torch.Tensor([t, loc]) #TODO

        with gpytorch.settings.fast_pred_var), torch.no_grad():
            pred_obj = model(test_x)
            pred = pred_obj.loc
            pred_var = pred_obj.
        
        return pred, 0.0 #TODO

    def get_snapshot(self, t):
        """Get a deterministic prediction of full state at time t.

                Args:
                        t (float): the global time

                Returns:
                        np.array: full state snapshot at time t
                """
        # create a grid over the world
        # draw and return samples from the grid
        pass #TODO

    def get_snapshot_prediction(self, t):
        """Get a deterministic prediction of full state and uncertainty at t.

                Args:
                        t (float): the global time

                Returns:
                        np.array: mean state snapshot
                        np.array: predicted variance of state snapshot
                """
        # create a grid over the world
        # draw and return samples from the grid
        pass #TODO

    def update(self, t, loc, obs):
        """Update model with observation at t, loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xyz or xy space
                        obs (tuple[float]): a sensor observation
                """
        pass #TODO Do we want to refit hyperparameters?

    def update_multiple(self, t, loc, obs):
        """Update model with a list of observations at t, loc.

        Args:
            t (list[float]): the global time
            loc (list[tuple[float]]): a list of locations, in xyz or xy space
            obs (list[tuple[float]]): a list of sensor observations
        """
        pass #TODO Do we want to refit hyperparameters?
