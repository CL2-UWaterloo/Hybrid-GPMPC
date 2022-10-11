import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import math

from ds_utils import GP_DS

class GP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """

        Parameters
        ----------
        train_x torch tensor of the shape (num_samples, input_dim)
        train_y torch tensor of the shape (num_samples, 1) since we assume all residual terms are independent and
                consists of stacks of 1-D outputs from independently trained GPs
        likelihood type of likelihood function to be used. Can vary depend on the kernel used
        """
        super(GP_Model, self).__init__(train_x, train_y, likelihood)
        self.train_x, self.train_y = train_x, train_y
        # Initialize the prior mean function to be a constant 0-vector'd mean as in Girard's paper
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance is a combination of the SE kernel with a scaling factor as is normal
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def get_hyperparameters(self,
                            as_numpy=False
                            ):
        """
        Code adapted from safe-control-gym
        Get the outputscale and lengthscale from the kernel matrices of the GPs.
        """
        lengthscale_list = []
        output_scale_list = []
        noise_list = []
        for gp in self.gp_list:
            lengthscale_list.append(gp.model.covar_module.base_kernel.lengthscale.detach())
            output_scale_list.append(gp.model.covar_module.outputscale.detach())
            noise_list.append(gp.model.likelihood.noise.detach())
        lengthscale = torch.cat(lengthscale_list)
        outputscale = torch.Tensor(output_scale_list)
        noise = torch.Tensor(noise_list)
        if as_numpy:
            return lengthscale.numpy(), outputscale.numpy(), noise.numpy(), self.K_plus_noise.detach().numpy()
        else:
            return lengthscale, outputscale, noise, self.K_plus_noise

    def forward(self, x):
        # Contains mean and covariance for the GP using the datapoints available.
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(model, likelihood_fn, optimizer, lossfn_callable, train_x, train_y, num_iter=400, verbose=True,
          return_trained_covs=False):
    loss_fn = lossfn_callable(likelihood_fn, model)
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        # print(train_x.shape, output.mean.shape, train_y.shape, train_x.dtype, train_y.dtype, output.mean.dtype)
        assert output.mean.dtype == train_y.dtype, "True labels and output labels must have matching dtypes in order to compute loss." \
                                                   "Current dtypes are: output.mean: %s, train_y: %s" % (
                                                   output.mean.dtype, train_y.dtype)
        loss = -loss_fn(output, train_y)
        loss.backward()
        if verbose:
            if i % 25 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
        optimizer.step()

    # All kernels have a lengthscale parameter applied independently to each dimension as indicated here
    # https://docs.gpytorch.ai/en/v1.6.0/kernels.html
    # if verbose:
    print('lengthscale: %.5f noise variance (sigma**2): %.5f, noise std dev (sigma): %.5f' % (
        model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item(), math.sqrt(model.likelihood.noise.item())))

    if return_trained_covs:
        return math.sqrt(model.likelihood.noise.item())
    else:
        return None


def get_gpds_attrs(gp_ds, no_squeeze=False):
    train_x = gp_ds.train_tensor.T
    if not no_squeeze:
        train_x = train_x.squeeze()
    return train_x, gp_ds.train_y.type(torch.FloatTensor), gp_ds.num_regions, gp_ds.output_dims


# TODO: Add handling for trained covs for multi-dim case for both pw and baseline method
def train_test(gp_ds, no_squeeze=False, verbose=True, return_trained_covs=False):
    '''
    This is the most common likelihood used for GP regression.
    (MLE of a Gaussian is Gaussian itself http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html).
    There are other options for exact GP regression, such as the [FixedNoiseGaussianLikelihood](http://docs.gpytorch.ai/likelihoods.html#fixednoisegaussianlikelihood),
    which assigns a different observed noise value to different training inputs but not sure if something like that would be of particular
    importance for a control system design when using a non-linear function to approximate residual dynamics.
    '''
    # print(gp_ds.train_x.shape)
    train_x, train_y, _, _ = get_gpds_attrs(gp_ds, no_squeeze=no_squeeze)
    # print("Test fn train_x shape: %s train_y shape: %s" % (train_x.shape, train_y.shape))
    likelihoods, independent_models = [], []
    for idx in range(train_y.shape[0]):
        i_dim_labels = train_y[idx, :].T
        # Squeeze is required other it yields an error saying that grad can be implicitly created only for scalar outputs.
        # Even though the ith row is scalar it still has a second unnecessary dimension so squeeze to remove.
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP_Model(train_x, i_dim_labels.squeeze(), likelihood)
        likelihoods.append(likelihood)
        independent_models.append(model)

    # Conventional pytorch training loop for hyperparameter optimization
    trained_covs = []
    for idx, model in enumerate(independent_models):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # Marginal loss as in Girard's paper
        trained_cov = train(model=model, likelihood_fn=likelihoods[idx],
                            optimizer=optimizer, lossfn_callable=gpytorch.mlls.ExactMarginalLogLikelihood,
                            train_x=train_x, train_y=train_y[idx, :].T, verbose=verbose, return_trained_covs=return_trained_covs)
        trained_covs.append(trained_cov)

    if not return_trained_covs:
        return likelihoods, independent_models
    else:
        return likelihoods, independent_models, trained_covs


def piecewise_train_test(gp_ds, no_squeeze=False, verbose=True, return_trained_covs=False):
    train_x, train_y, num_regions, dims = get_gpds_attrs(gp_ds, no_squeeze=no_squeeze)
    # print(train_x.shape, train_y.shape)
    regionwise_sample_idxs = gp_ds.get_region_idxs()
    likelihoods, models = [], []
    trained_covs = []
    for dim_idx in range(dims):
        for region_idx in range(len(gp_ds.regions)):
            if verbose:
                print("Training model: %s for region: %s" % (dim_idx + 1, region_idx + 1))
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            region_idx_samples = train_x[regionwise_sample_idxs[region_idx]]
            region_idx_labels = train_y[dim_idx, regionwise_sample_idxs[region_idx]]
            model = GP_Model(region_idx_samples, region_idx_labels, likelihood)
            likelihoods.append(likelihood)
            models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            trained_cov = train(model=model, likelihood_fn=likelihood,
                                optimizer=optimizer, lossfn_callable=gpytorch.mlls.ExactMarginalLogLikelihood,
                                train_x=region_idx_samples, train_y=region_idx_labels, verbose=verbose,
                                return_trained_covs=return_trained_covs)
            trained_covs.append(trained_cov)

    # print(likelihoods[0](models[0](train_x[regionwise_sample_idxs[0]])))
    if not return_trained_covs:
        return likelihoods, models
    else:
        return likelihoods, models, trained_covs


def trained_gp_viz_test_no_hm(models, likelihoods, gp_ds: GP_DS, fineness_param=(21, 51), model_type='global'):
    colours = ['r', 'g', 'b', 'cyan']
    dims, num_regions = gp_ds.input_dims, gp_ds.num_regions
    num_plots = dims * 2
    fig = plt.figure(figsize=(16, 16))
    axes = []
    for plot_idx in range(num_plots):
        ax = fig.add_subplot(2, dims, plot_idx + 1, projection='3d')
        axes.append(ax)

    fine_grid, fine_regionwise_idxs = gp_ds.plot_true_func_2d(axes=axes[:dims], fineness_param=fineness_param,
                                                              true_plot=True, samples_plot=False, colours=colours)

    for model, likelihood in zip(models, likelihoods):
        model.eval()
        likelihood.eval()

    colours = gp_ds.get_colours(1 if model_type == 'global' else gp_ds.num_regions)

    with torch.no_grad():
        for dim_idx in range(dims):
            if model_type == "local":
                dim_models, dim_likelihoods = [temp[dim_idx * num_regions: (dim_idx + 1) * num_regions] for temp in
                                               [models, likelihoods]]
                for region_idx in range(len(gp_ds.regions)):
                    region_samples = gp_ds.ret_regspec_samples([fine_grid], region_idx, fine_regionwise_idxs)[0]
                    # print(type(region_samples), region_samples.shape)
                    region_ops = dim_likelihoods[region_idx](dim_models[region_idx](torch.from_numpy(region_samples.T)))
                    axes[dims + dim_idx].scatter3D(region_samples[0, :], region_samples[1, :],
                                                   region_ops.mean, color=colours[region_idx])
            else:
                # print(dim_idx)
                # print(type(models[dim_idx](torch.from_numpy(fine_grid.T))), type(models[dim_idx]))
                ops = likelihoods[dim_idx](models[dim_idx](torch.from_numpy(fine_grid.T)))
                axes[dims + dim_idx].scatter3D(fine_grid[0, :], fine_grid[1, :],
                                               ops.mean, color=colours[0])


#

# TODO: Fix issue with model and likelihood inversion
def trained_gp_viz_test(models, likelihoods, gp_ds: GP_DS, fineness_param=(21, 51), model_type='global'):
    colours = ['r', 'g', 'b', 'cyan']
    dims, num_regions = gp_ds.input_dims, gp_ds.num_regions
    # First row plots true function, 2nd row plots learnt function, last row plots covariance heatmaps
    num_plots = dims * 3
    fig = plt.figure(figsize=(16, 16))
    axes = []
    for plot_idx in range(dims * 2):
        ax = fig.add_subplot(3, dims, plot_idx + 1, projection='3d')
        axes.append(ax)
    # Last row plots are 1-D
    for plot_idx in range(dims * 2, dims * 3):
        ax = fig.add_subplot(3, dims, plot_idx + 1)
        axes.append(ax)

    fine_grid, fine_regionwise_idxs = gp_ds.plot_true_func_2d(axes=axes[:dims], fineness_param=fineness_param,
                                                              true_plot=True, samples_plot=False, colours=colours)
    _, x_idxs = np.unique(fine_grid[0, :], return_index=True)
    _, y_idxs = np.unique(fine_grid[1, :], return_index=True)
    fine_x1_size, fine_x2_size = len(list(x_idxs)), len(list(y_idxs))
    # print(fine_x_size, fine_y_size)

    cov_mats = torch.zeros(fine_grid.shape)
    for model, likelihood in zip(models, likelihoods):
        model.eval()
        likelihood.eval()

    colours = gp_ds.get_colours(1 if model_type == 'global' else gp_ds.num_regions)

    with torch.no_grad():
        for dim_idx in range(dims):
            if model_type == "local":
                dim_models, dim_likelihoods = [temp[dim_idx * num_regions: (dim_idx + 1) * num_regions] for temp in
                                               [models, likelihoods]]
                for region_idx in range(len(gp_ds.regions)):
                    region_samples = gp_ds.ret_regspec_samples([fine_grid], region_idx, fine_regionwise_idxs)[0]
                    # print(type(region_samples), region_samples.shape)
                    # TODO: Add samples dots to plot.
                    region_ops = dim_likelihoods[region_idx](dim_models[region_idx](torch.from_numpy(region_samples.T)))
                    axes[dims + dim_idx].scatter3D(region_samples[0, :], region_samples[1, :],
                                                   region_ops.mean, color=colours[region_idx])
            else:
                # print(dim_idx)
                print(type(models[dim_idx](torch.from_numpy(fine_grid.T))), type(models[dim_idx]))
                ops = likelihoods[dim_idx](models[dim_idx](torch.from_numpy(fine_grid.T)))
                # ops = models[dim_idx](likelihoods[dim_idx](torch.from_numpy(fine_grid.T)))
                axes[dims + dim_idx].scatter3D(fine_grid[0, :], fine_grid[1, :],
                                               ops.mean, color=colours[0])
                # print(get_user_attributes(ops))
                print(type(ops.variance), ops.variance.shape, cov_mats[dim_idx, :].shape)
                cov_mats[dim_idx, :] += ops.variance

    for dim_idx in range(dims):
        cov_mat = cov_mats[dim_idx, :].reshape(fine_x1_size, fine_x2_size)
        # fine_x, fine_y = np.sort(np.unique(fine_grid[0, :])), np.sort(np.unique(fine_grid[1, :]))
        # print(fine_x, fine_y)
        # print(fine_x.shape, fine_y.shape)
        # axes[2*dims+dim_idx].pcolormesh(fine_x, fine_y, cov_mat[:-1, :-1],
        #                                 vmin=torch.min(cov_mat), vmax=torch.max(cov_mat),
        #                                 shading='flat')
        axes[2 * dims + dim_idx].pcolor(cov_mat)
