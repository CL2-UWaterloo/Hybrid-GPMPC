import math
import os.path

import torch
import matplotlib.pyplot as plt
import pyDOE
import numpy as np
import pprint
from collections import defaultdict

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.transforms import Bbox
# from mpl_toolkits import mplot3d

get_num_means = lambda x: max([dim_range[1] - dim_range[0] for dim_range in len(x)])


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def dir_exist_or_create(base_path, sub_path=None):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    store_dir = base_path
    if sub_path is not None:
        if not os.path.exists(base_path+sub_path):
            os.mkdir(base_path+sub_path)
        store_dir = base_path+sub_path
    return store_dir


def save_subplot(ax, figure, fig_name=None, base_path="C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\images\\",
                 sub_path="scalar_motivating_example\\"):
    assert fig_name is not None, "Need to set the fig_name attribute"
    # Save just the portion _inside_ the second axis's boundaries
    extent = full_extent(ax).transformed(figure.dpi_scale_trans.inverted())
    store_dir = dir_exist_or_create(base_path, sub_path=sub_path)
    figure.savefig(store_dir+fig_name+'.svg', bbox_inches=extent)


# Confidence region plotter
# Source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html#the-plotting-function-itself
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


class DS_Sampler:
    def __init__(self, dims, method, num_samples, min_max_range, **kwargs):
        """
        Note: Sample methods are implemented assuming each dim has a min, max range that is independent of the others
        i.e. there is no additional p-norm (p \neq \infty) constraint on the vector itself which is why each
        dimension's 0->1 output range can be scaled independently of the others.

        :param dims: dimension of the vector to be sampled.
        :param method: sampling method to be used.
        Current methods: uniform random, clustering (pre-specified means),
        :param num_samples: number of samples to be generated.
        :param min_max_range: list of tuples. assert len(list) == dims and list[i][0], list[i][1] gives the min and max value
        for the ith dim respectively.

        kwargs:
        if method == 'cluster_sampling'
        means (np.array): for clustering based sampling, this generates data clustered around pre-specified means.
        means.shape = (N, M) N = dimension of sample, M = num_means
        if kwargs.get('means') == None, the means are randomly selected while ensuring the kwargs.get('num_means')
        are atleast kwargs.get('mean_separation_dist') (default for mean_separation dist is (num_means/(max-min))/k)
        where k is some scaling constant to ensure the means are not just limited to be equidistant)
        apart from each other.
        kwargs.get('sampling_variances') is a list of length dims where each element specifies the covariance
        of the specific dimension ex: [0.1, 0.2] means we sample 2-d vectors from a multivariate gaussian
        around mean[i, :] with covariance = np.diag([0.1, 0.2]). See if there's any practical cases that might warrant
        allowing the acceptance of non-diagonal matrices.
        kwargs.get('cluster_weights') can be provided to give the weighting of number of samples generated for each
        distribution around a given mean. No need to sum to 1 since will normalize. ex: [1, 2] for 2 means implies
        twice as many samples are generated for the distribution centred around the second mean than the first.
        Default is uniform weighting.

        TODO:
        uniform random sampling, Random means clustering method
        Allow separate sample variances for each mean
        """
        self.dims = dims
        self.method = method
        self.num_samples = num_samples
        self.min_max_range = min_max_range
        self.config_options = kwargs.get('config_options', {})

        self.sample_variances = kwargs.get('sample_variances', None)

        if self.method == 'clustering':
            self.means = kwargs.get('means', None)
            if self.means is None:
                # Just a random function that gets the max range of all dimensions and then converts it to int
                # to get number of means. Divide the range by desired custom scaling factor to adjust means as desired
                custom_scaling_factor = 1
                self.num_means = kwargs.get('num_means', int(get_num_means(self.min_max_range)))
                # Refer to Resources.xoj for idea to implementation
                self.mean_separation_dist = 1
                if kwargs.get('sampling_variances', None) is None:
                    self.generate_sampling_variances()
                # While enforcing separation distance for this method, also ensure that means are sampled from
                # shrunk min-max range that ensures that each mean has their k-sigma regions within the original range
                # where k is a tunable parameter. Maybe just formulate as a CSP but how to specify cost in a way
                # that allows to generate any possible mean set randomly.
                self._generate_random_means()
            else:
                self.sample_variances = np.diag(self.sample_variances)

            # Default (2nd arg) is uniform weighting
            self.cluster_weights = self.normalize(kwargs.get('cluster_weights', np.ones(len(self.means[0, :]))))
            self.meanwise_numsamples = self.mean_specific_num_samples()

    def normalize(self, x):
        return (x / np.sum(x))

    def mean_specific_num_samples(self):
        return self.cluster_weights * self.num_samples

    def get_samples(self):
        if self.method == 'uniform random':
            return self._ur_sampling()

        if self.method == 'clustering':
            return self._clst_sampling()

    def _clst_sampling(self):
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
        # Returns an array of size (num_samples, self.dims)
        samples = np.zeros((self.means.shape[0], 1))
        for idx in range(self.means.shape[1]):
            mean = self.means[:, idx]
            temp = np.random.multivariate_normal(mean, self.sample_variances,
                                                 size=int(self.meanwise_numsamples[idx]))
            samples = np.hstack((samples, temp.T))
            # Clips samples in-place
            self._clip_samples(samples)
        # Discard initial column used to help with concatenation
        return samples[:, 1:]

    def _clip_samples(self, samples):
        for i in range(len(self.min_max_range)):
            idx_range = self.min_max_range[i]
            np.clip(samples[i, :], idx_range[0], idx_range[1], out=samples[i, :])

    def _get_meansample_idxs(self):
        temp = [0]
        end_idxs = list(np.cumsum(self.meanwise_numsamples))
        temp.extend(end_idxs)
        return np.array(temp, dtype=np.int32)

    def _get_cmap(self):
        """
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.
        """
        args = np.linspace(0, 0.8, num=len(self.means[0, :]) * 2)
        p = plt.cm.get_cmap('plasma')
        return [p(randomizer) for randomizer in args]

    def viz_2d_clst(self, samples, n_std=2):
        fig, ax = plt.subplots(figsize=self.config_options.get('figsize', (9, 3)))
        delimiters = self._get_meansample_idxs()
        colours = self._get_cmap()
        for i in range(delimiters[:-1].shape[0]):
            mean_samples = samples[:, delimiters[i]:delimiters[i + 1]]
            cart_coords = [mean_samples[axis_idx, :] for axis_idx in range(self.means.shape[0])]
            plt.scatter(*mean_samples, c=colours[2 * i])
            confidence_ellipse(*cart_coords, ax, n_std=n_std, edgecolor=colours[2 * i])
            plt.scatter(*[self.means[axis_idx, i] for axis_idx in range(self.means.shape[0])], c=colours[2 * i + 1])
        plt.show()

    # TODO:
    def _generate_random_means(self):
        pass

    def _ur_sampling(self):
        # To move from [0, 1) to [start_limit, end_limit) multiply by end_limit-start_limit and then shift by start_limit
        dimwise_vectors = np.vstack([np.random.rand(self.num_samples)*(self.min_max_range[i][1] - self.min_max_range[i][0])+self.min_max_range[i][0]
                                     for i in range(len(self.min_max_range))])
        # print(self.min_max_range)
        # print(dimwise_vectors)
        return dimwise_vectors

    @staticmethod
    def viz_2d_ur(samples):
        plt.figure()
        plt.scatter(samples[0, :], samples[1, :], c='r')
        plt.show()


def generate_fine_grid(start_limit, end_limit, fineness_param, viz_grid=False):
    # print(start_limit[0, :])
    # print(start_limit, start_limit.shape[-1])
    fine_coords_arrs = [torch.linspace(start_limit[idx, :].item(), end_limit[idx, :].item(),
                                       int(fineness_param[idx]*(end_limit[idx, :]-start_limit[idx, :]).item())) for idx in range(start_limit.shape[0])]
    meshed = np.meshgrid(*fine_coords_arrs)
    grid_vectors = np.vstack([mesh.flatten() for mesh in meshed]) # Shape = dims * num_samples where num_samples is controller by fineness_param and start and end lims
    # print(grid_vectors.shape)

    if viz_grid:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(*[grid_vectors[axis_idx, :] for axis_idx in range(grid_vectors.shape[0])], c='b')
    return grid_vectors


class GP_DS():
    def __init__(self, train_x, callables, state_space_polytope, regions, noise_vars, output_dims=None, **kwargs):
        """
        :param train_x => input data generated by DS_sampler class
        :param callables => number of callables = size of output vector

        TODO:
        Not recomm'd:
        kwargs.get(output_dims) => number of output_dims
        kwargs.get(Bd) => If number of output_dims > callables, then use Bd to zero pad vector and limit residuals to those
        dimensions dictated by Bd. Not really recomm'd since will have to use Bd for noise vector too so better to
        do both together.
        Outputs
        train_y => (train_x, train_y) used to train the GP model
        Extensions: Accept sympy functions as inputs instead of hardcoding them and use lambdify to be able
        to apply it to numpy arrays.
        Refs:
        https://stackoverflow.com/questions/58784150/how-to-integrate-sympy-and-numpy-in-python-numpy-array-of-sympy-symbols
        https://stackoverflow.com/questions/58738051/add-object-has-no-attribute-sinh-error-in-numerical-and-symbolic-expression/58744816#58744816
        https://docs.sympy.org/latest/modules/utilities/lambdify.html
        """
        self.train_x = train_x
        self.train_tensor = self._convert_to_2d_tensor(self.train_x)
        self.input_dims, self.num_samples = self.train_tensor.shape
        self.output_dims = output_dims
        # input_dims specifies the dimension of the input vector and output_dims specifies the dimension of the residual vector.
        if self.output_dims is None:
            self.output_dims = self.input_dims
        self.state_space_polytope = state_space_polytope
        self.regions = regions
        self.num_regions = len(self.regions)
        self.noise_vars = noise_vars
        self.callables = callables
        self.num_funcs = len(self.callables)
        assert self.num_funcs == self.output_dims*self.num_regions, "Number of func callables must be = dims*num_regions. If functions repeated across multiple positions then pass them multiple times"

    @staticmethod
    def _convert_to_2d_tensor(input_arr):
        return torch.Tensor(np.array(input_arr, ndmin=2))

    def _generate_regionspec_mask(self, input_arr=None):
        """
        :param input_samples: Data samples generated by custom random sampling method. Num samples = num columns.
        Num rows = dim of vector space from which samples are drawn
        :param regions: List of box constraints
        :return:
        """
        # If no input array then we're working with the created dataset instead of some passed in array and
        # so we can just pull num_samples from the class vars
        num_samples = self.num_samples if (input_arr is None) else input_arr.shape[-1]
        init_mask = np.zeros((1, num_samples))
        # All samples are initialized to not be in any region. Then we iterate through all samples and assign
        # 1 to the region containing the sample while the rest retain the default 0.
        region_masks = defaultdict(init_mask.copy)
        for sample_idx in range(num_samples):
            if input_arr is None:
                sample = self.train_tensor[:, sample_idx]
                # print("Train tensor shape: %s" % list(self.train_tensor.shape))
            else:
                sample = input_arr[:, sample_idx]
                # print("Input tensor shape: %s" % list(input_arr.shape))
            for region_idx, region in enumerate(self.regions):
                if region.check_satisfaction(sample):
                    region_masks[region_idx][:, sample_idx] = 1
        # print([region_mask.shape for region_mask in self.region_masks.values()])
        if input_arr is not None:
            return region_masks
        # If dealing with the dataset then overwrite the class variable region_masks
        self.region_masks = region_masks

    def get_region_idxs(self, mask_input=None):
        # dict of numpy arrays. array at key i tells us which indices of train_x belong to region_i
        if mask_input is None:
            mask_input = self.region_masks
        regionwise_sample_idxs = {}
        for region_idx, region in mask_input.items():
            # nonzero pulls out those indices of the mask which have values 1 and hence correspond
            # to those samples belonging to this region_idx in the input_arr that generate the mask_input
            # array.
            regionwise_sample_idxs[region_idx] = np.nonzero(mask_input[region_idx].squeeze() > 0)
            # print(regionwise_sample_idxs[region_idx])
        return regionwise_sample_idxs

    def _gen_white_noise(self, num_samples=None, region_masks=None, noise_verbose=False, ax=None):
        """
        :param dims: Dim of input vector (= train_x.shape[0] Inputs can be over joint state and input space
        of the control system. But writing train_*x* is just to keep convention with normal notation for ML training inputs)
        :param num_samples: number of training samples generated. Number of white noise vectors generated must be equal to this.
        :param noise_vars: List of noise variances for each region
        :param region_masks: Dictionary of masks that map samples to regions that are within. Masks are output from the
        generate_regionspec_mask function
        """
        if num_samples is None:
            num_samples = self.num_samples
            region_masks = self.region_masks
        noise_samples = torch.zeros((self.output_dims, num_samples))
        colours = ['r', 'g', 'b']
        for region_idx, region_mask in region_masks.items():
            noise_var = self.noise_vars[region_idx]
            region_noise = ((torch.randn((self.output_dims, num_samples)) * math.sqrt(noise_var)) * region_mask)
            noise_samples += region_noise
            if ax is not None:
                ax.hist(region_noise, bins=16, color=colours[region_idx])
            # print(noise_var)
        if noise_verbose:
            print("Generated noise_sample")
            print(noise_samples)
        # print(noise_samples.shape)
        return noise_samples

    def generate_outputs(self, input_arr=None, no_noise=False, return_op=False, noise_verbose=False, noise_plot_ax=None):
        """
        :param train_x: set of train set inputs
        :param noise_vars: variance of the white iid noise. length = len(regions)
        :param funcs: list of callable (partial) funcs that match with the order of boundary locations
        :param regions: for piecewise function this element gives the boundary locs. Length must be equal to that of len(funcs)-1
        :return: outputs from global/piecewise model
        """
        # Will be none when generating noisy samples and not None when passing a fine grained equispaced set of points
        # in the state space to visualize the true function mean (note the true function also has a covariance because of the stochasticity estimates.
        if input_arr is None:
            input_arr = self.train_tensor
        if no_noise or return_op:
            mask = self._generate_regionspec_mask(input_arr=input_arr)
        else:
            self._generate_regionspec_mask()
            mask = self.region_masks

        # Ex: For 2D case with 2 regions we have f1->f4. The mask remains same for output of (f1, f2) and (f3, f4) since f1, f2 are always active in region 1
        # and f3, f4 are always active in region 2.
        segregated_ops = [func(input_arr) * mask[idx // self.input_dims] for idx, func in enumerate(self.callables)]
        # print(segregated_ops)
        # if no_noise:
        #     print((mask[0//self.dims]+mask[2//self.dims] > 0).all())
        concd_ops_dims = []
        for i in range(self.num_regions):
            concd_ops_dims.append(np.vstack(segregated_ops[i*self.output_dims: (i+1)*self.output_dims]))

        train_outputs = torch.sum(torch.from_numpy(np.stack(concd_ops_dims)), 0)
        if no_noise or return_op:
            if no_noise:
                return train_outputs
            else:
                noise_samples = self._gen_white_noise(num_samples=input_arr.shape[-1], region_masks=mask,
                                                      noise_verbose=noise_verbose, ax=noise_plot_ax)
                return train_outputs + noise_samples

        noise_samples = self._gen_white_noise()
        self.train_y = train_outputs + noise_samples

    def viz_outputs_1d(self, fineness_param=(51, ), ax1=None):
        if ax1 is None:
            f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        start_limit, end_limit = self.state_space_polytope.lb, self.state_space_polytope.ub
        fine_x = torch.linspace(start_limit.item(), end_limit.item(), fineness_param[0]*(end_limit-start_limit).item())
        true_y = self.generate_outputs(input_arr=self._convert_to_2d_tensor(fine_x), no_noise=True)
        ax1.plot(fine_x, true_y.squeeze(), 'b')
        ax1.scatter(self.train_x, self.train_y.squeeze(), color='r')
        ax1.legend(["True function", "Noisy samples"], fontsize=25)
        # ax1.set_title('True function w/ noisy samples', fontsize=25)

    def generate_fine_grid(self, fineness_param, with_mask=False):
        fine_grid = generate_fine_grid(self.state_space_polytope.lb, self.state_space_polytope.ub, fineness_param=fineness_param)
        if with_mask:
            mask = self._generate_regionspec_mask(input_arr=fine_grid)
            return fine_grid, mask
        else:
            return fine_grid

    @staticmethod
    def ret_regspec_samples(input_arrs, region_idx, regionwise_idxs_dict):
        return [input_arr[:, regionwise_idxs_dict[region_idx]] for input_arr in input_arrs]

    @staticmethod
    def get_colours(num_regions, cm_name='Spectral'):
        cmap = plt.cm.get_cmap('Spectral')
        intervals = list(np.linspace(0, 1, num_regions))
        colours = [cmap(intervals[i]) for i in range(len(intervals))]
        return colours

    def plot_true_func_2d(self, axes, fineness_param, true_plot=False, samples_plot=False, op_arr=None, colours=None):
        assert len(axes) == self.output_dims, "Number of axes must match the dimension of the state vector " \
                                             "(ex: 2-dim residual output state vectors has 2 separate functions governing the dynamics of the output)"
        fine_grid, mask = self.generate_fine_grid(fineness_param=fineness_param, with_mask=True)
        true_y = self.generate_outputs(input_arr=torch.from_numpy(fine_grid), no_noise=True)
        fine_regionwise_idxs = self.get_region_idxs(mask_input=mask)
        sample_regionwise_idxs = self.get_region_idxs()
        if colours is None:
            colours = self.get_colours(self.num_regions)
        if op_arr is None:
            op_arr = self.train_y
        colours = ['lightsalmon', 'cyan', 'limegreen']
        for ax_idx, ax in enumerate(axes):
            for region_idx in range(len(self.regions)):
                region_finex, region_fineop = self.ret_regspec_samples([fine_grid, true_y], region_idx, fine_regionwise_idxs)
                region_samples, op_scalars = self.ret_regspec_samples([self.train_x, op_arr], region_idx, sample_regionwise_idxs)
                if true_plot:
                    ax.scatter3D(region_finex[0, :], region_finex[1, :], region_fineop[ax_idx, :].squeeze(), color=colours[region_idx],
                                 label="Region %s" % (region_idx+1))
                if samples_plot:
                    ax.scatter3D(region_samples[0, :], region_samples[1, :],
                                 op_scalars[ax_idx, :].squeeze(), color=colours[-region_idx-1])
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.zaxis.set_tick_params(labelsize=20)
            ax.set_xlabel("State 1", fontsize=20, labelpad=20)
            ax.set_ylabel("State 2", fontsize=20, labelpad=20)
            ax.set_zlabel("Residual (g(x))", fontsize=20, labelpad=20)
            ax.legend(loc="upper center", fontsize=20)
        return fine_grid, fine_regionwise_idxs

    def viz_outputs_2d(self, fineness_param=(51,), true_only=False, ax=None):
        """
        Parameters
        ----------
        fineness_param Specifies how many samples are generated across the range of the input space. Higher param for
        the dimension => more unique values for that dimension get generated for the cartesian product.
        true_only Only plot the true function
        dim_limit, dim_idxs If true then the input space has more than 3 dims so to visualize we limit x-y plane of the plot
        to only be 2 selected dimensions of the input space. THe selected dimensions are specified in dim_idxs
        ax if passing in an axis object then one is not created.
        """
        assert self.input_dims == 2, "Dimension of the input space must be 2 to use this function"
        if ax is None:
            fig = plt.figure(figsize=(16, 32))
            axes = []
            for i in range(self.output_dims):
                ax = fig.add_subplot(self.output_dims, 1, i, projection='3d')
                axes.append(ax)
            # ax1, ax2 = fig.add_subplot(2, 1, 1, projection='3d'), fig.add_subplot(2, 1, 2, projection='3d')
            # axes = [ax1, ax2]
        else:
            axes = [ax]
        # fine_grid = self.generate_fine_grid(fineness_param=fineness_param)
        # print("Fine grid shape: %s" % list(fine_grid.shape))
        # true_y = self.generate_outputs(input_arr=fine_grid, no_noise=True)
        self.plot_true_func_2d(axes=axes, fineness_param=fineness_param, true_plot=True, samples_plot=not true_only)
        str_append = ""

        for ax_idx, ax in enumerate(axes):
            # ax.scatter3D(fine_grid[0, :], fine_grid[1, :], true_y[ax_idx, :].squeeze(), 'b')
            if not true_only:
                ax.legend(["True function", "Noisy samples"])
                str_append = "w/ noisy samples"
            # ax.set_title('True function for state x%s %s' % (ax_idx+1, str_append))


def sinusoid_func(input_arr: torch.Tensor, sine_multiplier=1.0, skip_torch=False):
    """
    :param sine_multiplier: multiplier for sine frequency
    """
    if skip_torch:
        return np.sin(input_arr * sine_multiplier)
    else:
        return torch.sin(torch.Tensor(input_arr) * sine_multiplier)


def clipped_exp_sns(input_arr, shift_param=0.0, scale_param=1.0, min_clip_param=None, max_clip_param=None, skip_torch=False):
    """
    :param shift_param: at what input value does exp value go to 1
    :param scale_param: scale the exponential
    :param clip_param: supply int if desired to clip exp to a certain max value.
    """
    if skip_torch:
        op_arr = scale_param*np.exp(input_arr - shift_param)
    else:
        op_arr = scale_param*torch.exp(torch.Tensor(input_arr) - shift_param)
    if min_clip_param is not None:
        if skip_torch:
            op_arr = np.minimum(np.maximum(op_arr, min_clip_param), max_clip_param)
        else:
            op_arr = torch.minimum(torch.maximum(op_arr, torch.tensor(min_clip_param)), torch.tensor(max_clip_param))
    return op_arr


class box_constraint:
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, lb=None, ub=None):
        """
        :param lb: dimwise list of lower bounds.
        :param ub: dimwise list of lower bounds.
        """
        self.lb = np.array(lb, ndmin=2)
        self.ub = np.array(ub, ndmin=2)
        self.dim = self.lb.shape[0]
        assert (self.lb < self.ub).all(), "Lower bounds must be greater than corresponding upper bound for any given dimension"
        self.setup_constraint_matrix()

    def __str__(self): return "Lower bound: %s, Upper bound: %s" % (self.lb, self.ub)

    def get_random_vectors(self, num_samples):
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        dim = self.lb.shape[0]
        # print(dim)
        self.H_np = np.vstack((-np.eye(dim), np.eye(dim)))
        self.H = torch.Tensor(self.H_np)
        # self.b = torch.Tensor(np.hstack((-self.lb, self.ub)))
        self.b_np = np.vstack((-self.lb, self.ub))
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b

    def check_satisfaction(self, sample):
        # If sample is within the polytope defined by the constraints return 1 else 0.
        # print(sample, np.array(sample, ndmin=2).T, self.sym_func(sample), self.b)
        return (self.sym_func(sample) <= 0).all()


class box_constraint_direct(box_constraint):
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, H_np, b_np):
        self.dim = H_np.shape[0] // 2
        self.H_np = H_np
        self.b_np = b_np
        self.retrieve_ub_lb()
        self.setup_constraint_matrix()

    def retrieve_ub_lb(self):
        lb = -self.b_np[:self.dim]
        ub = self.b_np[self.dim:]
        self.lb = np.array(lb, ndmin=2)
        self.ub = np.array(ub, ndmin=2)

    def get_random_vectors(self, num_samples):
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        self.H = torch.Tensor(self.H_np)
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b


def combine_box(box1, box2, verbose=False):
    box1_lb, box1_ub = box1.lb, box1.ub
    box2_lb, box2_ub = box2.lb, box2.ub
    new_lb, new_ub = np.vstack((box1_lb, box2_lb)), np.vstack((box1_ub, box2_ub))
    new_constraint = box_constraint(new_lb, new_ub)
    if verbose:
        print(new_constraint)
    return new_constraint


class infnorm_constraint(box_constraint):
    def __init__(self, bound, dim=1):
        assert bound > 0, "Bound value must be positive"
        assert dim >= 1, "Dimension of state vector must be at least 1"
        self.lb = np.array(-bound*dim)
        self.ub = np.array(bound*dim)
        self.setup_constraint_matrix()


def viz_test_2d():
    test_means = np.array([[2, 5.5, 3.5], [-0.4, 0.5, 0.25]])
    sampler_inst = DS_Sampler(dims=2, method="clustering", num_samples=50, min_max_range=[(0, 7), (-0.5, 0.75)],
                              means=test_means, sample_variances=[0.03, 0.003], cluster_weights=[3, 12, 9])
    rand_samples = sampler_inst.get_samples()
    sampler_inst.viz_2d_clst(rand_samples)


def test_1d_class_outputs(sine_multiplier=1, exp_mean=0, exp_scale=1, exp_min_clip=2.5, exp_max_clip=10,
                          start_limit=-2, end_limit=5, num_points=200,
                          boundary_locs=(2.5, ), noise_vars=(0.5, 0.02), add_dim=False) -> GP_DS:
    coarse_x = torch.linspace(start_limit, end_limit, num_points)
    if add_dim:
        coarse_x = coarse_x[None]
        print(coarse_x.shape)
    func1 = lambda x: sinusoid_func(x, sine_multiplier)
    func2 = lambda x: clipped_exp_sns(x, shift_param=exp_mean, scale_param=exp_scale, min_clip_param=exp_min_clip, max_clip_param=exp_max_clip)
    state_space_polytope = box_constraint(start_limit, end_limit)
    partitions = []
    for idx, boundary_loc in enumerate(boundary_locs):
        if idx == 0:
            partitions.append([start_limit, boundary_loc])
        if idx == len(boundary_locs)-1:
            partitions.append([boundary_loc, end_limit])
        else:
            partitions.append([boundary_loc, boundary_locs[idx+1]])
    # regions = box_constraint(start_limit, boundary_loc), box_constraint(2.5, 5)
    regions = [box_constraint(region_start_limit, region_end_limit) for (region_start_limit, region_end_limit) in partitions]
    dataset = GP_DS(train_x=coarse_x, callables=[func1, func2], state_space_polytope=state_space_polytope, regions=regions, noise_vars=noise_vars)
    dataset.generate_outputs()

    dataset.viz_outputs_1d()
    return dataset


def test_2d_class_outputs(sine_multiplier=1, exp_mean=0, exp_scale=1, exp_min_clip=-10, exp_max_clip=15,
                          start_limit=np.array([[-2, -0.1]]).T, end_limit=np.array([[5, 0.3]]).T, num_points=50,
                          noise_vars=(0.05, 0.02, 0.03, 0.04),
                          fineness_param=(11, 11), no_viz=False) -> GP_DS:
    sampler_inst = DS_Sampler(dims=2, method="uniform random", num_samples=num_points,
                              min_max_range=[(start_limit[0, :], end_limit[0, :]), (start_limit[1, :], end_limit[1, :])])
    coarse_x = sampler_inst.get_samples()
    func1 = lambda x: sinusoid_func(x[0, :], sine_multiplier) # To make the output be a scalar I limit only the 1st dimension to pass through the sine.
    func2 = lambda x: clipped_exp_sns(x[1, :], shift_param=exp_mean, scale_param=exp_scale, min_clip_param=exp_min_clip, max_clip_param=exp_max_clip)
    func3 = lambda x: x[0, :] * x[1, :] # x1*x2
    state_space_polytope = box_constraint(start_limit, end_limit)
    callables = [func1, func2, func1, func3, func1, func2, func1, func3]

    x1_bounds = np.array([[-2, 1], [1, 5]])
    x2_bounds = np.array([[-0.1, 0.1], [0.1, 0.3]])
    regions = []

    for delim in range(x2_bounds.shape[-1]):
        for delim2 in range(x1_bounds.shape[-1]):
            regions.append(box_constraint(lb=np.vstack([x1_bounds[delim2][0], x2_bounds[delim][0]]), ub=np.vstack([x1_bounds[delim2][1], x2_bounds[delim][1]])))

    dataset = GP_DS(train_x=coarse_x, callables=callables, state_space_polytope=state_space_polytope, regions=regions, noise_vars=noise_vars)
    dataset.generate_outputs()

    if not no_viz:
        dataset.viz_outputs_2d(fineness_param=fineness_param, true_only=False)

    return dataset


def test_2d_op_3d_inp_allfeats(regions,
                               start_limit=np.array([[-2, -4, -3]]).T, end_limit=np.array([[5, 3, 4]]).T, num_points=50,
                               noise_vars=(0.05, 0.02, 0.03), get_defaults=False) -> GP_DS:
    assert len(regions) == 3, "Must pass 3 regions for this example."
    # Since the sampler generates train_x samples, the dims arg is 3 here. Samples are generated uniform randomly.
    sampler_inst = DS_Sampler(dims=3, method="uniform random", num_samples=num_points,
                              min_max_range=[(start_limit[i, :], end_limit[i, :]) for i in range(3)])
    coarse_x = sampler_inst.get_samples()
    sine_mult_1, sine_mult_2 = 0.25, 0.4
    sine_scale_1, sine_scale_2 = 0.25, 0.3
    exp_mean_1, exp_scale_1, exp_min_clip_1, exp_max_clip_1 = 4.5, 1, -1.4, 1.2
    exp_mean_2, exp_scale_2, exp_min_clip_2, exp_max_clip_2 = 0.5, 1, -1.2, 1.6
    if get_defaults:
        return [sine_mult_1, sine_mult_2, sine_scale_1, sine_scale_2,
                exp_mean_1, exp_scale_1, exp_min_clip_1, exp_max_clip_1,
                exp_mean_2, exp_scale_2, exp_min_clip_2, exp_max_clip_2]
    r1_d1 = lambda x: sinusoid_func(x[0, :], sine_mult_1) * sine_scale_1
    r1_d2 = lambda x: clipped_exp_sns(x[1, :], shift_param=exp_mean_1, scale_param=exp_scale_1, min_clip_param=exp_min_clip_1, max_clip_param=exp_max_clip_1)
    r2_d1 = lambda x: torch.minimum(torch.maximum(x[2, :] * x[1, :], torch.tensor(0.25)) * 0.1, torch.tensor(1.75))
    r2_d2 = lambda x: sinusoid_func(x[2, :], sine_mult_2) * sine_scale_2
    r3_d1 = lambda x: clipped_exp_sns(x[2, :], shift_param=exp_mean_2, scale_param=exp_scale_2, min_clip_param=exp_min_clip_2, max_clip_param=exp_max_clip_2)
    r3_d2 = lambda x: torch.minimum(torch.maximum(x[0, :] * x[2, :], torch.tensor(0.45)) * 0.25, torch.tensor(1.75))
    state_space_polytope = box_constraint(start_limit, end_limit)
    callables = [r1_d1, r2_d1, r3_d1, r1_d2, r2_d2, r3_d2]

    dataset = GP_DS(train_x=coarse_x, callables=callables, state_space_polytope=state_space_polytope,
                    regions=regions, noise_vars=noise_vars, output_dims=2)
    dataset.generate_outputs()

    return dataset


def test_1d_op_2d_inp_allfeats(regions,
                               start_limit=np.array([[-2, -2]]).T, end_limit=np.array([[2, 2]]).T, num_points=50,
                               noise_vars=(0.05, 0.02, 0.03), get_defaults=False, fineness_param=(11, 11), no_viz=True, cluster_based=False) -> GP_DS:
    assert len(regions) == 3, "Must pass 3 regions for this example."
    # Since the sampler generates train_x samples, the dims arg is 2 here. Samples are generated uniform randomly.
    min_max_range = [(start_limit[i, :], end_limit[i, :]) for i in range(2)]
    num_means = 3
    if cluster_based:
        eps = 0.15
        sep_param = 1.414
        eps_vec = np.ones(shape=[*start_limit.shape]) * eps
        mean_gen_inst = box_constraint(lb=start_limit+eps_vec, ub=end_limit-eps_vec)
        means = np.zeros([start_limit.shape[0], num_means])
        means[:, [0]] = mean_gen_inst.get_random_vectors(num_samples=1)
        mean_idx = 1
        while mean_idx < num_means:
            temp_mean = mean_gen_inst.get_random_vectors(num_samples=1)
            failed = False
            for fixed_mean_idx in range(mean_idx):
                if np.linalg.norm(temp_mean - means[:, [mean_idx]]) < sep_param:
                    failed = True
            if not failed:
                means[:, [mean_idx]] = temp_mean
                mean_idx += 1

        sampler_inst = DS_Sampler(dims=2, method="clustering", num_samples=num_points, min_max_range=min_max_range,
                                  means=means, sample_variances=[0.3, 0.3], cluster_weights=[1]*num_means)
    else:
        sampler_inst = DS_Sampler(dims=2, method="uniform random", num_samples=num_points,
                                  min_max_range=min_max_range)
    coarse_x = sampler_inst.get_samples()
    sine_mult_1, sine_mult_2 = 0.25, 0.4
    sine_scale_1, sine_scale_2 = 0.25, 0.3
    exp_mean_1, exp_scale_1, exp_min_clip_1, exp_max_clip_1 = 4.5, 1, -1.4, 0.09
    min_clip, max_clip, clip_scale = 0.25, 5, 0.05
    if get_defaults:
        return [sine_mult_1, sine_scale_1, exp_mean_1, exp_scale_1, exp_min_clip_1, exp_max_clip_1, min_clip, max_clip, clip_scale]
    r1_d1 = lambda x: sinusoid_func(x[0, :], sine_mult_1) * sine_scale_1
    r2_d1 = lambda x: clipped_exp_sns(x[1, :], shift_param=exp_mean_1, scale_param=exp_scale_1, min_clip_param=exp_min_clip_1, max_clip_param=exp_max_clip_1)
    r3_d1 = lambda x: torch.minimum(torch.maximum(x[0, :] * x[1, :], torch.tensor(min_clip)), torch.tensor(max_clip)) * clip_scale
    state_space_polytope = box_constraint(start_limit, end_limit)
    callables = [r1_d1, r2_d1, r3_d1]

    dataset = GP_DS(train_x=coarse_x, callables=callables, state_space_polytope=state_space_polytope,
                    regions=regions, noise_vars=noise_vars, output_dims=1)
    dataset.generate_outputs()
    if not no_viz:
        dataset.viz_outputs_2d(fineness_param=fineness_param, true_only=False)

    return dataset


def test_1d_op_1d_inp_allfeats(regions,
                               start_limit, end_limit, num_points=50,
                               noise_vars=(0.05, 0.02), get_defaults=False) -> GP_DS:
    assert len(regions) == 2, "Must pass 2 regions for this example."
    # Since the sampler generates train_x samples, the dims arg is 2 here. Samples are generated uniform randomly.
    sampler_inst = DS_Sampler(dims=1, method="uniform random", num_samples=num_points,
                              min_max_range=[(start_limit[i, :], end_limit[i, :]) for i in range(1)])
    coarse_x = sampler_inst.get_samples()
    sine_mult_1, sine_scale_1 = 0.25, 0.25
    min_clip, max_clip, clip_scale = 0.25, 5, 0.1
    if get_defaults:
        return [sine_mult_1, sine_scale_1, min_clip, max_clip, clip_scale]
    r1_d1 = lambda x: sinusoid_func(x[0, :], sine_mult_1) * sine_scale_1
    r2_d1 = lambda x: torch.minimum(torch.maximum(x[0, :], torch.tensor(min_clip)), torch.tensor(max_clip)) * clip_scale
    state_space_polytope = box_constraint(start_limit, end_limit)
    callables = [r1_d1, r2_d1]

    dataset = GP_DS(train_x=coarse_x, callables=callables, state_space_polytope=state_space_polytope,
                    regions=regions, noise_vars=noise_vars, output_dims=1)
    dataset.generate_outputs()

    return dataset


def test_box_constraint():
    start_limit, end_limit = [[-5, -3]], [[3, 4]]
    test_constraint = box_constraint(np.array(start_limit).T, np.array(end_limit).T)
    pprint.pprint(test_constraint.H)
    pprint.pprint(test_constraint.b)
    # No constraints violated
    print("H @ (x=(2, 2)) : %s" % (test_constraint.H @ np.array([2, 2])))
    print(test_constraint.sym_func([2, 2]))
    # 1 constraint violated for x[0]
    print(test_constraint.sym_func([-6, 3]))
    # 1 constraint each violated for x[0], x[1]
    print(test_constraint.sym_func([-6, 5]))

    start_limit, end_limit = np.array([[-7, -5]]).T, np.array([[5, 6]]).T
    grid_check = generate_fine_grid(start_limit, end_limit, fineness_param=(5, 5), viz_grid=False)
    # print(grid_check.shape)
    mask = []
    # print((test_constraint.sym_func([-7, -5]) <= 0).all().item())
    # print(test_constraint.sym_func([-3, -2]))
    # print((test_constraint.sym_func([-3, -2]) <= 0).all().item())
    for grid_vec_idx in range(grid_check.shape[-1]):
        grid_vec = grid_check[:, grid_vec_idx]
        mask.append((test_constraint.sym_func(grid_vec) <= 0).all().item())
    passed_vecs = grid_check[:, mask]
    plt.figure()
    plt.scatter(passed_vecs[0], passed_vecs[1], c='r')

    rand_sampled_vecs = test_constraint.get_random_vectors(50)
    print(rand_sampled_vecs)
    for i in range(test_constraint.dim):
        print(rand_sampled_vecs[i, :].min(), rand_sampled_vecs[i, :].max())


