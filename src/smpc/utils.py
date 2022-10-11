import torch
import numpy as np
import casadi as cs
import scipy
import math
from ds_utils import box_constraint, sinusoid_func, clipped_exp_sns
from IPython.display import display, Math
import os
# dill needed to pickle lambdas
import dill as pkl

def covSEard(x,
             z,
             ell,
             sf2
             ):
    """GP squared exponential kernel.

    This function is based on the 2018 GP-MPC library by Helge-André Langåker

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        SE kernel (casadi.MX/SX): SE kernel.

    """
    dist = cs.sum1((x - z)**2 / ell**2)
    return sf2 * cs.SX.exp(-.5 * dist)


# Taken from https://gist.github.com/KMChris/8fd878826453c3d55814b3293c7b084c
def np2bmatrix(arrays, return_list=False):
    matrices = ''
    if return_list:
        matrices = []
    for array in arrays:
        matrix = ''
        temp_arr = np.round(array, 5)
        for row in temp_arr:
            try:
                for number in row:
                    matrix += f'{number}&'
            except TypeError:
                matrix += f'{row}&'
            matrix = matrix[:-1] + r'\\'
        if not return_list:
            matrices += r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'
        else:
            matrices.append(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}')
    return matrices


class Sigma_u_Callback(cs.Callback):
    def __init__(self, name, K, opts={"enable_fd": True}):
        cs.Callback.__init__(self)
        self.K = K
        self.n_u, self.n_x = self.K.shape # K is an mxn matrix since BKx is in R^n, B in R^(nxm), K in R^(mxn) and x in R^n
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cs.Sparsity.dense(self.n_x, self.n_x)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_u, self.n_u)

    def eval(self, arg):
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L308
        # Sigma_u = self.K.T @ arg[0] @ self.K
        Sigma_u = self.K @ arg[0] @ self.K.T
        return [Sigma_u]


class Sigma_x_dynamics_Callback_LTI(cs.Callback):
    def __init__(self, name, affine_transform, n_in, n_x, opts={"enable_fd": True}):
        cs.Callback.__init__(self)
        self.affine_transform = affine_transform
        self.n_in = n_in
        self.n_x = n_x
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cs.Sparsity.dense(self.n_in, self.n_in)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_x, self.n_x)

    def eval(self, arg):
        Sigma_k = arg[0]
        Sigma_x = self.affine_transform @ Sigma_k @ self.affine_transform.T
        return [Sigma_x]


class GPR_Callback(cs.Callback):
    def __init__(self, name, likelihood_fn, model, state_dim=1, output_dim=None, opts={}):
        """
        Parameters
        ----------
        name Name is necessary for Casadi initialization using construct.
        likelihood_fn
        model
        state_dim size of the input dimension.
        opts
        """
        cs.Callback.__init__(self)
        self.likelihood = likelihood_fn
        self.model = model
        self.input_dims = state_dim
        self.output_dims = output_dim
        if self.output_dims is None:
            self.output_dims = self.input_dims
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 2

    def get_num_samples(self):
        return self.model.train_x.shape[0]

    def __len__(self): return 1

    def get_sparsity_in(self, i):
        # If this isn't specified then it doesn't accept the full input vector and only limits itself to the first element.
        # return cs.Sparsity.dense(self.state_dim, 1)
        return cs.Sparsity.dense(1, self.input_dims)

    def eval(self, arg):
        # likelihood will return a mean and variance but out differentiator only needs the mean
        # print(arg[0])
        mean, cov = self.postproc(self.likelihood(self.model(self.preproc(arg[0]))))
        return [mean, cov]

    @staticmethod
    def preproc(inp):
        return torch.from_numpy(np.array(inp).astype(np.float32))

    @staticmethod
    def postproc(op):
        # print(get_user_attributes(op))
        return op.mean.detach().numpy(), op.covariance_matrix.detach().numpy()


def setup_terminal_costs(A, B, Q, R):
    Q_lqr = Q
    R_lqr = R
    P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    K = -np.dot(np.linalg.inv(R + np.dot(btp, B)), np.dot(btp, A))
    return K, P


def get_inv_cdf(n_i, satisfaction_prob):
    # \overline{p} from the paper
    p_bar_i = 1 - (1 / n_i - (satisfaction_prob + 1) / (2 * n_i))
    # \phi^-1(\overline{p})
    inverse_cdf_i = scipy.stats.norm.ppf(p_bar_i)
    return inverse_cdf_i


class Piecewise_GPR_Callback(GPR_Callback):
    def __init__(self, name, likelihood_fns, models, output_dim, input_dim, num_regions, opts={}):
        """

        Parameters
        ----------
        name
        likelihood_fns, models: For the multidim piecewise case the ordering would be all models belonging to one state dim
        first before moving to the next. Ex: for the 2-D case with 4 regions we have 8 models [model_0 ... model_7] where
        model_0-model_3 correspond to the 4 piecewise models for the first state dim and model_4-model_7 correspond to those for the
        second state dim
        output_dim: Dimension of the GP output. This must match with the number of models passed obeying the formula
        output_dim*num_regions = len(models)
        input_dim: Dimension of the GP input. Must match with the input dimension of each GP.
        num_regions: number of regions in the piecewise/hybrid model
        opts

        Returns
        Note that this function only returns a (horizontal) concatenation of the means output from each GP. The
        application of delta to select the right mean is left to a Casadi function defined in the piecewise MPC class.
        """
        cs.Callback.__init__(self)
        assert output_dim*num_regions == len(models), "The models must have length = output_dim*num_regions = %s but got %s instead" %\
                                                      (output_dim*num_regions, len(models))
        for model in models:
            assert input_dim == model.train_x.shape[-1], "The value of input_dim must match the number of columns of the model's train_x set. input_dim: %s, train_x_shape: %s" % (input_dim, model.train_x.shape)
        self.likelihoods = likelihood_fns
        self.models = models
        self.output_dims = output_dim
        self.input_dims = input_dim
        self.num_models = len(self.models)
        self.num_regions = num_regions
        self.construct(name, opts)
        self.organize_models()

    def get_n_in(self): return 1
    # Can't return covariances as list or 3-D tensor-like array. Instead return all cov matrices separately. There are
    # num_regions number of cov matrices to return.
    def get_n_out(self): return 1+self.num_regions

    def get_num_samples(self):
        # Only need to sum over 1 dimension of the output since the samples must be summed regionwise.
        return np.sum([self.models[idx].train_x.shape[0] for idx in range(len(self.models)//self.output_dims)])

    def get_sparsity_out(self, i):
        if i == 0:
            # Output mean shape is (self.output_dims, 1) and 1 for every region stacked horizontally to get the below
            return cs.Sparsity.dense(self.output_dims, self.num_regions)
        else:
            # output residual covariance matrices. One of these for every region for a total of self.num_regions covariance
            # matrices as evidenced by get_n_out
            return cs.Sparsity.dense(self.output_dims, self.output_dims)

    def __len__(self): return self.num_models

    def organize_models(self):
        # self.dimwise_region_models, self.dimwise_region_likelihoods = [[] for _ in range(self.num_regions)], [[] for _ in range(self.num_regions)]
        self.dimwise_region_models, self.dimwise_region_likelihoods = [[] for _ in range(self.output_dims)], [[] for _ in range(self.output_dims)]
        # Partition models per dimension. Ex: For 2-D case with 4 regions dimwise_models[0] has models[0]->models[3]
        # and dimwise_models[1] has models[4]->models[7]
        for output_dim in range(self.output_dims):
            self.dimwise_region_models[output_dim] = self.models[output_dim*(self.num_regions):
                                                                 (output_dim+1)*(self.num_regions)]
            self.dimwise_region_likelihoods[output_dim] = self.likelihoods[output_dim*(self.num_regions):
                                                                           (output_dim+1)*(self.num_regions)]


    def eval(self, arg):
        # Note regarding the covariances.
        # Regionwise_covs is going to be a list of single values corresponding to the covariance output in 1 region
        # of a single output GP. In the same way as the MultidimGPR callback, the covariance outputs from the same region
        # must be stored together in a diag matrix. Thus, the final covs list is going be a list of length = num_regions
        # with each element being a *diagonal* (because of independence across dims assumption in residual terms) matrix
        # of size (n_d, n_d)
        dimwise_means, dimwise_covs = [], []
        for output_dim in range(self.output_dims):
            regionwise_means, regionwise_covs = [], []
            dim_likelihoods, dim_models = self.dimwise_region_likelihoods[output_dim], self.dimwise_region_models[output_dim]
            for likelihood, model in zip(dim_likelihoods, dim_models):
                gp_op = self.postproc(likelihood(model(self.preproc(arg[0]))))
                regionwise_means.append(gp_op[0])
                regionwise_covs.append(gp_op[1])
            # Note the gp output mean and covariances are of the shape (1,) (1,). When calling horzcat on these shapes, casadi ends up
            # up vertstacking them instead. So just use horzcat to generate a vector of shape (num_regions, 1) and then transpose. to
            # get the desired row vector instead of column vector.
            dimwise_means.append(cs.horzcat(regionwise_means).T)
            dimwise_covs.append(regionwise_covs)
        covs = [cs.diag([dim_cov[region_idx] for dim_cov in dimwise_covs]) for region_idx in range(self.num_regions)]
        means = cs.vertcat(*dimwise_means)
        return [means, *covs]


class hybrid_res_covar(cs.Callback):
    def __init__(self, name, n_d, num_regions, N, opts={}, delta_tol=0, test_softplus=False):
        cs.Callback.__init__(self)
        self.n_d = n_d
        self.num_regions = num_regions
        self.N = N
        self.delta_tol = delta_tol
        self.test_softplus = test_softplus
        self.sharpness_param = 75
        self.construct(name, opts)

    def get_n_in(self):
        # 1 for delta array followed by num_regions number of covariance mats
        return 1+self.num_regions
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.num_regions, 1)
        else:
            return cs.Sparsity.dense(self.n_d, self.n_d)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_d, self.n_d)

    def eval(self, arg):
        delta_k, Sigma_d_arr = arg[0], arg[1:]
        Sigma_d = cs.DM.zeros(self.n_d, self.n_d)
        for region_idx in range(len(Sigma_d_arr)):
            if not self.test_softplus:
                delta_vec = delta_k[region_idx, 0]
            else:
                delta_vec = np.log(1+np.exp(self.sharpness_param*delta_k[region_idx, 0]))/self.sharpness_param
            Sigma_d += Sigma_d_arr[region_idx] * (delta_vec + self.delta_tol)
        return [Sigma_d]


class MultidimGPR_Callback(GPR_Callback):
    def __init__(self, name, likelihood_fns, models, state_dim=1, output_dim=None, opts={}):
        cs.Callback.__init__(self)
        self.likelihoods = likelihood_fns
        self.models = models
        self.n_d = len(self.models)
        self.state_dim = state_dim
        self.input_dims = self.state_dim
        self.output_dim = output_dim
        self.output_dims = output_dim
        if self.output_dim is None:
            self.output_dim = self.state_dim
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 2

    def __len__(self): return self.n_d

    def get_num_samples(self):
        # Number of samples is constant across all models for the multidimensional case unlike the piecewise one.
        return self.models[0].train_x.shape[0]

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.output_dim, 1)
        elif i == 1:
            return cs.Sparsity.dense(self.output_dim, self.output_dim)

    def eval(self, arg):
        means, covs = [], []
        for likelihood, model in zip(self.likelihoods, self.models):
            gp_op = self.postproc(likelihood(model(self.preproc(arg[0]))))
            means.append(gp_op[0])
            covs.append(gp_op[1])
        # Simplifying assumption that the residuals output in each dimension are independent of others and hence off-diagonal elements are 0.
        return [cs.vertcat(means), cs.diag(covs)]


class computeSigma_meaneq(cs.Callback):
    # flag for taylor approx
    def __init__(self, name, feedback_mat, residual_dim, opts={}):
        cs.Callback.__init__(self)
        self.K = feedback_mat
        self.n_u, self.n_x = self.K.shape
        self.n_d = residual_dim
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 3
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.n_x, self.n_x)
        elif i == 1:
            return cs.Sparsity.dense(self.n_u, self.n_u)
        elif i == 2:
            return cs.Sparsity.dense(self.n_d, self.n_d)

    # This method is needed to specify Sigma's output shape matrix (as seen from casadi's callback.py example). Without it,
    # the symbolic output Sigma is treated as (1, 1) instead of (mat_dim, mat_dim)
    def get_sparsity_out(self, i):
        # Forward sensitivity
        mat_dim = self.n_x+self.n_u+self.n_d
        return cs.Sparsity.dense(mat_dim, mat_dim)

    def eval(self, arg):
        Sigma_x, Sigma_u, Sigma_d = arg[0], arg[1], arg[2]
        # print(Sigma_x.shape, Sigma_u.shape, Sigma_d.shape)
        assert Sigma_d.shape == (self.n_d, self.n_d), "Shape of Sigma_d must match with n_d value specified when creating instance"
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L310
        Sigma_xu = Sigma_x @ self.K.T
        # Sigma_xu = Sigma_x @ self.K
        # Sigma_zd is specific to mean equivalence
        Sigma_xd = np.zeros((self.n_x, self.n_d))
        Sigma_ud = np.zeros((self.n_u, self.n_d))

        Sigma_z = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu),
                             cs.horzcat(Sigma_xu.T, Sigma_u)
                            )

        Sigma = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu, Sigma_xd),
                           cs.horzcat(Sigma_xu.T, Sigma_u, Sigma_ud),
                           cs.horzcat(Sigma_xd.T, Sigma_ud.T, Sigma_d))

        return [Sigma]


def simulate_hld_lld(X_test, regions, state_dim=2, eps=1e-5,
                     samples=np.array([[-2, -3]]).T, num_samples=1, verbose=True, ret_lld=False, unsqueeze=False):
    num_samples = samples.shape[-1]
    delta_r_k = np.zeros((2*state_dim, len(regions), num_samples))

    if verbose:
        display(Math(r'\text{}\,\,: {}{}<={}'.format("{State constraints}",
                                                     np2bmatrix([X_test.H_np]),
                                                     np2bmatrix([samples]),
                                                     np2bmatrix([X_test.b_np]))))

    for region_idx, region in enumerate(regions):
        region_H, region_b = region.H_np, region.b_np
        if verbose:
            display(Math(r'\text{}\,\,{}\,\,\text{}: {}{}<={}'.format("{Region}", region_idx+1, "{constraints}",
                                                                      np2bmatrix([region_H]),
                                                                      np2bmatrix([samples]),
                                                                      np2bmatrix([region_b]))))
        dim_idx = 0
        for inequality_idx in range(region_H.shape[0]):
            if verbose:
                print("Inequality %s" % inequality_idx)
            b = region_b[inequality_idx, :]
            # First half of inequalities correspond to lower bounds
            if inequality_idx < region_H.shape[0]//2:
                m = -(X_test.ub[dim_idx, :] + b)
                M = -(X_test.lb[dim_idx, :] + b)
            # Second half of inequalities correspond to upper bounds
            else:
                m = X_test.lb[dim_idx, :] - b
                M = X_test.ub[dim_idx, :] - b
            if verbose:
                print("delta_1")
                print(-X_test.ub[dim_idx, :], -b, m)
                display(Math(r'{} <= {} {} <= {}'.format(np2bmatrix([b+m]), np2bmatrix([region_H[[inequality_idx], :]]),
                                                         np2bmatrix([samples]), np2bmatrix([b]))))
                print("delta_0")
                print(-X_test.lb[dim_idx, :], b, M)
                display(Math(r'{} <= {} {} <= {}'.format(np2bmatrix([(b+eps)]), np2bmatrix([region_H[[inequality_idx], :]]),
                                                         np2bmatrix([samples]), np2bmatrix([b+M]))))
            ineq_row = region_H[inequality_idx, :]
            if unsqueeze:
                ineq_row = region_H[[inequality_idx], :]
            delta_1_bool = ((b+m) <= (ineq_row @ samples) <= b)
            delta_0_bool = ((b+eps) <= (ineq_row @ samples) <= (b+M))
            delta_assgt = 0 if delta_0_bool else 1
            if verbose:
                print("Delta Assignment: %s" % delta_assgt)
            assert delta_1_bool != delta_0_bool, "delta0 = delta1"
            delta_r_k[inequality_idx, region_idx, :] = delta_assgt
            # Circular rotation of dim_idx going from 0->state_dim for the lbs and then repeating from 0 for the ubs
            dim_idx = (dim_idx+1) % state_dim

    if ret_lld:
        # This function is called for warmstarting. When warmstarting, we call it using 1 sample only. and override a 2-D lld DM array. Thus
        # we can squeeze to remove the unnecessary 3rd dimension which casadi can't handle for opt vars/params anyway.
        return delta_r_k.squeeze()

    valid = 1
    for inequality_idx in range(regions[0].H_np.shape[0]):
        valid = cs.logic_and(valid, delta_r_k[inequality_idx, 0, 0])
    print("Region mask from deltas: %s" % False if not valid else True)
