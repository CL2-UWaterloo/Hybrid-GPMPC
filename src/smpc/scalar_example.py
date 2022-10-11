import copy

import executing.executing
import numpy as np
import scipy
import torch
from scipy import stats
import casadi as cs
import datetime as dt
import sys
import traceback
import matplotlib.pyplot as plt

import ds_utils
from ds_utils import box_constraint, box_constraint_direct, test_1d_op_1d_inp_allfeats, combine_box, generate_fine_grid, GP_DS
from .utils import Piecewise_GPR_Callback, MultidimGPR_Callback, GPR_Callback, Sigma_u_Callback, Sigma_x_dynamics_Callback_LTI
from .utils import hybrid_res_covar, computeSigma_meaneq, \
    np2bmatrix, simulate_hld_lld
from models import piecewise_train_test, train_test
from .controller_debugging import OptiDebugger, retrieve_controller_results_piecewise, retrieve_controller_results

from IPython.display import display, Math


# Hardcoded for sigma_x and sigma_u not to be opt vars
class Hybrid_GPMPC:
    def __init__(self, A, B, Q, R, horizon_length, gp_fns,
                 X: box_constraint, U: box_constraint, satisfaction_prob,
                 regions=None, solver='bonmin', piecewise=True,
                 skip_shrinking=False, addn_solver_opts=None, sqrt_const=1e-4,
                 add_b_shrunk_opt=False, add_b_shrunk_param=False,
                 ignore_cost=False, ignore_variance_costs=False, relaxed=False,
                 add_delta_tol=False, test_softplus=False):
        """
        :param A, B: Assumption is that the system is LTI so A, B are the matrices from the nominal dynamics equation dx/dt=Ax+Bu
        :param Q, R: Cost matrices for LQR
        :param horizon_length: O.L. opt horizon for MPC
        :param gp_fns: List of gp_fns. Length = n_d i.e. the size of the output vector of the residual since we train a separate
        gp for each residual dimension on the assumption of independence.
        :param X, U: Constraint polytopes for X and U respectively
        :param satisfaction_prob: Probability with which the chance constraints need to be satisfied. Required for constraint set shrinking
        :param skip_shrinking: Skip constraint set shrinking i.e. ignore chance constraints
        :param addn_solver_opts: Solver opts in addition to max iterations and enabling forward diff
        :add_b_shrunk_opt: adds the b_shrunk_{x, u} vectors to the list of optimization variables. Useful for testing where we have a
        small residual where we can generate a random feasible trajectory with ease and then check constraint violation on the b_shrunk vectors
        to ensure that chaining of the covariance computations is happening fine.
        add_b_shrunk_param: This technique is useful for the case where we don't want the optimizer to have to deal with shrinking at every iteration
        of the solver. Rather, the shrunk vectors are generated from the previous timestep's O.L. trajectories and the hope is that it will remain approximately
        true since the covariance kernel is smooth causing the uncertainty to not vary too much between trajectories where the joint state-input vector
        is close between multiple O.L. opts.
        """
        self.A, self.B, self.Q, self.R = [np.array(x, ndmin=2) for x in [A, B, Q, R]]
        # Residual only adds to 0th dim of state vector
        self.Bd = np.array([[1]])
        self.N = horizon_length
        self.n_x, self.n_u, self.n_d = 1, 1, 1
        # Feedback matrix
        self.K = self._setup_terminal()
        self.X, self.U = X, U

        self.sqrt_constant = sqrt_const
        self.ignore_cost = ignore_cost
        self.ignore_variance_costs = ignore_variance_costs
        self.add_delta_tol = add_delta_tol
        self.test_softplus = test_softplus

        self.gp_approx = 'mean_eq'
        self.gp_inputs = 'state_input'
        # 2-D state 1-D input. Gp input is 2nd state and input
        self.input_mask = np.array([[1, 0]])
        self.delta_control_variables = 'state_input'
        # Both state variables control the region
        self.delta_input_mask = np.array([[1, 0]])
        self.regions = regions

        self.piecewise = piecewise
        self.relaxed = relaxed
        if self.piecewise:
            self.construct_delta_constraint(X=self.X, U=self.U)
            if not self.relaxed:
                assert solver == 'bonmin', 'Current support for piecewise only for bonmin'
                self.solver = 'bonmin'
            else:
                self.solver = 'ipopt'
            assert self.regions is not None, "You must pass a list of regions when dealing with a piecewise problem"
            self.num_regions = len(self.regions)
        else:
            assert solver == 'ipopt', "Are you sure you want a solver other than ipopt?"
            self.solver = solver

        self.satisfaction_prob = satisfaction_prob
        self.gp_fns = gp_fns
        self.inverse_cdf_x, self.inverse_cdf_u = self.get_inv_cdf(self.n_x), self.get_inv_cdf(self.n_u)

        self.add_b_shrunk_opt = add_b_shrunk_opt
        self.add_b_shrunk_param = add_b_shrunk_param
        assert (not (self.add_b_shrunk_opt and self.add_b_shrunk_param)), "b_shrunk can't be opt (debug) and param (real-time) at the same time"

        self.system_type = 'lti'
        self.affine_transform = np.concatenate((self.A, self.B, self.Bd), axis=1)
        self.sigma_x_opt = False
        self.test_Sigma_u_opti = False

        base_opts = {"enable_fd": True}
        self.solver_opts = base_opts
        if addn_solver_opts is not None:
            self.solver_opts.update(addn_solver_opts)

        self.skip_shrinking = skip_shrinking

    def construct_delta_constraint(self, X, U=None):
        constraint_obj = X
        # If U is passed -> delta controls is joint state and input.
        if U is not None:
            constraint_obj = combine_box(X, U, verbose=False)
        masked_lb, masked_ub = self.delta_input_mask @ constraint_obj.lb, self.delta_input_mask @ constraint_obj.ub
        self.delta_constraint_obj = box_constraint(masked_lb, masked_ub)
        self.big_M = np.abs(self.delta_constraint_obj.b_np)*4

    def get_inv_cdf(self, n_i):
        # \overline{p} from the paper
        p_bar_i = 1 - (1 / n_i - (self.satisfaction_prob + 1) / (2 * n_i))
        # \phi^-1(\overline{p})
        inverse_cdf_i = scipy.stats.norm.ppf(p_bar_i)
        return inverse_cdf_i

    def _setup_terminal(self):
        # As in the paper, choose Q and R matrices for the LQR solution to be the same matrices
        # in the cost function optimization
        Q_lqr = self.Q
        R_lqr = self.R
        self.P = scipy.linalg.solve_discrete_are(self.A, self.B, Q_lqr, R_lqr)
        btp = np.dot(self.B.T, self.P)
        K = -np.dot(np.linalg.inv(self.R + np.dot(btp, self.B)), np.dot(btp, self.A))
        return K

    def cost_fn(self, mu_i_x, mu_i_u, Sigma_x_i, Sigma_u_i, x_desired, u_desired, idx=-1, terminal=False):
        if not terminal:
            # mu_i_u for now just assumes we want to drive the system to stable/equilibrium state.
            x_des_dev, u_des_dev = (mu_i_x - x_desired[:, idx]), (mu_i_u - u_desired[:, idx])
            # Mahalanobis/weighted 2 norm for x, u.
            mu_i_x_cost = x_des_dev.T @ self.Q @ x_des_dev
            mu_i_u_cost = u_des_dev.T @ self.R @ u_des_dev
            var_x_cost, var_u_cost = 0, 0
            if not self.ignore_variance_costs:
                var_x_cost = cs.trace(self.Q @ Sigma_x_i)
                var_u_cost = cs.trace(self.R @ Sigma_u_i)
            return (mu_i_x_cost + mu_i_u_cost + var_x_cost + var_u_cost)
        else:
            x_des_dev = (mu_i_x - x_desired[:, -1])
            mu_i_x_cost = x_des_dev.T @ self.P @ x_des_dev
            var_x_cost = 0
            if not self.ignore_variance_costs:
                var_x_cost = cs.trace(self.P @ Sigma_x_i)
            return (mu_i_x_cost + var_x_cost)

    def get_opti_list(self):
        print(self.opti.x)

    def get_info_for_traj_gen(self):
        return self.A, self.B, self.Bd, self.gp_fns, self.gp_inputs, self.input_mask, self.N

    def print_residual_minmax_info(self):
        if hasattr(self.gp_fns, "models"):  # Ignored for the Test case that doesn't have a model
            if self.piecewise:
                self.gp_fns: Piecewise_GPR_Callback
                for dim_idx in range(self.gp_fns.output_dims):
                    dim_region_models = self.gp_fns.dimwise_region_models[dim_idx]
                    for region_idx in range(self.num_regions):
                        print("Dim: %s ; Region: %s" % (dim_idx+1, region_idx+1))
                        model = dim_region_models[region_idx]
                        max = model.train_y.max()
                        min = model.train_y.min()
                        print("GP residual max and min: \n Max: %s \n Min: %s" % (max, min))
            else:
                maxs, mins = [], []
                try:
                    for model in self.gp_fns.models:
                        maxs.append(model.train_y.max())
                        mins.append(model.train_y.min())
                except AttributeError:
                    maxs.append(self.gp_fns.model.train_y.max())
                    mins.append(self.gp_fns.model.train_y.min())
                print("GP residual max and min: \n Max: %s \n Min: %s" % (np.vstack(maxs), np.vstack(mins)))
        else:
            if hasattr(self.gp_fns, "print_max_info"):
                self.gp_fns.print_max_info()

    def get_opti_info(self):
        print(dir(self.opti))
        print("opti.x")
        print(self.opti.x)
        print(self.opti.x.shape)
        print("opti.g")
        print(self.opti.g)
        print(self.opti.g.shape)
        print("opti instance")
        print(self.opti)

    def get_attrs_from_dict(self, vals):
        return [self.opti_dict[val] for val in vals]

    def set_initial(self, **kwargs):
        # Set parameter values for initial state and desired state and input trajectory.
        self.opti.set_value(self.opti_dict["x_init"], kwargs.get('x_init'))
        self.opti.set_value(self.opti_dict["x_desired"], kwargs.get('x_desired', np.zeros([self.n_x, self.N + 1])))
        self.opti.set_value(self.opti_dict["u_desired"], kwargs.get('u_desired', np.zeros([self.n_u, self.N])))
        # If param is not set, set the full shrunk array to be replicas of the original constraint b vectors which just corresponds to
        # no shrinking.
        if self.add_b_shrunk_param:
            self.opti.set_value(self.opti_dict["b_shrunk_x"][:, 1:], kwargs.get('b_shrunk_x', np.ones([2*self.n_x, self.N + 1])*self.X.b_np)[:, 1:])
            self.opti.set_value(self.opti_dict["b_shrunk_u"][:, 1:], kwargs.get('b_shrunk_u', np.ones([2*self.n_u, self.N])*self.U.b_np)[:, 1:])
            print("Setting shrunk vector parameters")
            b_shrunk_x_init, b_shrunk_u_init = kwargs.get('b_shrunk_x'), kwargs.get('b_shrunk_u')
            print("b_shrunk_x: \n%s \n b_shrunk_u: \n%s \n" % (b_shrunk_x_init, b_shrunk_u_init))

        # Warmstarting opt vars
        if kwargs.get('mu_x', None) is not None:
            print("Warmstarting with feasible trajectory (mu_x, mu_u%s)" % (', hld' if self.piecewise else ''))
            mu_x_init, mu_u_init = kwargs.get('mu_x'), kwargs.get('mu_u')
            print("mu_x: \n%s \n mu_u: \n%s \n" % (mu_x_init, mu_u_init))
            self.opti.set_initial(self.opti_dict["mu_x"], mu_x_init)
            self.opti.set_initial(self.opti_dict["mu_u"], mu_u_init)
            if self.piecewise:
                hld_init = kwargs.get('hld')
                print("hld: \n%s \n" % (hld_init))
                self.opti.set_initial(self.opti_dict["hld"], hld_init)
            if self.add_b_shrunk_opt:
                print("Warmstarting with feasible trajectory (b_shrunk_x, b_shrunk_u)")
                b_shrunk_x_init, b_shrunk_u_init = kwargs.get('b_shrunk_x'), kwargs.get('b_shrunk_u')
                print("b_shrunk_x: \n%s \n b_shrunk_u: \n%s \n" % (b_shrunk_x_init, b_shrunk_u_init))
                self.opti.set_initial(self.opti_dict["b_shrunk_x"][:, 1:], b_shrunk_x_init[:, 1:])
                self.opti.set_initial(self.opti_dict["b_shrunk_u"][:, 1:], b_shrunk_u_init[:, 1:])

    def display_configs(self):
        print("Time of running test: %s" % (dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        print("Nominal system matrices")
        print("A: %s \n B: %s \n Q: %s\n R: %s\n N: %s \n" % (self.A, self.B, self.Q, self.R, self.N))
        print("Constraints")
        print("X: lb: %s ub: %s \n U: lb: %s ub: %s" % (self.X.lb, self.X.ub, self.U.lb, self.U.ub))
        print("Chance constraint satisfaction probability (p): %s" % (self.satisfaction_prob))
        print("GP Info")
        print("DS Size: %s \n GP Inputs: %s \n Bd: %s \n input_mask: %s \n" % (
        self.gp_fns.get_num_samples(), ' '.join(self.gp_inputs.split("_")).title(),
        self.Bd, self.input_mask))
        # This function has been implemented for the global case in this class. Needs to be overridden in the Piecewise case.
        self.print_residual_minmax_info()

        print("Config options")
        print("Solver opts: %s" % self.solver_opts)
        print("Sigma_u_opti: %s \n Sigma_x_opti: %s \n Sqrt Const: %s \n Skip Shrinking: %s" %
              (self.test_Sigma_u_opti, self.sigma_x_opt, self.sqrt_constant, self.skip_shrinking))

        if self.piecewise:
            print("Piece-wise info")
            print("delta input type: %s" % self.delta_control_variables)
            print("delta mask: %s" % self.delta_input_mask)
            print("delta constraint object bounds: ")
            print(self.delta_constraint_obj)

    def sigma_x_dynamics_lti(self, k, Sigma_x, Sigma_u_k, Sigma, residual_cov_mat):
        Sigma_k = self.computesigma_wrapped(Sigma_x[k], Sigma_u_k, residual_cov_mat)
        Sigma.append(Sigma_k)
        Sigma_x[k+1] = self.compute_sigx_callback(Sigma_k)

    def get_sigma_u_k(self, k, Sigma_x, Sigma_u):
        # When sigma_x_opt is False too, the parent class adds the 0 mat to Sigma_u. The rest of the Sigma_u
        # matrices over the horizon are populated after using the Sigma_x callback for the next timestep. So no need
        # to do anything for k=0. But for k >= 1, we use the Sigma^x_{k+1} computed at the previous timestep using
        # the compute_sigx_callback attr to compute Sigma_u and then append that to the list.
        if k == 0:
            Sigma_u_k = Sigma_u[k]
        else:
            Sigma_u_k = self.compute_sigma_u(Sigma_x[k])
        return Sigma_u_k

    def solve_optimization(self, ignore_initial=False, **kwargs):
        if not ignore_initial:
            self.set_initial(**kwargs)
        sol = self.opti.solve()
        return sol

    def init_cov_arrays(self):
        residual_covs = []
        # Because Sigma_x is not opt var.
        Sigma_x = [cs.MX.zeros((self.n_x, self.n_x)) for _ in range(self.N+1)]
        Sigma_x[0] = cs.MX.zeros((self.n_x, self.n_x))
        # Sigma_u needs to be an optimization variable because it is used to compute shrunk sets and casadi doesn't accept a raw MX variable in it's place.
        # Alternate would be to define Sigma_u as a callback function that takes in Sigma_x which is what we do here.
        Sigma_u = []  # Will be populated in for loop.
        Sigma_u_0 = cs.MX.zeros((self.n_u, self.n_u))
        Sigma_u.append(Sigma_u_0)
        # Joint cov matrix
        Sigma = []
        return residual_covs, Sigma_x, Sigma_u, Sigma

    def setup_OL_optimization(self):
        """
        Sets up convex optimization problem.
        Including cost objective, variable bounds and dynamics constraints.
        """
        opti = cs.Opti()
        self.opti = opti

        # Problem parameters
        x_init = opti.parameter(self.n_x, 1)
        x_desired = opti.parameter(self.n_x, self.N + 1)
        u_desired = opti.parameter(self.n_u, self.N)

        # All mu_x, Sigma_x are optimization variables. mu_x array is nxN and mu_u array is mxN.
        mu_x = opti.variable(self.n_x, self.N + 1)
        mu_u = opti.variable(self.n_u, self.N)
        mu_z = cs.vertcat(mu_x[:, :-1], mu_u)
        residual_means = []
        mu_d = cs.MX.sym('mu_d', self.n_d, self.N)

        # Mean function of GP takes (x, u) as input and outputs the mean and cov of the residual term for the current timestep.
        gp_inp_vec_size = np.linalg.matrix_rank(self.input_mask)
        gp_input = cs.MX.zeros(gp_inp_vec_size, self.N)
        temp_x, temp_z = cs.MX.sym('temp_x', self.n_x, 1), cs.MX.sym('temp_z', self.n_x + self.n_u, 1)
        try:
            gp_input_fn = cs.Function('gp_input', [temp_z], [self.input_mask @ temp_z], {"enable_fd": True})
        except TypeError:
            print(temp_z.shape, self.input_mask.shape)
        for k in range(self.N):
            gp_input[:, k] = gp_input_fn(mu_z[:, k])

        # Initial condition constraints. x0 is known with complete certainty.
        self.opti.subject_to(mu_x[:, 0] - x_init == 0)
        # Apparently casadi can't do matrix mult with Torch instances but only numpy instead. So have to use the np version of the H and b matrix/vector here.
        # Constraint for the initial state to be within the unshrunk state constraint set.
        self.opti.subject_to(self.X.H_np @ mu_x[:, 0] - self.X.b_np <= 0)

        self.ignore_covariances = self.add_b_shrunk_param and self.ignore_variance_costs
        if not self.ignore_covariances:
            residual_covs, Sigma_x, Sigma_u, Sigma = self.init_cov_arrays()

        if self.piecewise:
            # Piecewise functions to extract the right mean and covariance based on the hld vector at the current timestep
            hybrid_means_sym = cs.MX.sym('hybrid_means_sym', self.n_d, self.num_regions)
            delta_k_sym = cs.MX.sym('delta_k_sym', self.num_regions, 1)
            self.get_mu_d = cs.Function('get_mu_d', [hybrid_means_sym, delta_k_sym],
                                        [(delta_k_sym.T @ hybrid_means_sym.T).T],
                                        {'enable_fd': True})
            self.get_Sigma_d = hybrid_res_covar('hybrid_res_cov', self.n_d, self.num_regions,
                                                self.N, opts={'enable_fd': True},
                                                delta_tol=(1e-2 if self.add_delta_tol else 0), test_softplus=self.test_softplus)

            delta_inp_dim = np.linalg.matrix_rank(self.delta_input_mask)
            delta_controls = cs.MX.zeros(delta_inp_dim, self.N)
            self.setup_delta_control_inputs(delta_controls, mu_z)

            # Create deltas and setup relaxed constraints
            self.nx_before_deltas = self.opti.nx
            # This is the high_level_deltas vector. Each timestep has a column vector of 1's and 0's that must sum to 1.
            high_level_deltas = self.opti.variable(len(self.regions), self.N)
            self.setup_delta_domain(high_level_deltas=high_level_deltas)

        # All cov related callbacks
        self.compute_sigx_callback = Sigma_x_dynamics_Callback_LTI('sigma_x_compute', self.affine_transform,
                                                                   self.n_x + self.n_u + self.n_d, self.n_x, opts={'enable_fd': True})
        self.compute_sigma_u = Sigma_u_Callback('Sigma_u', self.K,
                                                opts={"enable_fd": True})
        self.computesigma_wrapped = computeSigma_meaneq('Sigma',
                                                        feedback_mat=self.K,
                                                        residual_dim=self.n_d,
                                                        opts={"enable_fd": True})

        for k in range(self.N):
            if self.piecewise:
                hybrid_means, *hybrid_covs = self.gp_fns(gp_input[:, k])
                mean_vec = self.get_mu_d(hybrid_means, high_level_deltas[:, [k]])
                mu_d[:, k] = mean_vec
                # Only need to ignore the residual covariance mat if we're not dealing with the approx'd case and the
                # variance costs and being ignored.
                if not self.ignore_covariances:
                    residual_cov_mat = self.get_Sigma_d(high_level_deltas[:, [k]], *hybrid_covs)
            else:
                mean_vec, residual_cov_mat = self.gp_fns(gp_input[:, k])
            residual_means.append(mean_vec)
            if not self.ignore_covariances:
                residual_covs.append(residual_cov_mat)

            # Computes Sigma^u_{k} from Sigma^x_{k} computed from the for loop's previous iteration.
            if not self.ignore_covariances:
                if k > 0:
                    # Sigma_x[k] was set in the previous iteration (i.e. (k-1)th iteration sets dynamics for Sigma^x_{(k-1)+1} = Sigma^x_{k})
                    Sigma_u_k = self.get_sigma_u_k(k, Sigma_x, Sigma_u)
                    Sigma_u.append(Sigma_u_k)
                else:
                    Sigma_u_k = Sigma_u[0]

                # Computes Sigma_{k} and also Sigma^x_{k+1}
                self.sigma_x_dynamics_lti(k, Sigma_x, Sigma_u_k, Sigma, residual_cov_mat)

            # State mean dynamics
            self.opti.subject_to(mu_x[:, k + 1] - (self.A @ mu_x[:, k] + self.B @ mu_u[:, k] + self.Bd @ (mean_vec)) == 0)

        if self.piecewise:
            # These constraints are all involving the true opt vars mu_x, mu_u (via the delta controls) and the deltas so
            # it doesn't matter where you put them.
            # Sets up constraints when the lld corresponding to the inequality row k of region r is 1 only if the delta_control input
            # satisfies it and 0 if it does not.
            self.setup_delta_constraints(high_level_deltas, delta_controls)

        # We need to have a way to set x_init, x_desired etc. at every iteration of the closed loop optimization. This dict will maintain
        # references to variables/parameters contained in the "opti" instance and set values for parameters/provide warm start solutions for variables
        self.opti_dict = {
            "mu_x": mu_x,
            "mu_u": mu_u,
            "mu_z": mu_z,
            "x_init": x_init,
            "x_desired": x_desired,
            "u_desired": u_desired,
            "gp_input": gp_input,
            "mu_d": residual_means
        }
        if not self.ignore_covariances:
            self.opti_dict.update({"Sigma_x": Sigma_x,
                                   "Sigma_u": Sigma_u,
                                   "Sigma": Sigma,
                                   "Sigma_d": residual_covs})
        if self.piecewise:
            self.opti_dict.update({"hld": high_level_deltas})

        if self.add_b_shrunk_param or self.add_b_shrunk_opt:
            if self.add_b_shrunk_opt:
                # First vector is always going to be X.b and U.b (i.e. no shrinking) for both assuming the state is known with certainty.
                b_shrunk_x = cs.horzcat(cs.DM(self.X.b_np), self.opti.variable(2*self.n_x, self.N))
                b_shrunk_u = cs.horzcat(cs.DM(self.U.b_np), self.opti.variable(2*self.n_u, self.N-1))
            else:
                b_shrunk_x = cs.horzcat(cs.DM(self.X.b_np), self.opti.parameter(2*self.n_x, self.N))
                b_shrunk_u = cs.horzcat(cs.DM(self.U.b_np), self.opti.parameter(2*self.n_u, self.N-1))
            self.opti_dict["b_shrunk_x"] = b_shrunk_x
            self.opti_dict["b_shrunk_u"] = b_shrunk_u

        # Now that Sigma_x and Sigma_u symbolic chaining has been setup, we can setup the cost function and shrinking.
        self.setup_cost_fn_and_shrinking()
        self.setup_solver()

    def setup_delta_domain(self, high_level_deltas):
        # Domain specification for the delta variable. Combined with the fact that variable is discrete this makes the domain {0, 1}
        for k in range(self.N):
            for region_idx in range(self.num_regions):
                self.opti.subject_to(-high_level_deltas[region_idx, k] <= 0)
                self.opti.subject_to(high_level_deltas[region_idx, k] - 1 <= 0)

    def setup_delta_control_inputs(self, delta_controls, mu_z):
        temp_x, temp_z = cs.MX.sym('temp_x', self.n_x, 1), cs.MX.sym('temp_z', self.n_x + self.n_u, 1)
        for k in range(self.N):
            # Extract the variables of the state (or joint state and input) vector to be used for checking the delta conditions
            f = cs.Function('select_delta_controllers', [temp_z], [self.delta_input_mask @ temp_z],
                            {"enable_fd": True})
            delta_controls[:, k] = f(mu_z[:, k])

    def setup_delta_constraints(self, high_level_deltas, delta_controls):
        for k in range(self.N):
            # Specific to H and b matrix ordering generated by the box constraint class
            for region_idx, region in enumerate(self.regions):
                region_H, region_b = region.H_np, region.b_np
                self.opti.subject_to(region_H @ delta_controls[:, [k]] <= (region_b + self.big_M*(1-high_level_deltas[region_idx, k])))
            # Enforce that all deltas at any time step must sum to 1.
            self.opti.subject_to(cs.DM.ones(1, self.num_regions) @ high_level_deltas[:, [k]] - 1 == 0)

    def shrink_constraints(self, Sigma_x, Sigma_u, k=None):
        # Tightened sets b vectors
        # Note we cannot use numpy functions here because we're dealing with symbolic variables. Hence we need to make use of equivalent Casadi functions that are
        # capable of handling this.
        # Add constant before taking square root so that derivative includes the constant term to stop derivative blowing up to NaN
        if self.add_b_shrunk_opt:
            self.opti.subject_to(self.opti_dict['b_shrunk_x'][:, k+1] - (self.X.b_np - (cs.fabs(self.X.H_np) @ (cs.sqrt(cs.diag(Sigma_x) + self.sqrt_constant) * self.inverse_cdf_x))) == 0)
            # Don't need subject to for 1st vector since no shrinking.
            if k > 0:
                self.opti.subject_to(self.opti_dict['b_shrunk_u'][:, k] - (self.U.b_np - (cs.fabs(self.U.H_np) @ (cs.sqrt(cs.diag(Sigma_u) + self.sqrt_constant) * self.inverse_cdf_u))) == 0)
            return
        if self.add_b_shrunk_param:
            return self.opti_dict['b_shrunk_x'][:, k], self.opti_dict['b_shrunk_u'][:, k]
        else:
            b_shrunk_x = self.X.b_np - (cs.fabs(self.X.H_np) @ (cs.sqrt(cs.diag(Sigma_x) + self.sqrt_constant) * self.inverse_cdf_x))
            b_shrunk_u = self.U.b_np - (cs.fabs(self.U.H_np) @ (cs.sqrt(cs.diag(Sigma_u) + self.sqrt_constant) * self.inverse_cdf_u))
            return b_shrunk_x, b_shrunk_u

    def set_shrunk_params(self, b_shrunk_x_precomp, b_shrunk_u_precomp):
        b_shrunk_u, b_shrunk_x = self.opti_dict['b_shrunk_u'], self.opti_dict['b_shrunk_x']
        for k in range(1, self.N+1):
            if k < self.N:
                self.opti.set_value(b_shrunk_u[:, k], b_shrunk_u_precomp[:, k])
            self.opti.set_value(b_shrunk_x[:, k], b_shrunk_x_precomp[:, k])

    def setup_cost_fn_and_shrinking(self):
        # Retrieve variables
        mu_x, mu_u, x_desired, u_desired = self.get_attrs_from_dict(['mu_x', 'mu_u', 'x_desired', 'u_desired'])
        if not self.ignore_covariances:
            Sigma_x, Sigma_u = self.get_attrs_from_dict(['Sigma_x', 'Sigma_u'])
        # Shrinking constraints
        if not (self.add_b_shrunk_opt or self.add_b_shrunk_param):
            b_shrunk_x_arr, b_shrunk_u_arr = [self.X.b_np], [self.U.b_np]
        for k in range(self.N):
            # Constraining mu^x_{i+1} to lie within a region that satisfies the state constraints with probability=self.satisfaction_prob
            # Similarly for mu^u_{i}
            if not (self.add_b_shrunk_opt or self.add_b_shrunk_param):
                b_shrunk_x, b_shrunk_u = self.shrink_constraints(Sigma_x[k + 1], Sigma_u[k])
                b_shrunk_x_arr.append(b_shrunk_x)
                # If this condition isn't here then there will be a small amount of shrinking for the first timestep on u owing
                # to the sqrt constant term in the shrinking equation. For future timesteps where shrinking is actually required, the
                # additional shrinking generated by the sqrt constant is negligible. Also note that this sqrt constant is only being
                # added for shrinking purposes but NOT to the actual covariance matrices. Hence it doesn't compound over the horizon but
                # is only applicable for that timestep.
                if k >= 1:
                    b_shrunk_u_arr.append(b_shrunk_u)
            if self.add_b_shrunk_opt:
                self.shrink_constraints(Sigma_x[k + 1], Sigma_u[k], k=k)

            if self.skip_shrinking:
                self.opti.subject_to(self.X.H_np @ mu_x[:, k + 1] - self.X.b_np <= 0)
                self.opti.subject_to(self.U.H_np @ mu_u[:, k] - self.U.b_np <= 0)
            else:
                # If shrunk vectors are opt, we need to retrieve them from the dict but if not opt they are created locally within this loop
                # so no need to retrieve.
                if self.add_b_shrunk_opt or self.add_b_shrunk_param:
                    # Note we're retrieving the k+1'th shrunk vector since the constraint is on mu_x[:, k+1]
                    b_shrunk_x = self.opti_dict['b_shrunk_x'][:, k+1]
                    b_shrunk_u = self.opti_dict['b_shrunk_u'][:, k]
                self.opti.subject_to(self.X.H_np @ mu_x[:, k + 1] - b_shrunk_x <= 0)
                self.opti.subject_to(self.U.H_np @ mu_u[:, k] - b_shrunk_u <= 0)

        # If b shrunk is opt, this entry has already been added to dict at the end of setup OL opt just before calling the cost_fn_and_shrinking
        # method to help with ease of setting up the b_shrunk constraints.
        if not (self.add_b_shrunk_opt or self.add_b_shrunk_param):
            self.opti_dict['b_shrunk_x'] = b_shrunk_x_arr
            self.opti_dict['b_shrunk_u'] = b_shrunk_u_arr

        # Cost function stuff
        cost = 0
        # Stage cost
        for k in range(self.N):
            Sigma_x_k = Sigma_x[k] if not self.add_b_shrunk_param else None
            Sigma_u_k = Sigma_u[k] if not self.add_b_shrunk_param else None
            cost += self.cost_fn(mu_i_x=mu_x[:, k], mu_i_u=mu_u[:, k], Sigma_x_i=Sigma_x_k, Sigma_u_i=Sigma_u_k,
                                 x_desired=x_desired, u_desired=u_desired, idx=k)
        # Terminal cost
        Sigma_x_N = Sigma_x[-1] if not self.add_b_shrunk_param else None
        cost += self.cost_fn(mu_x[:, -1], None, Sigma_x_N, None, x_desired=x_desired, u_desired=None,
                             terminal=True)
        if not self.ignore_cost:
            self.opti.minimize(cost)

    def setup_solver(self):
        if self.piecewise and not self.relaxed:
            self.nx_after_deltas = self.opti.nx
            # When sigma_x is an opt var, the shrinking and cost function set up is done in this child class instead of the parent
            # GP_MPC_Base class. As a result, mu_u is not added to the opt var list before entering this class and so nx_before_deltas
            # doesn't contain the number of mu_u variables. To amend this, we added number of mu_u's to nx_before_deltas
            num_control_vars = self.n_u * self.N
            self.nx_before_deltas += num_control_vars

            # for i in range(self.nx_before_deltas):
            #     print(self.opti.debug.x_describe(i))
            # print('after')
            # for i in range(self.nx_before_deltas, self.nx_after_deltas):
            #     print(self.opti.debug.x_describe(i))

            num_hld = self.num_regions * self.N
            num_shrunk = 0
            if self.add_b_shrunk_opt:
                # Note only times N for x because the first vector of zeros is not opt and similarly for u.
                num_shrunk = 2*self.n_x*(self.N) + 2*self.n_u*(self.N-1)
            num_added = num_hld + num_shrunk

            assert self.nx_after_deltas == (self.nx_before_deltas + num_added), "nx isn't created in the order expected. " \
                                                                                "nx before deltas: %s ;  nx after deltas: %s != %s + %s + %s; \n " \
                                                                                "mu_x, mu_u opt vars: \n %s \n" \
                                                                                "delta opti_vars: \n %s \n" \
                                                                                "rem opti_vars: \n %s" \
                                                                                % (self.nx_before_deltas,
                                                                                   self.nx_after_deltas,
                                                                                   self.nx_before_deltas,
                                                                                   num_hld, num_shrunk,
                                                                                   '\n'.join([
                                                                                       self.opti.debug.x_describe(
                                                                                           idx) for idx in
                                                                                       range(self.nx_before_deltas)]),
                                                                                   '\n'.join([
                                                                                       self.opti.debug.x_describe(
                                                                                           idx) for idx in
                                                                                       range(
                                                                                           self.nx_before_deltas,
                                                                                           self.nx_after_deltas-num_shrunk)]),
                                                                                   '\n'.join([
                                                                                       self.opti.debug.x_describe(
                                                                                           idx) for idx in
                                                                                       range(
                                                                                           self.nx_after_deltas-num_shrunk,
                                                                                           self.nx_after_deltas)]))
            discrete_bool_vec = [False] * (self.nx_before_deltas) + [True] * (num_hld) + [False] * num_shrunk
            # opts = {"enable_fd": True, "bonmin.max_iter": 20, "bonmin.print_level": 4}
            new_opts = {}
            self.solver_opts['discrete'] = discrete_bool_vec
            for opt in self.solver_opts.keys():
                if 'ipopt' not in opt:
                    new_opts[opt] = self.solver_opts[opt]
            self.solver_opts = new_opts
        else:
            new_opts = {}
            for opt in self.solver_opts.keys():
                if 'bonmin' not in opt:
                    new_opts[opt] = self.solver_opts[opt]
            self.solver_opts = new_opts

        self.opti.solver(self.solver, self.solver_opts)


def scalar_region_gen_and_viz(viz=True, s_start_limit=np.array([[-2]]).T, s_end_limit=np.array([[2]]).T,
                              x0_delim=-0.5):
    r1_start, r1_end = np.array([[s_start_limit[0, :].item()]]).T,\
                       np.array([[x0_delim]]).T
    r2_start, r2_end = np.array([[x0_delim]]).T,\
                       np.array([[s_end_limit[0, :].item()]]).T
    regions = [box_constraint(r1_start, r1_end), box_constraint(r2_start, r2_end)]

    # Add values to generate samples that lie outside of the constraint set to test those too
    grid_check = generate_fine_grid(s_start_limit-1, s_end_limit+1, fineness_param=(10, 10), viz_grid=False)
    # print(grid_check.shape)
    mask = [[], []]
    for grid_vec_idx in range(grid_check.shape[-1]):
        grid_vec = grid_check[:, grid_vec_idx]
        for region_idx in range(len(regions)):
            test_constraint = regions[region_idx]
            mask[region_idx].append((test_constraint.sym_func(grid_vec) <= 0).all().item())
    passed_vecs = [0, 0]
    colours = ['r', 'g']
    if viz:
        plt.figure()
        for i in range(len(regions)):
            passed_vecs[i] = grid_check[:, mask[i]]
            plt.scatter(passed_vecs[i][0], np.zeros((1, passed_vecs[i][0].shape[-1])), c=colours[i])
        print(grid_check)
    return regions


def construct_gp_wrapped_piecewise(hardcoded, regions, s_start_limit, s_end_limit, fixed_numeric_means,
                                   num_samples=50, noise_std_devs=(0.04, 0.01), viz=True, return_error_attrs=False):
    # Global and piecewise both use the same ds_gen step but the training function is different.
    if not hardcoded:
        ds_ndim_test: GP_DS
        ds_ndim_test = test_1d_op_1d_inp_allfeats(regions,
                                                  start_limit=s_start_limit, end_limit=s_end_limit, num_points=num_samples,
                                                  noise_vars=[noise_std_dev ** 2 for noise_std_dev in noise_std_devs])
    if not hardcoded:
        returned_train_ops = piecewise_train_test(ds_ndim_test, no_squeeze=True, return_trained_covs=return_error_attrs)
        if not return_error_attrs:
            likelihoods_piecewise_nd, piecewise_models_nd = returned_train_ops
        else:
            likelihoods_piecewise_nd, piecewise_models_nd, region_trained_covs = returned_train_ops
        likelihoods, models = likelihoods_piecewise_nd, piecewise_models_nd
        for model_idx in range(len(models)):
            models[model_idx].eval()
            likelihoods[model_idx].eval()
        res_input_dim = ds_ndim_test.input_dims
        num_regions = len(regions)
        res_output_dim = ds_ndim_test.output_dims
        piecewise_gp_wrapped = Piecewise_GPR_Callback('f', likelihoods, models,
                                                      output_dim=res_output_dim, input_dim=res_input_dim, num_regions=num_regions,
                                                      opts={"enable_fd": True})

    else:
        assert NotImplementedError, "This config option is redundant"

    if viz and not hardcoded:
        addn_rows = 0 if not return_error_attrs else 1
        fig, axes = plt.subplots(2+addn_rows, 2, figsize=(20, 30))
        ds_ndim_test.viz_outputs_1d(fineness_param=(70, ), ax1=axes[0, 0])
        # num_samples = 1000
        # region_masks = {0: np.hstack([np.zeros(500), np.ones(500)]), 1: np.hstack([np.ones(500), np.zeros(500)])}
        # ds_ndim_test._gen_white_noise(noise_verbose=False, ax=axes[1], num_samples=num_samples, region_masks=region_masks)
        fine_grid, mask = ds_ndim_test.generate_fine_grid(fineness_param=(30,), with_mask=True)
        region_xs, observed_preds, observed_preds_full = [], [], []
        if return_error_attrs:
            mean_error_list = []
            cov_list = []
        for idx in range(len(models)):
            likelihood, model = likelihoods[idx], models[idx]
            region_mask = mask[idx]
            region_idxs = np.nonzero(region_mask)[0]
            if return_error_attrs:
                region_errors = np.zeros([*region_idxs.shape])
            region_test_samples = fine_grid[np.nonzero(region_mask)]
            # print(region_test_samples)
            # print(type(region_test_samples))
            observed_pred = likelihood(model(GPR_Callback.preproc(region_test_samples)))
            # Check to ensure the callback method is working as intended
            with torch.no_grad():
                # callback sparsity is 1 sample at a time so need to iterate through all 1 at a time
                for sample_idx in range(region_test_samples.shape[-1]):
                    sample = region_test_samples[sample_idx]
                    residual_mean, *residual_covs = piecewise_gp_wrapped(sample)
                    true_mean = ds_ndim_test.generate_outputs(input_arr=torch.from_numpy(sample[None][None]), no_noise=True)
                    non_callback_mean = observed_pred.mean.numpy()[sample_idx]
                    cov_list.append((np.sqrt(residual_covs[idx]), idx))
                    if return_error_attrs:
                        region_errors[sample_idx] = np.abs(true_mean - non_callback_mean)
                    # assert np.isclose(residual_mean[:, idx], non_callback_mean, rtol=1e-4), \
                    #     "GP output mean (%s) and non-callback residual mean (%s) don't match: " % (residual_mean[:, idx], non_callback_mean)
            if return_error_attrs:
                mean_error_list.append({"region_samples": region_test_samples, "region_errors": region_errors})
            observed_pred_full = likelihood(model(GPR_Callback.preproc(fine_grid.squeeze())))
            observed_preds.append(observed_pred)
            observed_preds_full.append(observed_pred_full)
            region_xs.append(region_test_samples)
        idx = -1
        colours = ['r', 'g', 'b']
        with torch.no_grad():
            for region_x, observed_pred, observed_pred_full in zip(region_xs, observed_preds, observed_preds_full):
                idx += 1
                lower, upper = observed_pred.confidence_region()
                axes[1, 1].plot(region_x, observed_pred.mean.numpy(), colours[idx], label='GP %s' % (idx+1))
                axes[1, 1].fill_between(region_x, lower.numpy(), upper.numpy(), alpha=0.5)
                # axes[1, 0].set_title("Trained hybrid GPs visualized over corresponding regions", fontsize=25)
                lower_full, upper_full = observed_pred_full.confidence_region()
                axes[1, 0].plot(fine_grid.squeeze(), observed_pred_full.mean.numpy(), colours[idx], label='GP %s' % (idx+1))
                axes[1, 0].fill_between(fine_grid.squeeze(), lower_full.numpy(), upper_full.numpy(), alpha=0.5)
                # axes[1, 1].set_title("Trained hybrid GPs visualized across full input domain space", fontsize=25)

        # axes[0, 1].set_title("Trained global GP visualized across full input domain space", fontsize=25)

        axes[1, 0].legend(loc="upper center", fontsize=25)
        axes[1, 1].legend(loc="upper center", fontsize=25)
        # axes[0, 1].legend(loc="upper center", fontsize=20)

        labels = ["(a)", "(b)", "(c)", "(d)"]
        if return_error_attrs:
            labels.extend(["(e)", "(f)"])
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].text(0.5, -0.08, labels[i*2+j], horizontalalignment='center', verticalalignment='center',
                                transform=axes[i, j].transAxes, size=25)
                axes[i, j].xaxis.set_tick_params(labelsize=25)
                axes[i, j].yaxis.set_tick_params(labelsize=25)

        names = ["scalar_true_func_w_samples", "piecewise_regional", "piecewise_w_full_observed"]
        ds_utils.save_subplot(axes[0, 0], figure=fig, fig_name=names[0])
        ds_utils.save_subplot(axes[1, 1], figure=fig, fig_name=names[1])
        ds_utils.save_subplot(axes[1, 0], figure=fig, fig_name=names[2])

    if viz and not hardcoded:
        if return_error_attrs:
            return piecewise_gp_wrapped, ds_ndim_test, axes, fig, region_trained_covs, mean_error_list, cov_list
        else:
            return piecewise_gp_wrapped, ds_ndim_test, axes, fig
    else:
        if return_error_attrs:
            return piecewise_gp_wrapped, ds_ndim_test, region_trained_covs, mean_error_list, cov_list
        else:
            return piecewise_gp_wrapped, ds_ndim_test


def construct_gp_wrapped_global(hardcoded=False, fixed_numeric_means=False, ds_in: GP_DS=None, viz=True, ax=None, fig=None,
                                return_error_attrs=False):
    assert ds_in is not None, "You must pass an input dataset that is shared across the global and piecewise case and generated by construct_gp_wrapped_piecewise"
    ds_ndim_test = ds_in
    if not hardcoded:
        returned_train_ops = train_test(ds_ndim_test, no_squeeze=True, return_trained_covs=return_error_attrs)
        if not return_error_attrs:
            likelihoods_1d, models_1d = returned_train_ops
        else:
            likelihoods_1d, models_1d, baseline_trained_covs = returned_train_ops
        likelihoods, models = likelihoods_1d, models_1d
        for model_idx in range(len(models)):
            models[model_idx].eval()
            likelihoods[model_idx].eval()
        res_input_dim = ds_ndim_test.input_dims
        res_output_dim = ds_ndim_test.output_dims
        global_gp_wrapped = MultidimGPR_Callback('f', likelihoods, models,
                                                 state_dim=res_input_dim, output_dim=res_output_dim, opts={"enable_fd": True})

    else:
        assert NotImplementedError, "This config option is redundant now"

    if viz and not hardcoded:
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 13))
            ds_ndim_test.viz_outputs_1d(fineness_param=(70, ), ax1=axes[0])
            ax = axes[1]
        observed_preds = []
        fine_grid = ds_ndim_test.generate_fine_grid(fineness_param=(30,)).squeeze()
        if return_error_attrs:
            mean_error_list = []
            cov_list = []
        for idx in range(len(models)):
            likelihood, model = likelihoods[idx], models[idx]
            observed_pred = likelihood(model(GPR_Callback.preproc(fine_grid.squeeze())))
            if return_error_attrs:
                errors = np.zeros([fine_grid.shape[-1]])
            # Check to ensure the callback method is working as intended
            with torch.no_grad():
                # callback sparsity is 1 sample at a time so need to iterate through all 1 at a time
                for sample_idx in range(fine_grid.shape[-1]):
                    sample = fine_grid[sample_idx]
                    non_callback_mean = observed_pred.mean.numpy()[sample_idx]
                    residual_mean, residual_cov = global_gp_wrapped(sample)
                    # cov_list.append((np.sqrt(observed_pred.covariance_matrix.detach().numpy()), idx))
                    cov_list.append((np.sqrt(residual_cov), idx))
                    true_mean = ds_ndim_test.generate_outputs(input_arr=torch.from_numpy(sample[None][None]), no_noise=True)
                    if return_error_attrs:
                        errors[sample_idx] = np.abs(true_mean - non_callback_mean)
                    # residual_mean, *residual_covs = global_gp_wrapped(sample)
                    # assert np.isclose(residual_mean[:, idx], non_callback_mean, atol=1e-5), \
                    #     "GP output mean (%s) and non-callback residual mean (%s) don't match: " % (residual_mean[:, idx], non_callback_mean)
            if return_error_attrs:
                mean_error_list.append({"test_samples": fine_grid, "errors": errors})

            observed_preds.append(observed_pred)
        idx = -1
        colours = ['r', 'g', 'b']
        with torch.no_grad():
            for observed_pred in observed_preds:
                idx += 1
                lower, upper = observed_pred.confidence_region()
                ax.plot(fine_grid.squeeze(), observed_pred.mean.numpy(), colours[idx])
                ax.fill_between(fine_grid.squeeze(), lower.numpy(), upper.numpy(), alpha=0.5)
        ds_utils.save_subplot(ax, figure=fig, fig_name="global_trained")
    if return_error_attrs:
        return global_gp_wrapped, baseline_trained_covs, mean_error_list, cov_list
    else:
        return global_gp_wrapped


def construct_config_opts(print_level, add_monitor, early_termination, only_soln_limit, hsl_solver,
                          closed_loop, test_no_lam_p, warmstart, bonmin_warmstart=False, hessian_approximation=True,
                          enable_forward=False, relaxed=False, piecewise=True):
    if piecewise and not relaxed:
        opts = {"bonmin.print_level": print_level, 'bonmin.file_solution': 'yes', 'bonmin.expect_infeasible_problem': 'no'}
        # opts.update({"bonmin.allowable_gap": -100, 'bonmin.allowable_fraction_gap': -0.1, 'bonmin.cutoff_decr': -10})
        opts.update({"bonmin.allowable_gap": 1e5})
        # Ref Page 11 https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.432.3757&rep=rep1&type=pdf
        # opts.update({"bonmin.num_resolve_at_root": 2, 'bonmin.num_resolve_at_node': 2,
        #              "bonmin.max_random_point_radius": 0.5, "bonmin.random_point_type": "Andreas"})
        if enable_forward:
            opts["enable_fd"] = False
            opts["enable_forward"] = True
        if add_monitor:
            opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
        if early_termination:
            opts["bonmin.solution_limit"] = 1
            # opts["bonmin.heuristic_RINS"] = "yes"
            # opts["RINS.algorithm"] = "B-QG"
            # opts["bonmin.rins.solution_limit"] = 1
            # opts["bonmin.algorithm"] = "B-OA"
            # if warmstart:
            #     opts["bonmin.iteration_limit"] = 1
            #     opts["bonmin.node_limit"] = 1
        if not only_soln_limit:
            opts["bonmin.allowable_gap"] = 1e5
            opts["bonmin.integer_tolerance"] = 1e-2
        if hsl_solver:
            opts['bonmin.linear_solver'] = 'ma27'
        if closed_loop:
            opts["bonmin.nlp_failure_behavior"] = "fathom"
        # Attempt to fix grad_gamma_p error. Ref: https://groups.google.com/g/casadi-users/c/vDewPLPXLYA/m/96BP6Nt2BQAJ
        if test_no_lam_p:
            opts["calc_lam_p"] = False
        if warmstart:
            if bonmin_warmstart:
                opts["bonmin.warm_start"] = "interior_point"
        if hessian_approximation:
            # opts["hess_lag_options"] = {"Approximation": "True"}
            opts["bonmin.hessian_approximation"] = "limited-memory"
    else:
        opts = {"ipopt.print_level": print_level}
        acceptable_dual_inf_tol = 1e11
        acceptable_compl_inf_tol = 1e-3
        acceptable_iter = 1
        acceptable_constr_viol_tol = 1e-3
        acceptable_tol = 1e4

        if early_termination:
            additional_opts = {"ipopt.acceptable_tol": acceptable_tol, "ipopt.acceptable_constr_viol_tol":acceptable_constr_viol_tol,
                               "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol, "ipopt.acceptable_iter": acceptable_iter,
                               "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol}
            opts.update(additional_opts)
    return opts


def initialize_piecewise(x_init, warmstart, bonmin_warmstart, b_shrunk_opt, warmstart_dict, controller_test):
    # Set parameters and initial opt var assignment
    initialize_dict = {'x_init': x_init}
    if warmstart and not bonmin_warmstart:
        reqd_keys = ['mu_x', 'mu_u', 'hld', 'lld']
        if b_shrunk_opt:
            reqd_keys.append('b_shrunk_x')
            reqd_keys.append('b_shrunk_u')
        for key in reqd_keys:
            assert key in warmstart_dict.keys(), "key: %s must be present in solution to warmstart" % key
        initialize_dict.update(warmstart_dict)
    controller_test.set_initial(**initialize_dict)
    return initialize_dict


def initialize_global(x_init, warmstart, b_shrunk_opt, warmstart_dict, controller_test):
    # Set parameters and initial opt var assignment
    initialize_dict = {'x_init': x_init}
    if warmstart:
        reqd_keys = ['mu_x', 'mu_u']
        if b_shrunk_opt:
            reqd_keys.append('b_shrunk_x')
            reqd_keys.append('b_shrunk_u')
        for key in reqd_keys:
            assert key in warmstart_dict.keys(), "key: %s must be present in solution to warmstart" % key
        initialize_dict.update(warmstart_dict)
    controller_test.set_initial(**initialize_dict)
    return initialize_dict


def run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict, simulation_length=None, true_func_obj=None, piecewise=True):
    if simulation_length is None:
        # Reference: https://groups.google.com/g/casadi-users/c/n6TEPPatAg4/m/3UunhadtAQAJ
        if store_in_file:
            if file_name == '':
                file_name = 'solver_piecewise_hardcoded_fixed_numeric.txt'
            with open(file_name, 'w') as output_file:
                # Retrieve default stdout
                stdout_old = sys.stdout
                # Re-direct standard output
                sys.stdout = output_file
                try:
                    sol = controller_test.solve_optimization(ignore_initial=True, **initialize_dict)
                except:
                    # If error thrown -> print exception to old stdout
                    traceback.print_exc(file=stdout_old)
                finally:
                    # Re-direct stdout to default
                    sys.stdout = stdout_old
        else:
            try:
                try:
                    sol = controller_test.solve_optimization(ignore_initial=True, **initialize_dict)
                except executing.executing.NotOneValueFound:
                    print('error1')
                    print("NotOneValueFound error")
            except Exception as e:
                print(e)
    else:
        X_test, U_test, regions = controller_test.X, controller_test.U, controller_test.regions
        mu_x_cl = []
        for i in range(simulation_length):
            if store_in_file:
                output_file = open(file_name, 'w')
                # Retrieve default stdout
                stdout_old = sys.stdout
                # Re-direct standard output
                sys.stdout = output_file
            try:
                print("Solving for timestep: %s" % i)
                sol = controller_test.solve_optimization(ignore_initial=True, **initialize_dict)
                if not piecewise:
                    print("Solution")
                    debugger_inst = retrieve_controller_results(controller_test, X_test, U_test)
                    mu_x_ol = generate_and_set_warmstart_from_previous_iter_soln(controller_test, initialize_dict, X_test, U_test, regions, true_func_obj,
                                                                                 no_cov_case=controller_test.ignore_covariances, piecewise=piecewise, return_mu_x=True)
                    mu_x_cl.append(mu_x_ol)

            except Exception as e:
                # Error is thrown after finishing computing solution before generating the warmstarts for the next iteration.
                # So all the warmstarting code for the next iteration is put in this block after the exception has been caught.
                if piecewise:
                    print("Solution")
                    debugger_inst = retrieve_controller_results_piecewise(controller_test, X_test, U_test, ignore_lld=True)
                    mu_x_ol = generate_and_set_warmstart_from_previous_iter_soln(controller_test, initialize_dict, X_test, U_test, regions, true_func_obj,
                                                                                 no_cov_case=controller_test.ignore_covariances, piecewise=piecewise, return_mu_x=True)
                    mu_x_cl.append(mu_x_ol)
                print(e)
                if store_in_file:
                    # If error thrown -> print exception to old stdout
                    traceback.print_exc(file=stdout_old)
                    traceback.print_exc(file=sys.stdout)
            finally:
                if store_in_file:
                    # Re-direct stdout to default
                    sys.stdout = stdout_old
        return mu_x_cl


def gen_ds_and_train(hardcoded=False, fixed_numeric_means=False, num_samples=50, return_error_attrs=False,
                     seed=None, savefig=False):
    if seed is not None:
        np.random.seed(seed)

    s_start_limit, s_end_limit = setup_problem()[4:6]
    regions = setup_problem()[10]
    # In closed loop, the gp_ds obj contains info relating to the true function and can be used to generate the next state in simulation using the generate_outputs function
    # by passing a custom array (in this case just the first sample i.e. x_init for that O.L. timestep)
    pw_ops = construct_gp_wrapped_piecewise(hardcoded, regions, s_start_limit, s_end_limit, fixed_numeric_means, num_samples=num_samples,
                                            return_error_attrs=return_error_attrs)
    # print(pw_ops)
    if return_error_attrs:
        piecewise_gp_wrapped, gp_ds_inst, axes, fig, pw_covs, pw_mean_errors, pw_cov_list = pw_ops
    else:
        piecewise_gp_wrapped, gp_ds_inst, axes, fig = pw_ops
    glob_ops = construct_gp_wrapped_global(hardcoded=hardcoded, fixed_numeric_means=fixed_numeric_means, ds_in=gp_ds_inst, ax=axes[0, 1], fig=fig,
                                           return_error_attrs=return_error_attrs)
    if return_error_attrs:
        global_gp_wrapped, glob_covs, glob_mean_errors, glob_cov_list = glob_ops
    else:
        global_gp_wrapped = glob_ops
    # ds_utils.save_subplot(axes, figure=fig, fig_name="scalar_full_fig")
    if return_error_attrs:
        # axes[2, 0].set_title("Baseline mean function error plot over X domain")
        # axes[2, 1].set_title("Proposed approach mean function error plot over X domain")
        glob_samples, glob_errors = glob_mean_errors[0]["test_samples"], glob_mean_errors[0]["errors"]
        axes[2, 0].plot(glob_samples.squeeze(), glob_errors.squeeze())
        for region_idx in range(len(pw_mean_errors)):
            region_samples, region_errors = pw_mean_errors[region_idx]["region_samples"], pw_mean_errors[region_idx]["region_errors"]
            axes[2, 1].plot(region_samples.squeeze(), region_errors.squeeze())
    if savefig:
        plt.savefig("C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\images\\scalar_motivating_example\\scalar_full_fig.svg", bbox_inches='tight')
    if return_error_attrs:
        return piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst, pw_covs, glob_covs, pw_mean_errors, glob_mean_errors,\
                pw_cov_list, glob_cov_list

    else:
        return piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst


def cov_error_metric(num_runs=2):
    pw_r1_noise_errors, pw_r2_noise_errors = [], []
    glob_r1_noise_errors, glob_r2_noise_errors = [], []
    all_run_pw_mean_error, all_run_glob_mean_error = [], []
    x_mu_h_error, r1_mu_h_error, r2_mu_h_error, x_var_h_error, r1_var_h_error, r2_var_h_error = [0]*6
    x_mu_b_error, r1_mu_b_error, r2_mu_b_error, x_var_b_error, r1_var_b_error, r2_var_b_error = [0]*6

    for i in range(num_runs):
        piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst, \
        pw_std_dev, glob_std_dev, pw_mean_errors, glob_mean_errors, \
        pw_cov_list, glob_cov_list = gen_ds_and_train(hardcoded=False,
                                                      fixed_numeric_means=False,
                                                      num_samples=400, return_error_attrs=True,
                                                      seed=None, savefig=False)

        # glob_std_error, pw_std_error = 0, 0
        # true_covs = np.array([0.04, 0.01])
        true_covs = [0.04, 0.01]
        # print(pw_cov_list)
        # print(glob_cov_list)

        for cov_val, idx in pw_cov_list:
            cov_error = np.abs(cov_val - true_covs[idx])
            # print(cov_error)
            x_var_h_error += cov_error
            if idx == 0:
                r1_var_h_error += cov_error
            elif idx == 1:
                r2_var_h_error += cov_error
            else:
                raise AssertionError("Region value not 0 or 1")

        for region_idx in range(2):
            if region_idx == 0:
                r1_mu_h_error += np.sum(pw_mean_errors[0]["region_errors"])
            if region_idx == 1:
                r2_mu_h_error += np.sum(pw_mean_errors[1]["region_errors"])
        glob_samples, glob_errors = glob_mean_errors[0]["test_samples"], glob_mean_errors[0]["errors"]
        glob_delimiter_idx = np.nonzero(glob_samples > -0.5)[0][0]
        print(glob_delimiter_idx)
        # print(glob_errors)
        print(glob_errors.shape)
        for sample_idx, (cov_val, idx) in enumerate(glob_cov_list):
            cov_error = np.abs(cov_val - true_covs[idx])
            # print(cov_error)
            x_var_b_error += cov_error
            if sample_idx < glob_delimiter_idx:
                r1_var_b_error += cov_error
            else:
                r2_var_b_error += cov_error
        r1_mu_b_error += np.sum(glob_errors[:glob_delimiter_idx])
        r2_mu_b_error += np.sum(glob_errors[glob_delimiter_idx:])
        x_mu_h_error += np.sum([np.sum(region_mean_errors["region_errors"]) for region_mean_errors in pw_mean_errors])
        x_mu_b_error += np.sum([glob_mean_errors[0]["errors"]])
        # all_run_pw_mean_error.append(np.sum([np.sum(region_mean_errors["region_errors"]) for region_mean_errors in pw_mean_errors]))
        # all_run_glob_mean_error.append()

        # r1_pw_std_error = np.sum(pw_std_dev[0] - true_covs[0])
        # r1_glob_std_error = np.sum(true_covs[0] - np.array(glob_std_dev[0]))
        # r2_pw_std_error = np.sum(np.array(pw_std_dev[1]) - true_covs[1])
        # r2_glob_std_error = np.sum(true_covs[1] - np.array(glob_std_dev[0]))
        # # print(pw_std_error)
        # # print(glob_std_error)
        # # for region_idx in range(2):
        # #     glob_std_error += np.abs(glob_std_dev[0] - true_covs[region_idx])
        # #     pw_std_error += np.abs(pw_std_dev[region_idx] - true_covs[region_idx])
        # pw_r1_noise_errors.append(r1_pw_std_error)
        # pw_r2_noise_errors.append(r2_pw_std_error)
        # glob_r1_noise_errors.append(r1_glob_std_error)
        # glob_r2_noise_errors.append(r2_glob_std_error)
        # all_run_pw_mean_error.append(np.sum([np.sum(region_mean_errors["region_errors"]) for region_mean_errors in pw_mean_errors]))
        # all_run_glob_mean_error.append(np.sum([glob_mean_errors[0]["errors"]]))

    # print(all_run_pw_stddev_error)
    # print(all_run_glob_stddev_error)
    # print(all_run_pw_mean_error)
    # print(all_run_glob_mean_error)

    # print("PW R1 stddev error mean: %s variance: %s" % (np.mean(pw_r1_noise_errors), np.sqrt(np.array(pw_r1_noise_errors).var())))
    # print("PW R2 stddev error mean: %s variance: %s" % (np.mean(pw_r2_noise_errors), np.sqrt(np.array(pw_r2_noise_errors).var())))
    # print("Glob R1 stddev error mean: %s variance: %s" % (np.mean(glob_r1_noise_errors), np.sqrt(np.array(glob_r1_noise_errors).var())))
    # print("Glob R2 stddev error mean: %s variance: %s" % (np.mean(glob_r2_noise_errors), np.sqrt(np.array(glob_r2_noise_errors).var())))
    # print("PW mean function error mean: %s variance: %s" % (np.mean(all_run_pw_mean_error), np.sqrt(np.array(all_run_pw_mean_error).var())))
    # print("Glob mean function error mean: %s variance: %s" % (np.mean(all_run_glob_mean_error), np.sqrt(np.array(all_run_glob_mean_error).var())))
    print(x_mu_h_error/num_runs)
    print(r1_mu_h_error/num_runs)
    print(r2_mu_h_error/num_runs)
    print(x_mu_b_error/num_runs)
    print(r1_mu_b_error/num_runs)
    print(r2_mu_b_error/num_runs)
    print(x_var_h_error/num_runs)
    print(r1_var_h_error/num_runs)
    print(r2_var_h_error/num_runs)
    print(x_var_b_error/num_runs)
    print(r1_var_b_error/num_runs)
    print(r2_var_b_error/num_runs)


def setup_problem(stable_system=True, custom_U=False, U_big=True):
    if stable_system:
        A_test = np.array([[0.5]])
    else:
        A_test = np.array([[1.25]])
    B_test = np.array([[0.75]])
    Q_test, R_test = np.eye(1)*0.75, np.eye(1)*0.05
    s_start_limit, s_end_limit = np.array([[-2]]).T, np.array([[2]]).T
    if custom_U:
        u_start_limit, u_end_limit = custom_U
    else:
        if U_big:
            u_start_limit, u_end_limit = np.array([[-20]]).T, np.array([[20]]).T
        else:
            u_start_limit, u_end_limit = np.array([[-3]]).T, np.array([[3]]).T
    x_init = np.array([[-0.75]])

    x0_delim = -0.5
    regions = scalar_region_gen_and_viz(viz=False, s_start_limit=s_start_limit, s_end_limit=s_end_limit,
                                        x0_delim=x0_delim)
    X_test, U_test = box_constraint(s_start_limit, s_end_limit), box_constraint(u_start_limit, u_end_limit)
    return A_test, B_test, Q_test, R_test, s_start_limit, s_end_limit, u_start_limit, u_end_limit, x_init, x0_delim, regions, X_test, U_test


def GPR_test_hardcoded_allfeatsactive_piecewise(gp_fns, skip_shrinking=False, early_termination=False, only_soln_limit=False, minimal_print=True,
                                                stable_system=True, warmstart=False, warmstart_dict=None, add_monitor=True, store_in_file=True, file_name='',
                                                hsl_solver=False, no_solve=False, b_shrunk_opt=False,
                                                N=2, U_big=True, custom_U=None,
                                                approxd_shrinking=False, closed_loop=False, simulation_length=5,
                                                test_no_lam_p=False, bonmin_warmstart=False,
                                                hessian_approximation=False, ignore_cost=False, ignore_variance_costs=False,
                                                enable_forward=True, relaxed=False, add_delta_tol=False, test_softplus=False, true_func_obj: GP_DS=None,
                                                with_metrics=True, show_plot=True):
    A_test, B_test, Q_test, R_test, s_start_limit, s_end_limit,\
    u_start_limit, u_end_limit, x_init, x0_delim, regions, X_test, U_test = setup_problem(stable_system, custom_U, U_big)
    if closed_loop and with_metrics:
        assert true_func_obj is not None, "Need to pass a GP_DS inst that generated the data samples for closed loop simulation with metrics"

    gp_wrapped = gp_fns

    print_level = 0 if minimal_print else 5

    # U_shrunk = box_constraint(U_test.lb/100, U_test.ub/100)
    opts = construct_config_opts(print_level, add_monitor, early_termination, only_soln_limit, hsl_solver,
                                 closed_loop, test_no_lam_p, warmstart, bonmin_warmstart, hessian_approximation,
                                 enable_forward, relaxed)

    controller_test = Hybrid_GPMPC(A=A_test, B=B_test, Q=Q_test, R=R_test, horizon_length=N,
                                   gp_fns=gp_wrapped,  # Using gp model wrapped in casadi callback
                                   X=X_test, U=U_test, satisfaction_prob=0.4, regions=regions,
                                   skip_shrinking=skip_shrinking, addn_solver_opts=opts,
                                   add_b_shrunk_opt=b_shrunk_opt, piecewise=True, add_b_shrunk_param=approxd_shrinking,
                                   ignore_cost=ignore_cost, ignore_variance_costs=ignore_variance_costs, relaxed=relaxed,
                                   add_delta_tol=add_delta_tol, test_softplus=test_softplus)
    controller_test.display_configs()
    controller_test.setup_OL_optimization()

    if no_solve:
        controller_test.set_initial(**{'x_init': x_init})
        return controller_test
    else:
        initialize_dict = initialize_piecewise(x_init, warmstart, bonmin_warmstart, b_shrunk_opt, warmstart_dict, controller_test)

    if not closed_loop:
        run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict)
    else:
        mu_x_cl = run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict, simulation_length=simulation_length, true_func_obj=true_func_obj)
        if show_plot:
            plt.figure()
            for timestep, mu_x_ol in enumerate(mu_x_cl):
                plt.plot(range(timestep, timestep+N+1), mu_x_ol.squeeze(), 'bo')

    return controller_test, X_test, U_test


def generate_and_set_warmstart_from_previous_iter_soln(controller_test, initialize_dict, X_test, U_test, regions, true_func_obj: GP_DS, ignore_lld=True,
                                                       no_cov_case=False, piecewise=False, return_mu_x=False):
    inverse_cdf_x, inverse_cdf_u = controller_test.inverse_cdf_x, controller_test.inverse_cdf_u
    N = controller_test.N
    debugger_inst = OptiDebugger(controller_test)

    # Casadi squeezes variables that can be squeeze so an (N+1, 1) vector gets squeezed to (N+1) and hence the ndmin is required for the scalar case
    # without affecting planar or higher dim cases
    mu_x = np.array(debugger_inst.get_vals_from_opti_debug('mu_x'), ndmin=2)
    mu_u = np.array(debugger_inst.get_vals_from_opti_debug('mu_u'), ndmin=2)
    mu_x_orig = copy.deepcopy(mu_x)
    # Can only warmstart till penultimate state and input hence pad with 0s.
    initialize_dict['mu_x'] = np.hstack([mu_x[:, 1:], np.zeros((controller_test.n_x, 1))])
    initialize_dict['mu_u'] = np.hstack([mu_u[:, 1:], np.zeros((controller_test.n_u, 1))])

    gp_input_0 = controller_test.delta_input_mask @ mu_x[:, [0]] if controller_test.delta_control_variables == "state_only" else controller_test.delta_input_mask @ np.vstack([mu_x[:, [0]], mu_u[:, [0]]])
    # no_noise = False => the callables within the function generate the true mean and then a white noise vector corresponding to the true environmental stochasticity is added to that true mean.
    # apparently casadi autoconverts stuff to float64 so need to set it back to float32 as that's what our training inputs were.
    type_convd_inp = torch.from_numpy(gp_input_0.astype(np.float32))
    # print(type(type_convd_inp), type_convd_inp.dtype)
    sampled_residual = true_func_obj.generate_outputs(input_arr=type_convd_inp, no_noise=False, return_op=True, noise_verbose=True)
    # print(controller_test.Bd.shape, sampled_residual.shape)
    sampled_ns = controller_test.A @ mu_x[:, 0] + controller_test.B @ mu_u[:, 0] + controller_test.Bd @ sampled_residual.numpy()

    if piecewise:
        if not ignore_lld:
            lld = debugger_inst.get_vals_from_opti_debug('lld')
        hld = debugger_inst.get_vals_from_opti_debug('hld')
        hld = np.hstack([hld[:, 1:], np.zeros([hld.shape[0], 1])])
    initialize_dict['mu_x'][:, [0]] = sampled_ns
    initialize_dict['x_init'] = sampled_ns
    if piecewise:
        if controller_test.delta_control_variables == "state_only":
            delta_ctrl_inp = controller_test.delta_input_mask @ sampled_ns
        else:
            # print(sampled_ns.shape, mu_u[:, [1]].shape)
            delta_ctrl_inp = controller_test.delta_input_mask @ np.vstack([sampled_ns, mu_u[:, [1]]])
        if not ignore_lld:
            try:
                # Recompute hld vec and lld matrix for new timestep 0 based on sampled ns
                lld[0] = simulate_hld_lld(controller_test.delta_constraint_obj, regions, state_dim=1, eps=1e-5,
                                          samples=delta_ctrl_inp, num_samples=1, verbose=False, ret_lld=True, unsqueeze=True)
            except AssertionError:
                print("AssertionError. delta_ctrl_inp:")
                print(delta_ctrl_inp)
        for region_idx, region in enumerate(regions):
            hld[region_idx, 0] = 1 if region.check_satisfaction(delta_ctrl_inp).item() is True else 0
            # Setting hld for the last state is possible if delta_control_variable is only dependent in mu^x_N
            if controller_test.delta_control_variables == "state_only":
                hld[region_idx, -1] = 1 if region.check_satisfaction(mu_x[:, -1]).item() is True else 0
        if not ignore_lld:
            initialize_dict['lld'] = lld
        initialize_dict['hld'] = hld

    # Note while warmstarting, the first column vector is ignored since it will always be X.b_np, U.b_np because of the assumption
    # that we know the first state i.e. x_init with certainty. So we don't need to set it here as it is hardcoded in the OL opt set up
    # in the controller class and the warmstarting ignores that hardcoded vector. But still we set the first column as the print
    # statements in the warmstarting method use the full array despite the first column actually being discarded.
    b_shrunk_x, b_shrunk_u = np.zeros((2*controller_test.n_x, (N+1))), np.zeros((2*controller_test.n_u, N))
    b_shrunk_x[:, [0]] = controller_test.X.b_np
    b_shrunk_u[:, [0]] = controller_test.U.b_np
    # If config opts are such that Sigma_x and Sigma_u are not computed within the O.L. opt loop (i.e. when using approxd shrinking
    # and ignoring variances in cost function) then, need to forward simulate Sigma_x and Sigma_u before generating the b_shrunk vectors
    if not no_cov_case:
        Sigma_x = debugger_inst.get_vals_from_opti_debug("Sigma_x")
        Sigma_u = debugger_inst.get_vals_from_opti_debug("Sigma_u")
    else:
        if piecewise:
            shrunk_gen_inst = FeasibleTraj_Generator(controller_test.U, controller_test, verbose=True)
            Sigma_x, Sigma_u = shrunk_gen_inst.approxd_cov_gen(mu_x[:, 1:], mu_u[:, 1:], hld)
        else:
            shrunk_gen_inst = FeasibleTraj_Generator(controller_test.U, controller_test, verbose=True, piecewise=False)
            Sigma_x, Sigma_u = shrunk_gen_inst.approxd_cov_gen(mu_x[:, 1:], mu_u[:, 1:])
    # Compute Sigma^u_N to generate final bshrunk^u as param for the next time step
    Sigma_u.append(controller_test.compute_sigma_u(Sigma_x[-1]))

    if controller_test.gp_inputs == "state_only" and controller_test.delta_control_variables == "state_only":
        gp_vec = mu_x[:, [N+1]]
        hybrid_means, *hybrid_covs = controller_test.gp_fns(controller_test.input_mask @ gp_vec)

    for k in range(1, N+1):
        if k < N:
            b_shrunk_u[:, [k]] = U_test.b_np - (cs.fabs(U_test.H_np) @ (cs.sqrt(cs.diag(Sigma_u[k]) + controller_test.sqrt_constant) * inverse_cdf_u))
        if k < N or (controller_test.gp_inputs == "state_only" and controller_test.delta_control_variables == "state_only"):
            b_shrunk_x[:, [k]] = X_test.b_np - (cs.fabs(X_test.H_np) @ (cs.sqrt(cs.diag(Sigma_x[k]) + controller_test.sqrt_constant) * inverse_cdf_x))
    # Still need to get the final bshrunk^x vector for the next time step
    initialize_dict['b_shrunk_x'] = b_shrunk_x
    initialize_dict['b_shrunk_u'] = b_shrunk_u

    controller_test.set_initial(**initialize_dict)

    if return_mu_x:
        return mu_x_orig


def generate_feasible_traj_piecewise(gp_fns: Piecewise_GPR_Callback, u_start_limit=np.array([[-2]]).T, u_end_limit=np.array([[2]]).T, custom_U=None, U_big=False,
                                     internal_timeout_interval=10, external_timeout_interval=10, x_init=np.array([[-0.75]]), N=3):
    controller_inst = GPR_test_hardcoded_allfeatsactive_piecewise(gp_fns=gp_fns, minimal_print=True, store_in_file=True, add_monitor=True,
                                                                  hsl_solver=False,
                                                                  early_termination=True, no_solve=True, b_shrunk_opt=True, N=N,
                                                                  U_big=U_big, custom_U=custom_U)
    feasible_found = False
    U_shrunk_inst = box_constraint(u_start_limit, u_end_limit)
    for i in range(external_timeout_interval):
        print(i)
        traj_generator = FeasibleTraj_Generator(U_shrunk_inst, controller_inst, timeout_activate=True, timeout_interval=internal_timeout_interval, verbose=False)
        traj_info = traj_generator.get_traj(x_init)
        # Traj generator is given a counter. If unable to find a solution to any timestep within timeout_interval number of attempts then we restart the generator
        # in the hope that picking alternate inputs earlier in the sequence can help randomly find a feasible solution.
        if traj_info is False:
            feasible_found = False
            print("Failed after %s attempts with the traj generator. Restarting traj generator" % internal_timeout_interval)
            continue
        else:
            feasible_found = True
            x_traj, u_traj, hld_mat, lld_mat, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true, mu_d = traj_info
            break
    if not feasible_found:
        print("Not able to find a feasible solution after %s restarts" % (external_timeout_interval))
        return
    else:
        print("Feasible solution found (lld mat not used though printed)")
        print('x_traj\n %s \n u_traj\n %s \n mu_d\n %s \nhld_mat\n %s \n lld_mat\n %s \n b_shrunk_x\n %s \n b_shrunk_u\n %s \n'
              ' Sigma_x\n %s \n Sigma_u\n %s \n Sigma_d\n %s \n b_shrunk_u_true\n %s \n' % (
            x_traj, u_traj, mu_d, hld_mat, lld_mat, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true))
    warmstart_dict = {'mu_x': x_traj, "mu_u": u_traj, 'hld': hld_mat, 'lld': lld_mat, 'b_shrunk_x': b_shrunk_x, 'b_shrunk_u': b_shrunk_u_true,
                      'mu_d': mu_d}

    print("Checking violation")
    x, p = generate_x_and_p_vecs(x_init, controller_inst.n_x, controller_inst.n_u, controller_inst.N,
                                 x_traj, u_traj, hld_mat, b_shrunk_x, b_shrunk_u_true)
    f = cs.Function('f', [controller_inst.opti.x, controller_inst.opti.p], [controller_inst.opti.g])
    try:
        constr_viol_split = cs.vertsplit(f(x, p), 1)
    except RuntimeError:
        for i in range(controller_inst.opti.nx):
            print(controller_inst.opti.debug.x_describe(i))

    eq_constraint_idxs = []
    ineq_constraint_idxs = []
    constr_viol = 0
    for i in range(controller_inst.opti.ng):
        if '==' in controller_inst.opti.debug.g_describe(i):
            # print(constr_viol_split[i])
            if np.abs(constr_viol_split[i]) >= 1e-5:
                print("Error")
                print(i, controller_inst.opti.debug.g_describe(i))
                print(constr_viol_split[i])
            assert np.abs(constr_viol_split[i]) <= 1e-5, "%s %s %s" % (i, controller_inst.opti.debug.g_describe(i), constr_viol_split[i])
            eq_constraint_idxs.append(i)
            constr_viol += constr_viol_split[i]
        else:
            ineq_constraint_idxs.append(i)
            assert constr_viol_split[i] <= 0
    print(constr_viol)

    return warmstart_dict, f


def generate_feasible_traj_global(gp_fns: MultidimGPR_Callback, u_start_limit=np.array([[-2]]).T, u_end_limit=np.array([[2]]).T, custom_U=None, U_big=False,
                                  internal_timeout_interval=10, external_timeout_interval=10, x_init=np.array([[-0.75]]), N=3):
    controller_inst = GPR_test_hardcoded_allfeatsactive_global(gp_fns=gp_fns, minimal_print=True, store_in_file=True, add_monitor=True,
                                                               hsl_solver=False,
                                                               early_termination=True, no_solve=True, b_shrunk_opt=True, N=N,
                                                               U_big=U_big, custom_U=custom_U)
    feasible_found = False
    U_shrunk_inst = box_constraint(u_start_limit, u_end_limit)
    for i in range(external_timeout_interval):
        print(i)
        traj_generator = FeasibleTraj_Generator(U_shrunk_inst, controller_inst, timeout_activate=True, timeout_interval=internal_timeout_interval, verbose=False,
                                                piecewise=False)
        traj_info = traj_generator.get_traj(x_init)
        # Traj generator is given a counter. If unable to find a solution to any timestep within timeout_interval number of attempts then we restart the generator
        # in the hope that picking alternate inputs earlier in the sequence can help randomly find a feasible solution.
        if traj_info is False:
            feasible_found = False
            print("Failed after %s attempts with the traj generator. Restarting traj generator" % internal_timeout_interval)
            continue
        else:
            feasible_found = True
            print("Feasible solution found (lld mat not used though printed)")
            x_traj, u_traj, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true, mu_d = traj_info
            print('x_traj\n %s \n u_traj\n %s \n mu_d\n %s \n b_shrunk_x\n %s \n b_shrunk_u\n %s \n'
                  ' Sigma_x\n %s \n Sigma_u\n %s \n Sigma_d\n %s \n b_shrunk_u_true\n %s \n' % (
                x_traj, u_traj, mu_d, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true))
            break
    if not feasible_found:
        print("Not able to find a feasible solution after %s restarts" % (external_timeout_interval))
        return

    warmstart_dict = {'mu_x': x_traj, "mu_u": u_traj, 'b_shrunk_x': b_shrunk_x, 'b_shrunk_u': b_shrunk_u_true,
                      'mu_d': mu_d}

    print("Checking violation")
    x, p = generate_x_and_p_vecs(x_init, controller_inst.n_x, controller_inst.n_u, controller_inst.N,
                                 x_traj, u_traj, b_shrunk_x=b_shrunk_x, b_shrunk_u=b_shrunk_u_true, piecewise=False)
    f = cs.Function('f', [controller_inst.opti.x, controller_inst.opti.p], [controller_inst.opti.g])
    constr_viol_split = cs.vertsplit(f(x, p), 1)
    eq_constraint_idxs = []
    ineq_constraint_idxs = []
    constr_viol = 0
    for i in range(controller_inst.opti.ng):
        if '==' in controller_inst.opti.debug.g_describe(i):
            # print(constr_viol_split[i])
            if np.abs(constr_viol_split[i]) >= 1e-5:
                print("Error")
                print(i, controller_inst.opti.debug.g_describe(i))
                print(constr_viol_split[i])
            assert np.abs(constr_viol_split[i]) <= 1e-5, "%s %s %s" % (i, controller_inst.opti.debug.g_describe(i), constr_viol_split[i])
            eq_constraint_idxs.append(i)
            constr_viol += constr_viol_split[i]
        else:
            ineq_constraint_idxs.append(i)
            assert constr_viol_split[i] <= 0
    print(constr_viol)

    return warmstart_dict, f


def GPR_test_hardcoded_allfeatsactive_global(gp_fns, skip_shrinking=False, early_termination=False, only_soln_limit=False, minimal_print=True,
                                             stable_system=True, warmstart=False, warmstart_dict=None, add_monitor=True, store_in_file=True, file_name='',
                                             hsl_solver=False, no_solve=False, b_shrunk_opt=False,
                                             N=2, U_big=True, custom_U=None,
                                             approxd_shrinking=False, closed_loop=False, simulation_length=5,
                                             test_no_lam_p=False, hessian_approximation=False, ignore_cost=False, ignore_variance_costs=False,
                                             true_func_obj: GP_DS=None, with_metrics=True):
    A_test, B_test, Q_test, R_test, s_start_limit, s_end_limit,\
    u_start_limit, u_end_limit, x_init, x0_delim, regions, X_test, U_test = setup_problem(stable_system, custom_U, U_big)
    if closed_loop and with_metrics:
        assert true_func_obj is not None, "Need to pass a GP_DS inst that generated the data samples for closed loop simulation with metrics"

    gp_wrapped = gp_fns

    print_level = 0 if minimal_print else 5

    opts = construct_config_opts(print_level, add_monitor, early_termination, only_soln_limit, hsl_solver,
                                 closed_loop, test_no_lam_p, warmstart, hessian_approximation=hessian_approximation, piecewise=False)

    controller_test = Hybrid_GPMPC(A=A_test, B=B_test, Q=Q_test, R=R_test, horizon_length=N,
                                   gp_fns=gp_wrapped,  # Using gp model wrapped in casadi callback
                                   X=X_test, U=U_test, satisfaction_prob=0.4, regions=regions,
                                   skip_shrinking=skip_shrinking, addn_solver_opts=opts,
                                   add_b_shrunk_opt=b_shrunk_opt, piecewise=False, add_b_shrunk_param=approxd_shrinking,
                                   ignore_cost=ignore_cost, ignore_variance_costs=ignore_variance_costs, solver='ipopt')
    controller_test.display_configs()
    controller_test.setup_OL_optimization()

    if no_solve:
        controller_test.set_initial(**{'x_init': x_init})
        return controller_test
    else:
        initialize_dict = initialize_global(x_init, warmstart, b_shrunk_opt, warmstart_dict, controller_test)

    if not closed_loop:
        run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict)
    else:
        run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict,
                        simulation_length=simulation_length, true_func_obj=true_func_obj, piecewise=False)

    return controller_test, X_test, U_test


class FeasibleTraj_Generator:
    def __init__(self, U_shrunk: box_constraint, GP_MPC_inst, verbose=False, max_iter=5,
                 timeout_activate=False, timeout_interval=10, piecewise=True):
        """
        Parameters
        ----------
        x_init initial starting state assumed to be known with certainty
        U input constraint set
        N MPC O.L. horizon
        GP_MPC_inst: GP_MPC_Global instance with appropriate system matrices passed during instantiations

        Returns
        -------
        A trajectory that obeys the system dynamics
        """
        self.A, self.B, self.Bd, self.gp_fns, self.gp_input_type, self.input_mask, self.N = GP_MPC_inst.get_info_for_traj_gen()
        self.n_x, self.n_u = GP_MPC_inst.n_x, GP_MPC_inst.n_u
        self.controller = GP_MPC_inst
        self.piecewise = piecewise
        if piecewise:
            self.regions = GP_MPC_inst.regions
            self.delta_control_variables, self.delta_input_mask = GP_MPC_inst.delta_control_variables, GP_MPC_inst.delta_input_mask
            self.delta_constraint_obj = GP_MPC_inst.delta_constraint_obj
        self.gp_input_type, self.gp_input_mask = GP_MPC_inst.gp_inputs, GP_MPC_inst.input_mask
        # Bad naming sorry :3
        try:
            self.res_dim = GP_MPC_inst.gp_fns.output_dims
        except AttributeError:
            self.res_dim = GP_MPC_inst.gp_fns.output_dim
        if piecewise:
            self.get_mu_d = GP_MPC_inst.get_mu_d
            self.get_Sigma_d = GP_MPC_inst.get_Sigma_d
        self.computesigma_wrapped = GP_MPC_inst.computesigma_wrapped
        self.compute_sigx_callback = GP_MPC_inst.compute_sigx_callback
        self.compute_sigma_u = GP_MPC_inst.compute_sigma_u
        self.verbose = verbose
        self.K = GP_MPC_inst.K
        self.affine_transform = GP_MPC_inst.affine_transform
        self.max_iter = max_iter
        self.U_shrunk = U_shrunk
        self.timeout_activate = timeout_activate
        self.timeout_interval = timeout_interval
        self.U_orig = GP_MPC_inst.U
        self.sqrt_const = GP_MPC_inst.sqrt_constant
        self.X = GP_MPC_inst.X
        self.inverse_cdf_x, self.inverse_cdf_u = GP_MPC_inst.inverse_cdf_x, GP_MPC_inst.inverse_cdf_u

    def get_traj(self, x_init):
        x_traj = np.zeros(shape=(self.n_x, self.N+1))
        x_traj[:, [0]] = x_init
        u_traj = self.U_shrunk.get_random_vectors(num_samples=self.N)
        gp_inputs = np.zeros(shape=(np.linalg.matrix_rank(self.input_mask), self.N))
        mu_d = np.zeros(shape=(self.res_dim, self.N))
        Sigma_d = [np.zeros((self.res_dim, self.res_dim)) for _ in range(self.N)]
        Sigma_x = [np.zeros((self.n_x, self.n_x)) for _ in range(self.N+1)]
        Sigma_u = [np.zeros((self.n_u, self.n_u)) for _ in range(self.N)]
        Sigma = []
        b_shrunk_x, b_shrunk_u, b_shrunk_u_true = np.zeros((2*self.n_x, (self.N+1))), np.zeros((2*self.n_u, self.N)),\
                                                  np.zeros((2*self.n_u, self.N))
        b_shrunk_x[:, [0]] = self.X.b_np
        b_shrunk_u[:, [0]] = self.U_shrunk.b_np
        b_shrunk_u_true[:, [0]] = self.U_orig.b_np
        temp_U_shrunk = self.U_shrunk
        if self.piecewise:
            num_delta_inp = np.linalg.matrix_rank(self.delta_input_mask)
            delta_controls = cs.DM.zeros(num_delta_inp, self.N)
            hld_mat = cs.DM.zeros(len(self.regions), self.N)
            lld_mat = [cs.DM.zeros(2*num_delta_inp, len(self.regions)) for _ in range(self.N)]

        for i in range(self.N):
            if self.verbose:
                print("Timestep: %s" % i)
                print()
            # Sigma_u[i] only gets computed after Sigma^x_{i} was generated in the previous iteration from a feasible
            # mu^x_{i+1}
            Sigma_u_temp = self.K @ Sigma_x[i] @ self.K.T
            Sigma_u[i] = self.compute_sigma_u(Sigma_x[i])
            assert (np.array(Sigma_u[i]) == Sigma_u_temp).all(), "Manual Sigma^u_{k} doesn't match with callback"
            if i > 0:
                b_shrunk_u[:, [i]] = self.U_shrunk.b_np - (np.absolute(self.U_shrunk.H_np) @ (np.sqrt(np.diag(Sigma_u[i]) + self.sqrt_const)[:, None] * self.inverse_cdf_u))
                b_shrunk_u_temp = self.U_shrunk.b_np - (cs.fabs(self.U_shrunk.H_np) @ (cs.sqrt(cs.diag(Sigma_u[i]) + self.sqrt_const) * self.inverse_cdf_u))
                b_shrunk_u_true[:, [i]] = self.U_orig.b_np - (np.absolute(self.U_orig.H_np) @ (np.sqrt(np.diag(Sigma_u[i]) + self.sqrt_const)[:, None] * self.inverse_cdf_u))
                # b_shrunk_u_true[:, [i]] = self.U_orig.b_np - (cs.fabs(self.U_orig.H_np) @ (cs.sqrt(cs.diag(Sigma_u[i]) + sqrt_constant) * self.inverse_cdf_u))
                assert (np.isclose(np.array(b_shrunk_u_temp).squeeze(), np.array(b_shrunk_u[:, i]), atol=1e-3).all()), "cs and np don't match %s %s %s %s" % (b_shrunk_u_temp, b_shrunk_u[:, i], b_shrunk_u_temp.shape, b_shrunk_u[:, i].shape)
                temp_U_shrunk = box_constraint_direct(H_np=self.U_shrunk.H_np, b_np=b_shrunk_u[:, [i]])
            timeout_counter = -1
            valid_input = False
            while not valid_input:
                timeout_counter += 1
                if timeout_counter == self.timeout_interval:
                    return False
                # u_temp will be feasible if the mu^x_{k+1} it generates is within the temp_shrunk_X generated after
                # computing Sigma^x_{k+1}
                u_temp_k = temp_U_shrunk.get_random_vectors(num_samples=1)
                u_traj[:, [i]] = u_temp_k
                # Generating hld vector and lld array elements
                joint_vec = np.vstack([x_traj[:, [i]], u_traj[:, [i]]])
                if self.piecewise:
                    control_vec = x_traj[:, [i]] if self.delta_control_variables == "state_only" else joint_vec
                    delta_controls[:, [i]] = self.delta_input_mask @ control_vec
                    if self.verbose:
                        display(Math(r'\delta_{} = {} = {}'.format('{ctrl, %s}' % i,
                                                                   "\,\, \,\,".join(np2bmatrix([self.delta_input_mask, control_vec], return_list=True)),
                                                                   np2bmatrix([delta_controls[:, [i]]]))))
                    # Typically to get just the hld vectors for the current timestep, since the vectors we take as input are numeric, we can use the
                    # check_satisfaction method of the box_constraint instance. But here, we want to generate the lld matrix for the current timestep
                    # too so we can pass that for warmstart or for checking constraint violation for a true feasible solution. Hence we use the
                    # simulate_hld_lld function. Note that self.delta_constraint_obj takes the role of X_test in the simulate function but it is not
                    # actually X_test since when the delta_control input is joint state and input, we need to generate a modified constraint object which
                    # will supply the bounds which is the delta_constraint_obj generated within the GP_MPC controller class.
                    lld_mat[i] = simulate_hld_lld(self.delta_constraint_obj, self.regions, state_dim=self.n_x, eps=1e-5,
                                                  samples=delta_controls[:, [i]], verbose=False, ret_lld=True, unsqueeze=True)
                    # print(lld_mat[i])
                    for region_idx, region in enumerate(self.regions):
                        hld_mat[region_idx, i] = 1 if region.check_satisfaction(delta_controls[:, i].T).item() is True else 0
                    if self.verbose:
                        print("LLD Mat")
                        display(Math(r'\delta_{} = {}'.format("{:, :, %s}" % i, np2bmatrix([lld_mat[i]]))))
                        print("HLD Row")
                        display(Math(r'\delta_{} = {}'.format("{:, %s}" % i, np2bmatrix([hld_mat[:, [i]]]))))

                # Getting output means and covs from piecewise gp class for all regions.
                gp_vec = x_traj[:, [i]] if self.gp_input_type == "state_only" else joint_vec
                gp_inputs[:, [i]] = self.input_mask @ gp_vec
                if self.piecewise:
                    hybrid_means, *hybrid_covs = self.gp_fns(gp_inputs[:, i])
                    if self.verbose:
                        display(Math(r'g_{} = {} = {}'.format('{inp, %s}' % i,
                                                              "\,\, \,\,".join(np2bmatrix([self.gp_input_mask, gp_vec], return_list=True)),
                                                              np2bmatrix([gp_inputs[:, [i]]]))))
                        print("Region-wise Means")
                        display(Math(r'{}'.format(np2bmatrix([hybrid_means]))))
                        print("Region-wise Covariances")
                        for region_idx in range(len(self.regions)):
                            display(Math(r'Region {}'.format(region_idx+1, np2bmatrix([hybrid_covs[region_idx]]))))

                    # Applying deltas to select correct mean and cov.
                    mu_d[:, [i]] = self.get_mu_d(hybrid_means, hld_mat[:, [i]])
                    Sigma_d[i] = self.get_Sigma_d(hld_mat[:, [i]], *hybrid_covs)
                    if self.verbose:
                        print("Selected Mean")
                        display(Math(r'\mu^d_{} = {}'.format(i, np2bmatrix([mu_d[:, [i]]]))))
                        print("Selected Cov")
                        display(Math(r'\Sigma^d_{} = {}'.format(i, np2bmatrix([Sigma_d[i]]))))
                    if self.verbose and i == 0:
                        print(x_traj[:, [i]], "\n", u_traj[:, [i]], "\n", gp_inputs[:, [i]])
                        print(self.A, x_traj[:, [i]], self.B, u_traj[:, [i]], self.Bd, self.gp_fns(gp_inputs[:, [i]]), sep="\n")
                else:
                    mu_d[:, [i]], Sigma_d[i] = self.gp_fns(gp_inputs[:, i])

                # Final dynamics equations for x and Sigma x.
                # Dynamics for x
                x_traj[:, [i+1]] = self.A @ x_traj[:, [i]] + self.B @ u_traj[:, [i]] + self.Bd @ mu_d[:, [i]]
                # Dynamics for Sigma x
                Sigma_i = self.computesigma_wrapped(Sigma_x[i], Sigma_u[i], Sigma_d[i])
                Sigma_x[i+1] = self.affine_transform @ Sigma_i @ self.affine_transform.T
                Sigma_x_temp = self.compute_sigx_callback(Sigma_i)
                Sigma_x_temp2 = np.hstack([self.A, self.B, self.Bd]) @ Sigma_i @ np.vstack([self.A.T, self.B.T, self.Bd.T])
                assert (np.array(Sigma_x[i+1]) == Sigma_x_temp).all(), "Manual Sigma^x_{k+1} doesn't match with callback"
                assert (np.array(Sigma_x[i+1]) == Sigma_x_temp2).all(), "Manual Sigma^x_{k+1} 2 doesn't match with callback"

                sqrt_constant = 1e-4
                b_shrunk_x[:, [i+1]] = self.X.b_np - (np.absolute(self.X.H_np) @ (np.sqrt(np.diag(Sigma_x[i+1]) + sqrt_constant)[:, None] * self.inverse_cdf_x))
                b_shrunk_x_temp = self.X.b_np - (cs.fabs(self.X.H_np) @ (cs.sqrt(cs.diag(Sigma_x[i+1]) + sqrt_constant) * self.inverse_cdf_x))
                assert (np.isclose(np.array(b_shrunk_x_temp).squeeze(), np.array(b_shrunk_x[:, i+1]), atol=1e-3).all()), "cs and np don't match %s %s %s %s" % (b_shrunk_x_temp, b_shrunk_x[:, i+1], b_shrunk_x_temp.shape, b_shrunk_x[:, i+1].shape)

                temp_X_shrunk = box_constraint_direct(H_np=self.X.H_np, b_np=b_shrunk_x[:, i+1])
                if temp_X_shrunk.check_satisfaction(x_traj[:, i+1]):
                    valid_input = True
                #     print('Passed')
                #     print('Sigma^x_{i+1}')
                #     print(Sigma_x[i+1])
                # else:
                #     print('Failed')

            Sigma.append(Sigma_i)

        if self.piecewise:
            return x_traj, u_traj, hld_mat, lld_mat, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true, mu_d
        else:
            return x_traj, u_traj, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true, mu_d

    def fwd_sim_w_opt_var_assgt(self, mu_x, mu_u, hld):
        assert self.piecewise, "This function was only written to debug nans for the piecewise case. Not intended for use with global " \
                               "case currently."
        x_traj = mu_x
        u_traj = mu_u
        hld_mat = hld

        gp_inputs = np.zeros(shape=(np.linalg.matrix_rank(self.input_mask), self.N))
        mu_d = np.zeros(shape=(self.res_dim, self.N))
        Sigma_d = [np.zeros((self.res_dim, self.res_dim)) for _ in range(self.N)]
        Sigma_x = [np.zeros((self.n_x, self.n_x)) for _ in range(self.N+1)]
        Sigma_u = [np.zeros((self.n_u, self.n_u)) for _ in range(self.N)]
        Sigma = []

        for i in range(self.N):
            joint_vec = np.vstack([x_traj[:, [i]], u_traj[:, [i]]])
            # Getting output means and covs from piecewise gp class for all regions.
            gp_vec = x_traj[:, [i]] if self.gp_input_type == "state_only" else joint_vec
            gp_inputs[:, [i]] = self.input_mask @ gp_vec
            hybrid_means, *hybrid_covs = self.gp_fns(gp_inputs[:, i])
            mu_d[:, [i]] = self.get_mu_d(hybrid_means, hld_mat[:, [i]])
            Sigma_d[i] = self.get_Sigma_d(hld_mat[:, [i]], *hybrid_covs)
            if self.verbose:
                print("Timestep: %s" % i)
                display(Math(r'g_{} = {} = {}'.format('{inp, %s}' % i,
                                                      "\,\, \,\,".join(np2bmatrix([self.gp_input_mask, gp_vec], return_list=True)),
                                                      np2bmatrix([gp_inputs[:, [i]]]))))
                print("Region-wise Means")
                display(Math(r'{}'.format(np2bmatrix([hybrid_means]))))
                print("Region-wise Covariances")
                for region_idx in range(len(self.regions)):
                    display(Math(r'Region {}: {}'.format(region_idx+1, np2bmatrix([hybrid_covs[region_idx]]))))

                print("Summed Mean")
                display(Math(r'\mu^d_{} = {}'.format(i, np2bmatrix([mu_d[:, [i]]]))))
                print("Summed Cov")
                display(Math(r'\Sigma^d_{} = {}'.format(i, np2bmatrix([Sigma_d[i]]))))

            Sigma_u[i] = self.compute_sigma_u(Sigma_x[i])
            Sigma_i = self.computesigma_wrapped(Sigma_x[i], Sigma_u[i], Sigma_d[i])
            Sigma.append(Sigma_i)
            Sigma_x[i+1] = self.affine_transform @ Sigma_i @ self.affine_transform.T
            print(Sigma_x[i+1] + np.eye(self.n_x)*self.sqrt_const)

            if self.verbose:
                display(Math(r'\Sigma^{}_{} = {}\,\,;\,\, '
                             r'\Sigma^{}_{} = {}\,\,;\,\, '
                             r'\Sigma^{}_{} = {}\,\,;\,\, '
                             r'\Sigma^_{} = {}\,\,;\,\, '.format("{x}", i, np2bmatrix([Sigma_x[i]]),
                                                                 "{u}", i, np2bmatrix([Sigma_u[i]]),
                                                                 "{d}", i, np2bmatrix([Sigma_d[i]]),
                                                                 i, np2bmatrix([Sigma[i]]))))

    def approxd_cov_gen(self, mu_x, mu_u, hld_mat):
        assert mu_x.shape[0] == self.N, "mu_x array passed as input should only be from timestep 1 to N of the O.L. opt solution"
        assert mu_u.shape[0] == self.N-1, "mu_u array passed as input should only be from timestep 1 to N-1 of the O.L. opt solution"
        Sigma_d = [np.zeros((self.res_dim, self.res_dim)) for _ in range(self.N)]
        Sigma_x = [np.zeros((self.n_x, self.n_x)) for _ in range(self.N+1)]
        Sigma_u = [np.zeros((self.n_u, self.n_u)) for _ in range(self.N)]
        Sigma = []
        for i in range(self.N):
            Sigma_u[i] = self.compute_sigma_u(Sigma_x[i])
            joint_vec = np.vstack([mu_x[:, [i]], mu_u[:, [i]]])
            gp_vec = mu_x[:, [i]] if self.gp_input_type == "state_only" else joint_vec
            if self.piecewise:
                # Getting output means and covs from piecewise gp class for all regions.
                hybrid_means, *hybrid_covs = self.gp_fns(self.input_mask @ gp_vec)
                Sigma_d[i] = self.get_Sigma_d(hld_mat[:, [i]], *hybrid_covs)
            else:
                _, Sigma_d[i] = self.gp_fns(self.input_mask @ gp_vec)
            Sigma_i = self.computesigma_wrapped(Sigma_x[i], Sigma_u[i], Sigma_d[i])
            Sigma_x[i+1] = self.affine_transform @ Sigma_i @ self.affine_transform.T
            Sigma.append(Sigma_i)
        return Sigma_x, Sigma_u


def shape_into_arr(inp_arr, orig_list, delim, horizon):
    for k in range(horizon):
        inp_arr[:, k] = np.array(orig_list[k*delim:(k+1)*delim])


def generate_x_and_p_vecs(x_init, n_x, n_u, N, x_traj, u_traj, hld_mat=None, b_shrunk_x=None, b_shrunk_u=None, piecewise=True):
    if piecewise:
        assert hld_mat is not None, "Need to pass hld mat to use this function in the piecewise case."
    x_des_vec, u_des_vec = np.zeros(n_x*(N+1)), np.zeros(n_u*N)
    p = np.concatenate((x_init.flatten(order='F'), x_des_vec, u_des_vec))
    x = np.array([])
    x = np.concatenate((x, x_traj.flatten(order='F')))
    print(x.shape)
    x = np.concatenate((x, u_traj.flatten(order='F')))
    print(x.shape)
    if piecewise:
        x = np.concatenate((x, np.array(hld_mat).flatten(order='F')))
        print(x.shape)
    if b_shrunk_x is not None:
        # the 1: is to ignore first shrunk vector which, based on assumption that there is no uncertainty
        # of first state means that b_shrunk_x[:, 0] always equal X.b and so we dont need that constrained
        # in the optimization even to debug.
        x = np.concatenate((x, b_shrunk_x[:, 1:].flatten(order='F')))
        print(x.shape)
        x = np.concatenate((x, b_shrunk_u[:, 1:].flatten(order='F')))
        print(x.shape)
    return x, p

