import copy
import os
import warnings
import pprint
import contextlib
import executing.executing
import numpy as np
import scipy
import torch
from scipy import stats
import casadi as cs
import datetime as dt
import sys
import traceback
import dill as pkl
import matplotlib.pyplot as plt

from ds_utils import box_constraint, box_constraint_direct, test_1d_op_2d_inp_allfeats, combine_box, generate_fine_grid, GP_DS, dir_exist_or_create, save_subplot
from .utils import Piecewise_GPR_Callback, MultidimGPR_Callback, GPR_Callback, Sigma_u_Callback, Sigma_x_dynamics_Callback_LTI
from .utils import hybrid_res_covar, computeSigma_meaneq, \
    np2bmatrix, simulate_hld_lld, setup_terminal_costs, get_inv_cdf
from models import piecewise_train_test, train_test
from .controller_debugging import OptiDebugger, retrieve_controller_results_piecewise, retrieve_controller_results

from IPython.display import display, Math, Markdown
from collections import defaultdict


# Hardcoded for sigma_x and sigma_u not to be opt vars
class GP_MPC_Alt_Delta:
    def __init__(self, A, B, Bd, Q, R, horizon_length, gp_fns,
                 X: box_constraint, U: box_constraint, satisfaction_prob,
                 n_x, n_u, n_d, gp_inputs, gp_input_mask, delta_control_variables, delta_input_mask,
                 regions=None, solver='bonmin', piecewise=True,
                 skip_shrinking=False, addn_solver_opts=None, sqrt_const=1e-4,
                 add_b_shrunk_opt=False, add_b_shrunk_param=False,
                 ignore_cost=False, ignore_variance_costs=False, relaxed=False,
                 add_delta_tol=False, test_softplus=False, skip_feedback=False,
                 ignore_init_constr_check=False):
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
        self.Bd = Bd
        self.N = horizon_length
        self.n_x, self.n_u, self.n_d = n_x, n_u, n_d
        # Feedback matrix
        self.skip_feedback = skip_feedback
        self.K = self._setup_terminal()
        if self.skip_feedback:
            self.K = np.zeros([self.n_u, self.n_x])
        self.X, self.U = X, U

        self.sqrt_constant = sqrt_const
        self.ignore_cost = ignore_cost
        self.ignore_variance_costs = ignore_variance_costs
        self.add_delta_tol = add_delta_tol
        self.test_softplus = test_softplus
        self.ignore_init_constr_check = ignore_init_constr_check

        self.gp_approx = 'mean_eq'
        self.gp_inputs = gp_inputs
        # 2-D state 1-D input. Gp input is 2nd state and input
        self.input_mask = gp_input_mask
        self.delta_control_variables = delta_control_variables
        # Both state variables control the region
        self.delta_input_mask = delta_input_mask
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
        inverse_cdf_i = get_inv_cdf(n_i, self.satisfaction_prob)
        return inverse_cdf_i

    def _setup_terminal(self):
        # As in the paper, choose Q and R matrices for the LQR solution to be the same matrices
        # in the cost function optimization
        K, self.P = setup_terminal_costs(self.A, self.B, self.Q, self.R)
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
            if kwargs.get('verbose', True):
                print("Setting shrunk vector parameters")
            b_shrunk_x_init, b_shrunk_u_init = kwargs.get('b_shrunk_x'), kwargs.get('b_shrunk_u')
            if kwargs.get('verbose', True):
                print("b_shrunk_x: \n%s \n b_shrunk_u: \n%s \n" % (b_shrunk_x_init, b_shrunk_u_init))

        # Warmstarting opt vars
        if kwargs.get('mu_x', None) is not None:
            mu_x_init, mu_u_init = kwargs.get('mu_x'), kwargs.get('mu_u')
            if kwargs.get('verbose', True):
                print("Warmstarting with feasible trajectory (mu_x, mu_u, hld)")
                print("mu_x: \n%s \n mu_u: \n%s \n" % (mu_x_init, mu_u_init))
            self.opti.set_initial(self.opti_dict["mu_x"], mu_x_init)
            self.opti.set_initial(self.opti_dict["mu_u"], mu_u_init)
            if self.piecewise:
                hld_init = kwargs.get('hld')
                if kwargs.get('verbose', True):
                    print("hld: \n%s \n" % (hld_init))
                self.opti.set_initial(self.opti_dict["hld"], hld_init)
            if self.add_b_shrunk_opt:
                if kwargs.get('verbose', True):
                    print("Warmstarting with feasible trajectory (b_shrunk_x, b_shrunk_u)")
                b_shrunk_x_init, b_shrunk_u_init = kwargs.get('b_shrunk_x'), kwargs.get('b_shrunk_u')
                if kwargs.get('verbose', True):
                    print("b_shrunk_x: \n%s \n b_shrunk_u: \n%s \n" % (b_shrunk_x_init, b_shrunk_u_init))
                self.opti.set_initial(self.opti_dict["b_shrunk_x"][:, 1:], b_shrunk_x_init[:, 1:])
                self.opti.set_initial(self.opti_dict["b_shrunk_u"][:, 1:], b_shrunk_u_init[:, 1:])

    def display_configs(self):
        print("Time of running test: %s" % (dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
        print("Nominal system matrices")
        print("A: %s \n B: %s \n Q: %s\n R: %s\n N: %s \n" % (self.A, self.B, self.Q, self.R, self.N))
        print("Feedback matrix, Terminal cost matrix")
        print("K: %s \n P: %s \n" % (self.K, self.P))
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
        if kwargs.get('verbose', True):
            sol = self.opti.solve()
        else:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
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
        # Constraint for the initial state to be within the unshrunk state constraint set. We add a config option to neglect this constraint while working to gather
        # data for the constraint violation test.
        if not self.ignore_init_constr_check:
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
        elif self.add_b_shrunk_param:
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
            Sigma_x_k = Sigma_x[k] if not self.ignore_variance_costs else None
            Sigma_u_k = Sigma_u[k] if not self.ignore_variance_costs else None
            cost += self.cost_fn(mu_i_x=mu_x[:, k], mu_i_u=mu_u[:, k], Sigma_x_i=Sigma_x_k, Sigma_u_i=Sigma_u_k,
                                 x_desired=x_desired, u_desired=u_desired, idx=k)
        # Terminal cost
        Sigma_x_N = Sigma_x[-1] if not self.ignore_variance_costs else None
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


def setup_problem_basic(stable_system=False, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = {}
    if stable_system:
        A_test = np.array([[0.85, 0.5], [0, -0.75]])
    else:
        A_test = np.array([[1.25, 0.5], [0, -1.75]])
    problem_setup_dict["A_test"] = A_test
    B_test = np.array([[1.25], [2]])
    Bd_test = np.array([[1], [0]])
    Q_test, R_test = np.eye(2)*0.75, np.eye(1)*0.05
    problem_setup_dict.update({"B_test": B_test, "Bd_test": Bd_test, "Q_test": Q_test, "R_test": R_test})
    n_x, n_u, n_d = 2, 1, 1
    problem_setup_dict.update({"n_x": n_x, "n_u": n_u, "n_d": n_d})
    s_start_limit, s_end_limit = np.array([[-2, -2]]).T, np.array([[2, 2]]).T
    if custom_U:
        u_start_limit, u_end_limit = custom_U
    else:
        if U_big:
            u_start_limit, u_end_limit = np.array([[-50]]).T, np.array([[50]]).T
        else:
            u_start_limit, u_end_limit = np.array([[-3]]).T, np.array([[3]]).T
    # x_init = np.array([[-0.75], [0.501]])
    x_init = np.array([[1.5], [1.5]])
    problem_setup_dict.update({"s_start_limit": s_start_limit, "s_end_limit": s_end_limit,
                               "u_start_limit": u_start_limit, "u_end_limit": u_end_limit, "x_init": x_init})

    x0_delim, x1_delim = -0.5, 0.5
    regions = planar_region_gen_and_viz(viz=False, s_start_limit=s_start_limit, s_end_limit=s_end_limit,
                                        x0_delim=x0_delim, x1_delim=x1_delim)
    X_test, U_test = box_constraint(s_start_limit, s_end_limit), box_constraint(u_start_limit, u_end_limit)
    problem_setup_dict.update({"x0_delim": x0_delim, "x1_delim": x1_delim,
                               "regions": regions, "X_test": X_test, "U_test": U_test})

    gp_inputs = 'state_input'
    # 2-D state 1-D input. Gp input is 1st and 2nd state variables
    gp_input_mask = np.array([[1, 0, 0], [0, 1, 0]])
    delta_control_variables = 'state_input'
    # Both state variables control the region
    delta_input_mask = np.array([[1, 0, 0], [0, 1, 0]])
    problem_setup_dict.update({"gp_inputs": gp_inputs, "gp_input_mask": gp_input_mask,
                               "delta_control_variables": delta_control_variables, "delta_input_mask": delta_input_mask})

    problem_setup_dict.update({"satisfaction_prob": satisfaction_prob})

    return problem_setup_dict


def setup_constr_viol_test(stable_system=True, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = setup_problem_basic(stable_system=stable_system, custom_U=custom_U,
                                             U_big=U_big, satisfaction_prob=satisfaction_prob)
    A_test = np.array([[0.85, 0], [0, 0.8]])
    B_test = np.array([[1.2], [1.2]])
    problem_setup_dict.update({"B_test": B_test, "A_test": A_test})
    Q_test = np.eye(2)*3
    problem_setup_dict["Q_test"] = Q_test
    s_start_limit, s_end_limit = problem_setup_dict["s_start_limit"], problem_setup_dict["s_end_limit"]
    x0_delim, x1_delim = problem_setup_dict["x0_delim"], problem_setup_dict["x1_delim"]
    ext_start_limit, ext_end_limit = s_start_limit-1, s_end_limit+1
    regions = planar_region_gen_and_viz(viz=False, s_start_limit=ext_start_limit, s_end_limit=ext_end_limit,
                                        x0_delim=x0_delim, x1_delim=x1_delim)
    problem_setup_dict["regions"] = regions
    return problem_setup_dict


def setup_smaller_ctrl_unstable(stable_system=False, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = setup_problem_basic(stable_system=stable_system, custom_U=custom_U,
                                             U_big=U_big, satisfaction_prob=satisfaction_prob)
    u_start_limit, u_end_limit = np.array([[-0.75]]).T, np.array([[0.75]]).T
    A_test = np.array([[1.25, 0], [0, 1.1]])
    B_test = np.array([[1.25, 1.25]]).T
    problem_setup_dict.update({"A_test": A_test, "B_test": B_test})
    problem_setup_dict.update({"u_start_limit": u_start_limit, "u_end_limit": u_end_limit})
    U_test = box_constraint(u_start_limit, u_end_limit)
    problem_setup_dict.update({"U_test": U_test})
    R_test = np.eye(1)*0.45
    problem_setup_dict.update({"R_test": R_test})

    return problem_setup_dict


def setup_smaller_ctrl_stable(stable_system=True, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = setup_problem_basic(stable_system=stable_system, custom_U=custom_U,
                                             U_big=U_big, satisfaction_prob=satisfaction_prob)
    u_start_limit, u_end_limit = np.array([[-0.25]]).T, np.array([[0.25]]).T
    A_test = np.array([[0.95, 0], [0, 0.85]])
    B_test = np.array([[0.85], [0.95]])
    problem_setup_dict.update({"A_test": A_test, "B_test": B_test})
    problem_setup_dict.update({"u_start_limit": u_start_limit, "u_end_limit": u_end_limit})
    U_test = box_constraint(u_start_limit, u_end_limit)
    # print(U_test)
    problem_setup_dict.update({"U_test": U_test})
    Q_test, R_test = np.eye(2)*2, np.eye(1)*0.45
    problem_setup_dict.update({"R_test": R_test, "Q_test": Q_test})

    return problem_setup_dict


def cost_comp_stable(stable_system=True, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = setup_problem_basic(stable_system=stable_system, custom_U=custom_U,
                                             U_big=U_big, satisfaction_prob=satisfaction_prob)
    u_start_limit, u_end_limit = np.array([[-0.4]]).T, np.array([[0.4]]).T
    A_test = np.array([[0.95, 0], [0, 0.95]])
    B_test = np.array([[0.5], [0.5]])
    problem_setup_dict.update({"A_test": A_test, "B_test": B_test})
    problem_setup_dict.update({"u_start_limit": u_start_limit, "u_end_limit": u_end_limit})
    U_test = box_constraint(u_start_limit, u_end_limit)
    # print(U_test)
    problem_setup_dict.update({"U_test": U_test})
    Q_test, R_test = np.eye(2)*2, np.eye(1)*0.05
    problem_setup_dict.update({"R_test": R_test, "Q_test": Q_test})

    return problem_setup_dict


def cost_comp_boundary(stable_system=True, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = setup_problem_basic(stable_system=stable_system, custom_U=custom_U,
                                             U_big=U_big, satisfaction_prob=satisfaction_prob)
    u_start_limit, u_end_limit = np.array([[-0.6]]).T, np.array([[0.6]]).T
    A_test = np.array([[0.95, 0], [0, 0.95]])
    B_test = np.array([[0.75], [0.75]])
    problem_setup_dict.update({"A_test": A_test, "B_test": B_test})
    problem_setup_dict.update({"u_start_limit": u_start_limit, "u_end_limit": u_end_limit})
    U_test = box_constraint(u_start_limit, u_end_limit)
    # print(U_test)
    problem_setup_dict.update({"U_test": U_test})
    Q_test, R_test = np.eye(2)*2, np.eye(1)*0.05
    problem_setup_dict.update({"R_test": R_test, "Q_test": Q_test})

    s_start_limit, s_end_limit = problem_setup_dict["s_start_limit"], problem_setup_dict["s_end_limit"]
    x0_delim, x1_delim = problem_setup_dict["x0_delim"], problem_setup_dict["x1_delim"]
    ext_start_limit, ext_end_limit = s_start_limit-1, s_end_limit+1
    regions = planar_region_gen_and_viz(viz=False, s_start_limit=ext_start_limit, s_end_limit=ext_end_limit,
                                        x0_delim=x0_delim, x1_delim=x1_delim)
    problem_setup_dict["regions"] = regions

    return problem_setup_dict


def setup_2d_input(stable_system=True, custom_U=False, U_big=False, satisfaction_prob=0.8):
    problem_setup_dict = setup_problem_basic(stable_system=stable_system, custom_U=custom_U,
                                             U_big=U_big, satisfaction_prob=satisfaction_prob)
    u_start_limit, u_end_limit = np.array([[-0.6, -0.6]]).T, np.array([[0.6, 0.6]]).T
    A_test = np.array([[0.95, 0], [0, 0.95]])
    B_test = np.array([[0.75, 0], [0, 0.75]])
    problem_setup_dict.update({"A_test": A_test, "B_test": B_test})
    problem_setup_dict.update({"u_start_limit": u_start_limit, "u_end_limit": u_end_limit})
    U_test = box_constraint(u_start_limit, u_end_limit)
    # print(U_test)
    problem_setup_dict.update({"U_test": U_test})
    Q_test, R_test = np.eye(2)*2, np.eye(2)*0.05
    problem_setup_dict.update({"R_test": R_test, "Q_test": Q_test})

    s_start_limit, s_end_limit = problem_setup_dict["s_start_limit"], problem_setup_dict["s_end_limit"]
    x0_delim, x1_delim = problem_setup_dict["x0_delim"], problem_setup_dict["x1_delim"]
    ext_start_limit, ext_end_limit = s_start_limit-1, s_end_limit+1
    regions = planar_region_gen_and_viz(viz=False, s_start_limit=ext_start_limit, s_end_limit=ext_end_limit,
                                        x0_delim=x0_delim, x1_delim=x1_delim)
    problem_setup_dict["regions"] = regions

    gp_input_mask = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    delta_control_variables = 'state_input'
    # Both state variables control the region
    delta_input_mask = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    problem_setup_dict.update({"gp_input_mask": gp_input_mask, "delta_input_mask": delta_input_mask})

    problem_setup_dict.update({"n_u": 2})

    return problem_setup_dict


valid_setups = [setup_problem_basic, setup_smaller_ctrl_unstable, setup_smaller_ctrl_stable, setup_constr_viol_test,
                cost_comp_stable, cost_comp_boundary, setup_2d_input]


def planar_region_gen_and_viz(viz=True, s_start_limit=np.array([[-2, -2]]).T, s_end_limit=np.array([[2, 2]]).T,
                              x0_delim=-0.5, x1_delim=0.5):
    # r1 spans full x1 but x0 \in [-2, -0.5]
    r1_start, r1_end = np.array([[s_start_limit[0, :].item(), s_start_limit[1, :].item()]]).T,\
                       np.array([[x0_delim, s_end_limit[1, :].item()]]).T
    # r2 spans the remainder of x0 and x1 is limited to be 0.5 -> 2
    r2_start, r2_end = np.array([[x0_delim, x1_delim]]).T,\
                       np.array([[s_end_limit[0, :].item(), s_end_limit[1, :].item()]]).T
    # r3 also spans the remainder of x0 and now x1 too [-2, 0.5].
    r3_start, r3_end = np.array([[x0_delim, s_start_limit[1, :].item()]]).T,\
                       np.array([[s_end_limit[0, :].item(), x1_delim]]).T
    regions = [box_constraint(r1_start, r1_end), box_constraint(r2_start, r2_end), box_constraint(r3_start, r3_end)]

    # Add values to generate samples that lie outside of the constraint set to test those too
    grid_check = generate_fine_grid(s_start_limit-1, s_end_limit+1, fineness_param=(10, 10), viz_grid=False)
    # print(grid_check.shape)
    mask = [[], [], []]
    for grid_vec_idx in range(grid_check.shape[-1]):
        grid_vec = grid_check[:, grid_vec_idx]
        for region_idx in range(len(regions)):
            test_constraint = regions[region_idx]
            mask[region_idx].append((test_constraint.sym_func(grid_vec) <= 0).all().item())
    passed_vecs = [0, 0, 0]
    colours = ['r', 'g', 'b']
    if viz:
        plt.figure()
        for i in range(len(regions)):
            passed_vecs[i] = grid_check[:, mask[i]]
            plt.scatter(passed_vecs[i][0], passed_vecs[i][1], c=colours[i])
        print(grid_check)
    return regions


def construct_gp_wrapped_piecewise(hardcoded, regions, s_start_limit, s_end_limit, fixed_numeric_means,
                                   num_samples=50, noise_std_devs=(0.05, 0.02, 0.03),
                                   viz=True, fineness_param=(20, 20), verbose=True, ds_inst_in=None, cluster_based=False):
    # Global and piecewise both use the same ds_gen step but the training function is different.
    noise_vars = [noise_std_dev ** 2 for noise_std_dev in noise_std_devs]
    if not hardcoded:
        if ds_inst_in is None:
            ds_ndim_test = test_1d_op_2d_inp_allfeats(regions,
                                                      start_limit=s_start_limit, end_limit=s_end_limit, num_points=num_samples,
                                                      noise_vars=[noise_std_dev ** 2 for noise_std_dev in noise_std_devs],
                                                      cluster_based=cluster_based)
        else:
            ds_ndim_test = ds_inst_in
    if verbose:
        print("Training piecewise GP")
    if not hardcoded:
        likelihoods_piecewise_nd, piecewise_models_nd = piecewise_train_test(ds_ndim_test, no_squeeze=True, verbose=verbose)
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
        fig = plt.figure(figsize=(32, 32))
        axes = []
        for i in range(3):
            ax = fig.add_subplot(3, 1, i+1, projection='3d')
            axes.append(ax)
        ds_ndim_test.viz_outputs_2d(fineness_param=fineness_param, ax=axes[0])
        # num_samples = 1000
        # region_masks = {0: np.hstack([np.zeros(500), np.ones(500)]), 1: np.hstack([np.ones(500), np.zeros(500)])}
        # ds_ndim_test._gen_white_noise(noise_verbose=False, ax=axes[1], num_samples=num_samples, region_masks=region_masks)
        fine_grid, mask = ds_ndim_test.generate_fine_grid(fineness_param=fineness_param, with_mask=True)
        region_xs, observed_preds = [], []
        for idx in range(len(models)):
            likelihood, model = likelihoods[idx], models[idx]
            region_mask = mask[idx]
            # print(region_mask.nonzero())
            region_test_samples = fine_grid[:, region_mask.nonzero()[1]]
            # print(region_test_samples.shape, fine_grid.shape)
            # print(type(region_test_samples))
            observed_pred = likelihood(model(GPR_Callback.preproc(region_test_samples.T)))
            # Check to ensure the callback method is working as intended
            with torch.no_grad():
                # callback sparsity is 1 sample at a time so need to iterate through all 1 at a time
                for sample_idx in range(region_test_samples.shape[-1]):
                    sample = region_test_samples[:, sample_idx]
                    residual_mean, *residual_covs = piecewise_gp_wrapped(sample)
                    non_callback_mean = observed_pred.mean.numpy()[sample_idx]
                    assert np.abs(residual_mean[:, idx] - non_callback_mean) <= 1e-4, \
                        "GP output mean (%s) and non-callback residual mean (%s) don't match: " % (residual_mean[:, idx], non_callback_mean)
            observed_preds.append(observed_pred)
            region_xs.append(region_test_samples)
        idx = -1
        colours = ['r', 'g', 'b']
        with torch.no_grad():
            for region_x, observed_pred in zip(region_xs, observed_preds):
                idx += 1
                axes[1].scatter3D(region_x[0, :], region_x[1, :],
                                  observed_pred.mean.numpy(), color=colours[idx])

    if viz and not hardcoded:
        return piecewise_gp_wrapped, ds_ndim_test, axes
    else:
        return piecewise_gp_wrapped, ds_ndim_test


def construct_gp_wrapped_global(hardcoded=False, fixed_numeric_means=False, ds_in: GP_DS=None,
                                viz=True, ax=None, fineness_param=(20, 20), verbose=True):
    assert ds_in is not None, "You must pass an input dataset that is shared across the global and piecewise case and generated by construct_gp_wrapped_piecewise"
    if verbose:
        print("Training global GP")
    ds_ndim_test = ds_in
    if not hardcoded:
        likelihoods_2d, models_2d = train_test(ds_ndim_test, no_squeeze=True, verbose=verbose)
        likelihoods, models = likelihoods_2d, models_2d
        for model_idx in range(len(models)):
            models[model_idx].eval()
            likelihoods[model_idx].eval()
        res_input_dim = ds_ndim_test.input_dims
        res_output_dim = ds_ndim_test.output_dims
        global_gp_wrapped = MultidimGPR_Callback('f', likelihoods, models,
                                                 state_dim=res_input_dim, output_dim=res_output_dim, opts={"enable_fd": True})

    else:
        assert fixed_numeric_means is True, "The hardcoded function based test has not been created for this example."
        global_gp_wrapped = Global_Callback_Hardcoded_2D('f', 1, 1, opts={"enable_fd": True})

    if viz and not hardcoded:
        if ax is None:
            # First half of plots shows the true mean function and second half shows the GP learnt mean function
            fig, axes = plt.subplots(1, res_output_dim*2, figsize=(10, 13))
            ds_ndim_test.viz_outputs_2d(fineness_param=fineness_param, ax=axes[:res_output_dim])
            ax = axes[res_output_dim:]
        else:
            assert len(ax) == res_output_dim, "The number of axes passed must be the same as the number of residual output dims %s but got %s" % (res_output_dim, len(ax))
        observed_preds = []
        fine_grid = ds_ndim_test.generate_fine_grid(fineness_param=fineness_param).squeeze()
        for idx in range(len(models)):
            likelihood, model = likelihoods[idx], models[idx]
            observed_pred = likelihood(model(GPR_Callback.preproc(fine_grid.squeeze().T)))
            # Check to ensure the callback method is working as intended
            with torch.no_grad():
                # callback sparsity is 1 sample at a time so need to iterate through all 1 at a time
                for sample_idx in range(fine_grid.shape[-1]):
                    sample = fine_grid[:, sample_idx]
                    residual_mean, *residual_covs = global_gp_wrapped(sample)
                    non_callback_mean = observed_pred.mean.numpy()[sample_idx]
                    assert np.abs(residual_mean[:, idx] - non_callback_mean) <= 1e-4, \
                        "GP output mean (%s) and non-callback residual mean (%s) don't match: " % (residual_mean[:, idx], non_callback_mean)
            observed_preds.append(observed_pred)
        idx = -1
        colours = ['r', 'g', 'b']
        with torch.no_grad():
            for observed_pred in observed_preds:
                idx += 1
                ax[idx].scatter3D(fine_grid[0, :], fine_grid[1, :],
                                  observed_pred.mean.numpy(), color=colours[idx])
    return global_gp_wrapped


def construct_config_opts(print_level, add_monitor, early_termination, only_soln_limit, hsl_solver,
                          closed_loop, test_no_lam_p, warmstart, bonmin_warmstart=False, hessian_approximation=True,
                          enable_forward=False, relaxed=False, piecewise=True, max_iter=70):
    if piecewise and not relaxed:
        opts = {"bonmin.print_level": print_level, 'bonmin.file_solution': 'yes', 'bonmin.expect_infeasible_problem': 'no'}
        # opts.update({"bonmin.allowable_gap": -100, 'bonmin.allowable_fraction_gap': -0.1, 'bonmin.cutoff_decr': -10})
        opts.update({"bonmin.allowable_gap": 2})
        # Ref Page 11 https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.432.3757&rep=rep1&type=pdf
        # opts.update({"bonmin.num_resolve_at_root": 2, 'bonmin.num_resolve_at_node': 2,
        #              "bonmin.max_random_point_radius": 0.5, "bonmin.random_point_type": "Andreas"})
        if enable_forward:
            opts["enable_fd"] = False
            opts["enable_forward"] = True
        if add_monitor:
            opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
        if early_termination:
            opts["bonmin.solution_limit"] = 4
            opts.update({"bonmin.allowable_gap": 5})
            # opts['bonmin.max_iter'] = max_iter
            # opts["bonmin.heuristic_RINS"] = "yes"
            # opts["RINS.algorithm"] = "B-QG"
            # opts["bonmin.rins.solution_limit"] = 1
            # opts["bonmin.algorithm"] = "B-OA"
            # if warmstart:
            #     opts["bonmin.iteration_limit"] = 1
            #     opts["bonmin.node_limit"] = 1
        if not only_soln_limit:
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
        acceptable_iter = 3
        acceptable_constr_viol_tol = 1e-3
        acceptable_tol = 1e4

        if early_termination:
            additional_opts = {"ipopt.acceptable_tol": acceptable_tol, "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                               "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol, "ipopt.acceptable_iter": acceptable_iter,
                               "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol}
            opts.update(additional_opts)
    return opts


def initialize_piecewise(x_init, warmstart, bonmin_warmstart, b_shrunk_opt, warmstart_dict, controller_test, x_desired=None, verbose=True):
    # Set parameters and initial opt var assignment
    initialize_dict = {'x_init': x_init, 'verbose': verbose, "x_desired": x_desired}
    if x_desired is None:
        del initialize_dict["x_desired"]
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


def initialize_global(x_init, warmstart, b_shrunk_opt, warmstart_dict, controller_test, x_desired=None, verbose=True):
    # Set parameters and initial opt var assignment
    initialize_dict = {'x_init': x_init, 'verbose': verbose, "x_desired": x_desired}
    if x_desired is None:
        del initialize_dict["x_desired"]
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


def run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict, simulation_length=None,
                    true_func_obj=None, piecewise=True, x_desired=None, verbose=True, ignore_covs=False):
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
        return sol
    else:
        X_test, U_test, regions = controller_test.X, controller_test.U, controller_test.regions
        data_dicts = []
        sol_stats = []
        for i in range(simulation_length):
            if store_in_file:
                output_file = open(file_name, 'w')
                # Retrieve default stdout
                stdout_old = sys.stdout
                # Re-direct standard output
                sys.stdout = output_file
            try:
                if verbose:
                    print("Solving for timestep: %s" % i)
                sol = controller_test.solve_optimization(ignore_initial=True, **initialize_dict)
            except Exception as e:
                # Error is thrown after finishing computing solution before generating the warmstarts for the next iteration.
                # So all the warmstarting code for the next iteration is put in this block after the exception has been caught.
                print(e)
                if store_in_file:
                    # If error thrown -> print exception to old stdout
                    traceback.print_exc(file=stdout_old)
                    traceback.print_exc(file=sys.stdout)
            finally:
                infeasible_prob = False
                try:
                    print("t: %s, Ret. stat.: %s" % (i, sol.stats()["return_status"]), end=" ")
                except UnboundLocalError:
                    print("t: %s, Ret. stat.: %s" % (i, "Infeasible_Problem_Detected"), end=" ")
                    infeasible_prob = True
                if verbose:
                    print("Solution")
                if not piecewise:
                    debugger_inst, data_dict = retrieve_controller_results(controller_test, X_test, U_test, return_data_dict=True, verbose=verbose)
                else:
                    debugger_inst, data_dict = retrieve_controller_results_piecewise(controller_test, X_test, U_test, ignore_lld=True,
                                                                                     return_data_dict=True, verbose=verbose, ignore_covs=ignore_covs)
                    if debugger_inst is False:
                        data_dict["minlp_error"] = True
                        sol_stats.append("minlp_error")
                        data_dicts.append(data_dict)
                        return data_dicts, sol_stats
                if not infeasible_prob:
                    sol_stats.append(sol.stats())
                else:
                    if not piecewise:
                        data_dict["infeasible_error_gend"] = True
                data_dicts.append(data_dict)
                # if infeasible_prob:
                #     break
                x_des_end_delim = controller_test.N + 1
                generate_and_set_warmstart_from_previous_iter_soln(controller_test, initialize_dict, X_test, U_test, regions, true_func_obj,
                                                                   no_cov_case=controller_test.ignore_covariances, piecewise=piecewise, verbose=verbose,
                                                                   x_desired=(x_desired[:, i+1:i+1+x_des_end_delim] if x_desired is not None else None))
                if store_in_file:
                    # Re-direct stdout to default
                    sys.stdout = stdout_old
        return data_dicts, sol_stats


def gen_ds_and_train(hardcoded=False, fixed_numeric_means=False, num_samples=50, viz=True, verbose=True,
                     problem_setup_fn=setup_problem_basic, ds_inst_in=None, cluster_based=False, noise_std_devs=(0.05, 0.02, 0.03)):
    # Note: Only valid here for ds gen since both the delta control inputs and gp inputs depend on the same 2 variables i.e. the
    # state vector
    if problem_setup_fn not in valid_setups:
        raise NotImplementedError(
            "This function currently doesn't work for anything other than the basic setup problem. Need to "
            "incorporate a limit vector generation function to be passed for dataset generation.")

    s_start_limit, s_end_limit = problem_setup_fn()["s_start_limit"], problem_setup_fn()["s_end_limit"]
    regions = problem_setup_fn()["regions"]
    # In closed loop, the gp_ds obj contains info relating to the true function and can be used to generate the next state in simulation using the generate_outputs function
    # by passing a custom array (in this case just the first sample i.e. x_init for that O.L. timestep)
    piecewise_gp_wrapped, *rem = construct_gp_wrapped_piecewise(hardcoded, regions, s_start_limit, s_end_limit, fixed_numeric_means,
                                                                num_samples=num_samples,
                                                                viz=viz, verbose=verbose, ds_inst_in=ds_inst_in, cluster_based=cluster_based,
                                                                noise_std_devs=noise_std_devs)
    if viz:
        gp_ds_inst, axes = rem
    else:
        gp_ds_inst = rem[0]
    global_gp_wrapped = construct_gp_wrapped_global(hardcoded=hardcoded, fixed_numeric_means=fixed_numeric_means,
                                                    ds_in=gp_ds_inst, ax=(None if not viz else [axes[-1]]),
                                                    viz=viz, verbose=verbose)

    return piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst


def GPR_test_hardcoded_allfeatsactive_piecewise(gp_fns, skip_shrinking=False, early_termination=False, only_soln_limit=False, minimal_print=True,
                                                stable_system=False, warmstart=False, warmstart_dict=None, add_monitor=False, store_in_file=False, file_name='',
                                                hsl_solver=False, no_solve=False, b_shrunk_opt=False,
                                                N=2, U_big=True, custom_U=None,
                                                approxd_shrinking=False, closed_loop=False, simulation_length=5,
                                                test_no_lam_p=False, bonmin_warmstart=False,
                                                hessian_approximation=False, ignore_cost=False, ignore_variance_costs=False,
                                                enable_forward=True, relaxed=False, add_delta_tol=False, test_softplus=False, true_func_obj: GP_DS=None,
                                                with_metrics=True, show_plot=True, skip_feedback=True, return_run_info=False,
                                                problem_setup_fn=setup_problem_basic, verbose=True, custom_xinit=None,
                                                x_desired=None, ignore_init_constr_check=False, satis_prob=None):
    A_test, B_test, Bd_test, Q_test, R_test, n_x, n_u, n_d, \
    s_start_limit, s_end_limit, u_start_limit, u_end_limit,\
    x_init, x0_delim, x1_delim, regions, X_test, U_test,\
    gp_inputs, gp_input_mask, delta_control_variables, delta_input_mask, satisfaction_prob = problem_setup_fn(stable_system, custom_U, U_big).values()
    # print(U_test)
    if custom_xinit is not None:
        x_init = custom_xinit
    if satis_prob is not None:
        satisfaction_prob = satis_prob

    if closed_loop and with_metrics:
        assert true_func_obj is not None, "Need to pass a GP_DS inst that generated the data samples for closed loop simulation with metrics"

    gp_wrapped = gp_fns

    print_level = 0 if minimal_print else 5

    # U_shrunk = box_constraint(U_test.lb/100, U_test.ub/100)
    opts = construct_config_opts(print_level, add_monitor, early_termination, only_soln_limit, hsl_solver,
                                 closed_loop, test_no_lam_p, warmstart, bonmin_warmstart, hessian_approximation,
                                 enable_forward, relaxed)

    controller_test = GP_MPC_Alt_Delta(A=A_test, B=B_test, Bd=Bd_test, Q=Q_test, R=R_test, horizon_length=N,
                                       n_x=n_x, n_u=n_u, n_d=n_d,
                                       gp_inputs=gp_inputs, gp_input_mask=gp_input_mask,
                                       delta_control_variables=delta_control_variables, delta_input_mask=delta_input_mask,
                                       gp_fns=gp_wrapped,  # Using gp model wrapped in casadi callback
                                       X=X_test, U=U_test, satisfaction_prob=satisfaction_prob, regions=regions,
                                       skip_shrinking=skip_shrinking, addn_solver_opts=opts,
                                       add_b_shrunk_opt=b_shrunk_opt, piecewise=True, add_b_shrunk_param=approxd_shrinking,
                                       ignore_cost=ignore_cost, ignore_variance_costs=ignore_variance_costs, relaxed=relaxed,
                                       add_delta_tol=add_delta_tol, test_softplus=test_softplus, skip_feedback=skip_feedback,
                                       ignore_init_constr_check=ignore_init_constr_check)
    if verbose:
        controller_test.display_configs()
    controller_test.setup_OL_optimization()

    if no_solve:
        controller_test.set_initial(**{'x_init': x_init})
        return controller_test
    else:
        initialize_dict = initialize_piecewise(x_init, warmstart, bonmin_warmstart, b_shrunk_opt, warmstart_dict, controller_test, verbose=verbose,
                                               x_desired=(x_desired[:, :N+1] if x_desired is not None else None))

    if not closed_loop:
        run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict)
    else:
        data_dict_cl, sol_stats_cl = run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict,
                                                     simulation_length=simulation_length, true_func_obj=true_func_obj,
                                                     x_desired=x_desired, verbose=verbose, ignore_covs=(approxd_shrinking and ignore_variance_costs))
        mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
        if show_plot:
            plt.figure()
            print(data_dict_cl)
            print(sol_stats_cl)
            colours = ['r', 'g', 'b']
            for timestep, mu_x_ol in enumerate(mu_x_cl):
                plt.plot(mu_x_ol[0, :].squeeze(), mu_x_ol[1, :].squeeze(), color=colours[timestep], marker='x',
                         linestyle='dashed', linewidth=2, markersize=12, label='OL output: Timestep: %s' % timestep)
            plt.legend(loc='upper center')

    if return_run_info and closed_loop:
        return controller_test, X_test, U_test, data_dict_cl, sol_stats_cl
    else:
        return controller_test, X_test, U_test


def generate_and_set_warmstart_from_previous_iter_soln(controller_test, initialize_dict, X_test, U_test, regions, true_func_obj: GP_DS, ignore_lld=True,
                                                       no_cov_case=False, piecewise=False, return_mu_x=False, x_desired=None, verbose=True):
    inverse_cdf_x, inverse_cdf_u = controller_test.inverse_cdf_x, controller_test.inverse_cdf_u
    N = controller_test.N
    debugger_inst = OptiDebugger(controller_test)

    if x_desired is not None:
        initialize_dict["x_desired"] = x_desired
    # print(x_desired)

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
    sampled_residual = true_func_obj.generate_outputs(input_arr=type_convd_inp, no_noise=False, return_op=True, noise_verbose=verbose)
    sampled_ns = controller_test.A @ mu_x[:, [0]] + controller_test.B @ mu_u[:, [0]] + controller_test.Bd @ sampled_residual.numpy()
    # print(initialize_dict['mu_x'].shape, sampled_ns.shape, (controller_test.A @ mu_x[:, [0]]).shape, (controller_test.B @ mu_u[:, [0]]).shape,
    #       (controller_test.Bd @ sampled_residual.numpy()).shape)

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
            hld[region_idx, 0] = 1 if region.check_satisfaction(delta_ctrl_inp.squeeze()).item() is True else 0
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
            # print(mu_x, mu_u)
            # print(mu_x.shape, mu_u.shape)
            # print(N)
            shrunk_gen_inst = FeasibleTraj_Generator(controller_test.U, controller_test, verbose=True)
            Sigma_x, Sigma_u = shrunk_gen_inst.approxd_cov_gen(mu_x[:, 1:], mu_u[:, 1:], hld)
        else:
            shrunk_gen_inst = FeasibleTraj_Generator(controller_test.U, controller_test, verbose=True, piecewise=False)
            Sigma_x, Sigma_u = shrunk_gen_inst.approxd_cov_gen(mu_x[:, 1:], mu_u[:, 1:], None)
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

    initialize_dict['verbose'] = verbose
    controller_test.set_initial(**initialize_dict)

    if return_mu_x:
        return mu_x_orig


def generate_feasible_traj_piecewise(gp_fns: Piecewise_GPR_Callback, u_start_limit=np.array([[-2]]).T, u_end_limit=np.array([[2]]).T, custom_U=None, U_big=False,
                                     internal_timeout_interval=10, external_timeout_interval=10, x_init=np.array([[-0.75], [0.501]]), N=3, skip_feedback=False,
                                     stable_system=False, problem_setup_fn=setup_problem_basic, verbose=True):
    if x_init is None:
        x_init = problem_setup_fn()["x_init"]
    controller_inst = GPR_test_hardcoded_allfeatsactive_piecewise(gp_fns=gp_fns, minimal_print=True, store_in_file=True, add_monitor=True,
                                                                  hsl_solver=False, stable_system=stable_system,
                                                                  early_termination=True, no_solve=True, b_shrunk_opt=True, N=N,
                                                                  U_big=U_big, custom_U=custom_U, skip_feedback=skip_feedback,
                                                                  problem_setup_fn=problem_setup_fn, verbose=verbose)
    feasible_found = False
    U_shrunk_inst = box_constraint(u_start_limit, u_end_limit)
    for i in range(external_timeout_interval):
        if verbose:
            print(i)
        traj_generator = FeasibleTraj_Generator(U_shrunk_inst, controller_inst, timeout_activate=True, timeout_interval=internal_timeout_interval, verbose=False)
        traj_info = traj_generator.get_traj(x_init)
        # Traj generator is given a counter. If unable to find a solution to any timestep within timeout_interval number of attempts then we restart the generator
        # in the hope that picking alternate inputs earlier in the sequence can help randomly find a feasible solution.
        if traj_info is False:
            feasible_found = False
            if verbose:
                print("Failed after %s attempts with the traj generator. Restarting traj generator" % internal_timeout_interval)
            continue
        else:
            feasible_found = True
            x_traj, u_traj, hld_mat, lld_mat, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true, mu_d = traj_info
            break
    if not feasible_found:
        print("Not able to find a feasible solution after %s restarts" % (external_timeout_interval))
        return False
    else:
        print("Feasible solution found for piecewise case", end="\t")
        if verbose:
            print(" (lld mat not used though printed)")
            print('x_traj\n %s \n u_traj\n %s \n mu_d\n %s \nhld_mat\n %s \n lld_mat\n %s \n b_shrunk_x\n %s \n b_shrunk_u\n %s \n'
                  ' Sigma_x\n %s \n Sigma_u\n %s \n Sigma_d\n %s \n b_shrunk_u_true\n %s \n' % (
                x_traj, u_traj, mu_d, hld_mat, lld_mat, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true))
    warmstart_dict = {'mu_x': x_traj, "mu_u": u_traj, 'hld': hld_mat, 'lld': lld_mat, 'b_shrunk_x': b_shrunk_x, 'b_shrunk_u': b_shrunk_u_true,
                      'mu_d': mu_d}

    if verbose:
        print("Checking violation")
    x, p = generate_x_and_p_vecs(x_init, controller_inst.n_x, controller_inst.n_u, controller_inst.N,
                                 x_traj, u_traj, hld_mat, b_shrunk_x, b_shrunk_u_true)
    f = cs.Function('f', [controller_inst.opti.x, controller_inst.opti.p], [controller_inst.opti.g])
    try:
        constr_viol_split = cs.vertsplit(f(x, p), 1)
    except Exception as e:
        print(e)
        if verbose:
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
    if verbose:
        print("Constraint violation: %s" % constr_viol)

    return warmstart_dict, f


def generate_feasible_traj_global(gp_fns: MultidimGPR_Callback, u_start_limit=np.array([[-2]]).T, u_end_limit=np.array([[2]]).T, custom_U=None, U_big=False,
                                  internal_timeout_interval=10, external_timeout_interval=10, x_init=np.array([[-0.75], [0.501]]), N=3, skip_feedback=False,
                                  stable_system=False, problem_setup_fn=setup_problem_basic, verbose=True):
    if x_init is None:
        x_init = problem_setup_fn()["x_init"]
    controller_inst = GPR_test_hardcoded_allfeatsactive_global(gp_fns=gp_fns, minimal_print=True, store_in_file=True, add_monitor=True,
                                                               hsl_solver=False,
                                                               early_termination=True, no_solve=True, b_shrunk_opt=True, N=N,
                                                               U_big=U_big, custom_U=custom_U, skip_feedback=skip_feedback, stable_system=stable_system,
                                                               problem_setup_fn=problem_setup_fn, verbose=verbose)
    feasible_found = False
    U_shrunk_inst = box_constraint(u_start_limit, u_end_limit)
    for i in range(external_timeout_interval):
        if verbose:
            print(i)
        traj_generator = FeasibleTraj_Generator(U_shrunk_inst, controller_inst, timeout_activate=True, timeout_interval=internal_timeout_interval, verbose=False,
                                                piecewise=False)
        traj_info = traj_generator.get_traj(x_init)
        # Traj generator is given a counter. If unable to find a solution to any timestep within timeout_interval number of attempts then we restart the generator
        # in the hope that picking alternate inputs earlier in the sequence can help randomly find a feasible solution.
        if traj_info is False:
            feasible_found = False
            if verbose:
                print("Failed after %s attempts with the traj generator. Restarting traj generator" % internal_timeout_interval)
            continue
        else:
            feasible_found = True
            print("Feasible solution found for global case", end="\t")
            x_traj, u_traj, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true, mu_d = traj_info
            if verbose:
                print(" (lld mat not used though printed)")
                print('x_traj\n %s \n u_traj\n %s \n mu_d\n %s \n b_shrunk_x\n %s \n b_shrunk_u\n %s \n'
                      ' Sigma_x\n %s \n Sigma_u\n %s \n Sigma_d\n %s \n b_shrunk_u_true\n %s \n' % (
                x_traj, u_traj, mu_d, b_shrunk_x, b_shrunk_u, Sigma_x, Sigma_u, Sigma_d, b_shrunk_u_true))
            break
    if not feasible_found:
        print("Not able to find a feasible solution after %s restarts" % (external_timeout_interval))
        return False

    warmstart_dict = {'mu_x': x_traj, "mu_u": u_traj, 'b_shrunk_x': b_shrunk_x, 'b_shrunk_u': b_shrunk_u_true,
                      'mu_d': mu_d}

    if verbose:
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
    if verbose:
        print(constr_viol)

    return warmstart_dict, f


def GPR_test_hardcoded_allfeatsactive_global(gp_fns, skip_shrinking=False, early_termination=False, only_soln_limit=False, minimal_print=True,
                                             stable_system=True, warmstart=False, warmstart_dict=None, add_monitor=False, store_in_file=False, file_name='',
                                             hsl_solver=False, no_solve=False, b_shrunk_opt=False,
                                             N=2, U_big=True, custom_U=None,
                                             approxd_shrinking=False, closed_loop=False, simulation_length=5,
                                             test_no_lam_p=False, hessian_approximation=False, ignore_cost=False, ignore_variance_costs=False,
                                             true_func_obj: GP_DS=None, with_metrics=True, skip_feedback=False, return_run_info=False,
                                             problem_setup_fn=setup_problem_basic, verbose=True, custom_xinit=None, x_desired=None,
                                             enable_forward=False, ignore_init_constr_check=False, satis_prob=None):
    A_test, B_test, Bd_test, Q_test, R_test, n_x, n_u, n_d, \
    s_start_limit, s_end_limit, u_start_limit, u_end_limit,\
    x_init, x0_delim, x1_delim, regions, X_test, U_test,\
    gp_inputs, gp_input_mask, delta_control_variables, delta_input_mask, satisfaction_prob = problem_setup_fn(stable_system, custom_U, U_big).values()
    if custom_xinit is not None:
        x_init = custom_xinit
    if satis_prob is not None:
        satisfaction_prob = satis_prob
    if closed_loop and with_metrics:
        assert true_func_obj is not None, "Need to pass a GP_DS inst that generated the data samples for closed loop simulation with metrics"

    gp_wrapped = gp_fns

    print_level = 0 if minimal_print else 5

    opts = construct_config_opts(print_level, add_monitor, early_termination, only_soln_limit, hsl_solver,
                                 closed_loop, test_no_lam_p, warmstart, hessian_approximation=hessian_approximation, piecewise=False)

    controller_test = GP_MPC_Alt_Delta(A=A_test, B=B_test, Bd=Bd_test, Q=Q_test, R=R_test, horizon_length=N,
                                       n_x=n_x, n_u=n_u, n_d=n_d,
                                       gp_inputs=gp_inputs, gp_input_mask=gp_input_mask,
                                       delta_control_variables=delta_control_variables, delta_input_mask=delta_input_mask,
                                       gp_fns=gp_wrapped,  # Using gp model wrapped in casadi callback
                                       X=X_test, U=U_test, satisfaction_prob=satisfaction_prob, regions=regions,
                                       skip_shrinking=skip_shrinking, addn_solver_opts=opts,
                                       add_b_shrunk_opt=b_shrunk_opt, piecewise=False, add_b_shrunk_param=approxd_shrinking,
                                       ignore_cost=ignore_cost, ignore_variance_costs=ignore_variance_costs,
                                       solver='ipopt', skip_feedback=skip_feedback, ignore_init_constr_check=ignore_init_constr_check)
    if verbose:
        controller_test.display_configs()
    controller_test.setup_OL_optimization()

    if no_solve:
        controller_test.set_initial(**{'x_init': x_init})
        return controller_test
    else:
        initialize_dict = initialize_global(x_init, warmstart, b_shrunk_opt, warmstart_dict, controller_test, verbose=verbose,
                                            x_desired=(x_desired[:, :N+1] if x_desired is not None else None))

    if not closed_loop:
        run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict)
    else:
        data_dict_cl, sol_stats_cl = run_opt_attempt(store_in_file, file_name, controller_test, initialize_dict,
                                                     simulation_length=simulation_length, true_func_obj=true_func_obj,
                                                     piecewise=False, verbose=verbose, x_desired=x_desired)
        # mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
        if verbose:
            print(data_dict_cl)
            print(sol_stats_cl)

    if return_run_info and closed_loop:
        return controller_test, X_test, U_test, data_dict_cl, sol_stats_cl
    else:
        return controller_test, X_test, U_test


class Piecewise_Callback_Hardcoded_2D(cs.Callback):
    def __init__(self, name, inp_dim=2, output_dim=1, num_regions=3, opts={}):
        cs.Callback.__init__(self)
        self.inp_dim = inp_dim
        self.output_dims = output_dim
        self.num_regions = num_regions
        assert inp_dim == 2, "Input dim must be 2 to use this example callback"
        assert output_dim == 1, "Output dim must be 1 to use this example callback"
        assert num_regions == 3, "Num regions must be 3 to use this example callback"
        cov1, cov2, cov3 = np.array([[0.1]]), np.array([[0.05]]), np.array([[0.2]])
        self.cov = [cov1, cov2, cov3]
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1+self.num_regions

    def get_num_samples(self):
        return "Function is continuous for testing."

    def get_sparsity_in(self, i):
        # If this isn't specified then it doesn't accept the full input vector and only limits itself to the first element.
        return cs.Sparsity.dense(self.inp_dim, 1)

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.output_dims, self.num_regions)
        elif i >= 1:
            return cs.Sparsity.dense(self.output_dims, self.output_dims)

    def eval(self, arg):
        mean = cs.DM(np.array([[0.02, 0.075, 0.032]]))
        # Throwaway. There is no input dependence on the output of this callback. Essentially it is equivalent to saying that
        # the piecewise nature of the system is just in the nominal model and there is no non-linearity or uncertainty in the
        # residual
        sample = arg[0]

        return [mean, *self.cov]


class Global_Callback_Hardcoded_2D(cs.Callback):
    def __init__(self, name, inp_dim, output_dim, opts={}):
        cs.Callback.__init__(self)
        self.inp_dim = inp_dim
        self.output_dims = output_dim
        cov = np.diag([0.05, 0.02])
        self.cov = cov
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 2

    def get_num_samples(self):
        return "Function is continuous for testing."

    def get_sparsity_in(self, i):
        # If this isn't specified then it doesn't accept the full input vector and only limits itself to the first element.
        return cs.Sparsity.dense(self.inp_dim, 1)

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.output_dims, 1)
        elif i >= 1:
            return cs.Sparsity.dense(self.output_dims, self.output_dims)

    def eval(self, arg):
        mean = cs.DM(np.array([[0.04], [0.075]]))
        # Throwaway. There is no input dependence on the output of this callback. Essentially it is equivalent to saying that
        # the piecewise nature of the system is just in the nominal model and there is no non-linearity or uncertainty in the
        # residual
        sample = arg[0]

        return [mean, self.cov]


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
        assert mu_x.shape[-1] == self.N, "mu_x array passed as input should only be from timestep 1 to N of the O.L. opt solution"
        assert mu_u.shape[-1] == self.N-1, "mu_u array passed as input should only be from timestep 1 to N-1 of the O.L. opt solution"
        Sigma_d = [np.zeros((self.res_dim, self.res_dim)) for _ in range(self.N)]
        Sigma_x = [np.zeros((self.n_x, self.n_x)) for _ in range(self.N+1)]
        Sigma_u = [np.zeros((self.n_u, self.n_u)) for _ in range(self.N)]
        Sigma = []
        for i in range(self.N-1):
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


def retrieve_opt_var_values(opt_var_assgt, N, n_x, n_u, delta_inp_dim, num_regions, verbose=True, bypass_lld=False):
    # First variable is mu_x which has N+1*state_dim elements
    mu_x_delim = ((N+1)*n_x)
    mu_u_delim = mu_x_delim + (N*n_u)
    lld_delim = mu_u_delim + 0
    if not bypass_lld:
        lld_delim = mu_u_delim + (delta_inp_dim*2*num_regions*N) # *2 for box constraint only
    hld_delim = lld_delim + num_regions*N
    assert hld_delim == len(opt_var_assgt), "Something wrong with length of input assignment"
    mu_x = opt_var_assgt[0:mu_x_delim]
    mu_x_array = np.zeros((n_x, N+1))
    shape_into_arr(mu_x_array, mu_x, n_x, N+1)
    # for k in range(N+1):
    #     mu_x_array[:, k] = mu_x[k*n_x:(k+1)*n_x]
    mu_u = opt_var_assgt[mu_x_delim:mu_u_delim]
    mu_u_array = np.zeros((n_u, N))
    shape_into_arr(mu_u_array, mu_u, n_u, N)
    if not bypass_lld:
        single_lld_delim = delta_inp_dim*2*num_regions
        all_llds = opt_var_assgt[mu_u_delim: lld_delim]
        # Just placeholder to populate with arrays
        lld_list = [np.zeros((delta_inp_dim*2, num_regions)) for _ in range(N)]
        for k in range(N):
            timestep_lld = all_llds[k*single_lld_delim:(k+1)*single_lld_delim]
            shape_into_arr(lld_list[k], timestep_lld, delta_inp_dim*2, num_regions)
    all_hlds = opt_var_assgt[lld_delim: hld_delim]
    hld_array = np.zeros((num_regions, N))
    shape_into_arr(hld_array, all_hlds, num_regions, N)

    if verbose:
        print('mu_x')
        print(mu_x_array)
        print('mu_u')
        print(mu_u_array)
        if not bypass_lld:
            print('llds')
            print(lld_list)
        print('hlds')
        print(hld_array)
    if not bypass_lld:
        return mu_x_array, mu_u_array, hld_array, lld_list
    else:
        return mu_x_array, mu_u_array, hld_array


def generate_x_and_p_vecs(x_init, n_x, n_u, N, x_traj, u_traj, hld_mat=None, b_shrunk_x=None, b_shrunk_u=None, piecewise=True):
    if piecewise:
        assert hld_mat is not None, "Need to pass hld mat to use this function in the piecewise case."
    x_des_vec, u_des_vec = np.zeros(n_x*(N+1)), np.zeros(n_u*N)
    p = np.concatenate((x_init.flatten(order='F'), x_des_vec, u_des_vec))
    x = np.array([])
    x = np.concatenate((x, x_traj.flatten(order='F')))
    # print(x.shape)
    x = np.concatenate((x, u_traj.flatten(order='F')))
    # print(x.shape)
    if piecewise:
        x = np.concatenate((x, np.array(hld_mat).flatten(order='F')))
        # print(x.shape)
    if b_shrunk_x is not None:
        # the 1: is to ignore first shrunk vector which, based on assumption that there is no uncertainty
        # of first state means that b_shrunk_x[:, 0] always equal X.b and so we dont need that constrained
        # in the optimization even to debug.
        x = np.concatenate((x, b_shrunk_x[:, 1:].flatten(order='F')))
        # print(x.shape)
        x = np.concatenate((x, b_shrunk_u[:, 1:].flatten(order='F')))
        # print(x.shape)
    return x, p


def nan_simulate_test(controller_inst:GP_MPC_Alt_Delta, x=None, p=None, bypass_lld=True):
    if x is None:
        x = [-0.75, 0.501, -0.101376, 0.0711469, -0.0104019, 0.00589548, -0.0002509, 0.000152554, 0.223448, 0.0296278, 0.00228708, -9.18137e-009, -0.00999999, -0.00999999, -1.38482e-009, 1.4853e-007, 1, 3.03331e-007, 5.12612e-006, 0.999995]
    if p is None:
        p = [-0.75, 0.501, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    delta_inp_dim = np.linalg.matrix_rank(controller_inst.delta_input_mask)
    mu_x, mu_u, hld = retrieve_opt_var_values(opt_var_assgt=x,
                                              N=controller_inst.N, n_x=controller_inst.n_x, n_u=controller_inst.n_u,
                                              delta_inp_dim=delta_inp_dim, num_regions=controller_inst.num_regions, verbose=True, bypass_lld=bypass_lld)
    debugging_inst = FeasibleTraj_Generator(controller_inst.U, controller_inst, verbose=True)
    debugging_inst.fwd_sim_w_opt_var_assgt(mu_x, mu_u, hld)


def data_gen_for_metrics(num_initializations=1, num_runs_per_init=1, problem_setup_fn=setup_problem_basic,
                         base_path="C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\",
                         sub_path=None, datastore_file='run_info.pkl', N=3,
                         skip_feedback=False, stable_system=False, hsl_solver=False, simulation_length=3,
                         verbose=False, cluster_based=False, num_samples=50, edge_case_xdes=False,
                         x_init_limits=(0.5, 1), only_run_with_warmstart=True, early_termination=True,
                         test_softplus=False, noise_std_devs=(0.05, 0.02, 0.03), ignore_global=False,
                         ignore_init_constr_check=False, only_neg_xinit=False, only_pos_xinit=False,
                         satisfaction_prob=0.8):
    print("Steps per init run: \n"
          " 1) Generate dataset, train piecewise and global models and wrap them in callbacks\n"
          " 2) Generate warmstarts for both piecewise and global case\n"
          " 3-4) Run closed loop case for piecewise and global")
    store_dir = dir_exist_or_create(base_path, sub_path=sub_path)
    existing_data = []
    if os.path.exists(store_dir+datastore_file):
        try:
            with open(store_dir+datastore_file, "rb") as pklfile:
                existing_data = pkl.load(pklfile)
        except EOFError:
            existing_data = []
    existing_data.append([])

    all_accd_info = {"run_info": []}
    config_info = {"num_initializations": num_initializations, "num_runs_per_init": num_runs_per_init, "cluster_based": cluster_based, "num_samples": num_samples,
                   "x_init_limits": x_init_limits, "only_run_with_warmstart": only_run_with_warmstart, "early_termination": early_termination,
                   "noise_std_devs": noise_std_devs, "ignore_init_constr_check": ignore_init_constr_check,
                   "only_neg_xinit": only_neg_xinit, "only_pos_xinit": only_pos_xinit, "satisfaction_prob": satisfaction_prob}
    all_accd_info["config_info"] = config_info

    try:
        piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst = gen_ds_and_train(hardcoded=False,
                                                                               fixed_numeric_means=False, num_samples=num_samples,
                                                                               viz=False, verbose=verbose, problem_setup_fn=problem_setup_fn,
                                                                               cluster_based=cluster_based, noise_std_devs=noise_std_devs)
    except Exception as e:
        # print("Failed case %s" % setup_idx)
        print(e)
        # continue
    for setup_idx in range(num_initializations):
        print("Trained global and piecewise models for trial %s" % setup_idx)
        U_big = False
        if problem_setup_fn in [setup_problem_basic, setup_constr_viol_test]:
            u_start_feas_gen, u_end_feas_gen = np.array([[-0.75]]).T, np.array([[0.75]]).T
        elif problem_setup_fn in [setup_smaller_ctrl_unstable, setup_smaller_ctrl_stable,
                                  cost_comp_stable, cost_comp_boundary, setup_2d_input]:
            u_start_feas_gen, u_end_feas_gen = problem_setup_fn()["u_start_limit"], problem_setup_fn()["u_end_limit"]
        else:
            raise NotImplementedError

        if problem_setup_fn in valid_setups:
            s_start_limit1, s_end_limit1 = np.array([[-x_init_limits[1]]]).T, np.array([[-x_init_limits[0]]]).T
            s_start_limit2, s_end_limit2 = np.array([[x_init_limits[0]]]).T, np.array([[x_init_limits[1]]]).T
            X1 = box_constraint(s_start_limit1, s_end_limit1)
            X2 = box_constraint(s_start_limit2, s_end_limit2)
        else:
            raise NotImplementedError

        traj_gen_invalid = True
        while traj_gen_invalid:
            neg_samples = X1.get_random_vectors(num_samples=2)
            pos_samples = X2.get_random_vectors(num_samples=2)
            probs = np.random.uniform()
            # probs = np.random.uniform(size=(2, 1))
            # x0 = neg_samples[0, 0] if probs[0, 0] < 0.5 else pos_samples[0, 0]
            # x1 = neg_samples[0, 1] if probs[1, 0] < 0.5 else pos_samples[0, 1]
            # x_init = np.round(np.array(np.hstack([x0, x1]), ndmin=2).T, 3)
            if only_neg_xinit:
                x_init = neg_samples.T
            elif only_pos_xinit:
                x_init = pos_samples.T
            else:
                x_init = neg_samples.T if probs < 0.5 else pos_samples.T
            # print(problem_setup_fn()["A_test"], problem_setup_fn()["B_test"], problem_setup_fn()["Q_test"], problem_setup_fn()["R_test"])
            traj_gen_op_piecewise = generate_feasible_traj_piecewise(gp_fns=piecewise_gp_wrapped,
                                                                     external_timeout_interval=50,
                                                                     u_start_limit=u_start_feas_gen, u_end_limit=u_end_feas_gen,
                                                                     U_big=U_big, problem_setup_fn=problem_setup_fn, N=N,
                                                                     x_init=x_init,
                                                                     skip_feedback=skip_feedback, stable_system=stable_system,
                                                                     verbose=verbose)
            piecewise_warmstart = True
            if traj_gen_op_piecewise is False:
                piecewise_warmstart = False
                warmstart_dict_piecewise = {}
            else:
                warmstart_dict_piecewise, violation_fn_piecewise = traj_gen_op_piecewise
            print()
            traj_gen_op_global = generate_feasible_traj_global(gp_fns=global_gp_wrapped,
                                                               external_timeout_interval=50,
                                                               u_start_limit=u_start_feas_gen, u_end_limit=u_end_feas_gen,
                                                               U_big=U_big, problem_setup_fn=problem_setup_fn, N=N,
                                                               x_init=x_init,
                                                               skip_feedback=skip_feedback, stable_system=stable_system, verbose=verbose)
            global_warmstart = True
            if traj_gen_op_global is False:
                global_warmstart = False
                warmstart_dict_global = {}
            else:
                warmstart_dict_global, violation_fn_global = traj_gen_op_global
            # If we only want to run for those x_inits that found a feasible solution, set the valid traj bool depending on
            # if feasible warmstarts were found for both the piecewise and global case.
            if only_run_with_warmstart:
                traj_gen_invalid = not (piecewise_warmstart and global_warmstart)
            else:
                traj_gen_invalid = False
        print()
        x_lim = 2
        x_desired = None if not edge_case_xdes else np.ones([problem_setup_fn()["n_x"], simulation_length+N+1]) * x_lim
        common_kwargs = {"minimal_print": True, "hsl_solver": hsl_solver, "early_termination": early_termination, "only_soln_limit": False,
                         "no_solve": False, "N": N, "U_big": U_big, "test_no_lam_p": True, "hessian_approximation": True,
                         "ignore_cost": False, "ignore_variance_costs": True, "closed_loop": True, "true_func_obj": gp_ds_inst, "enable_forward": False,
                         "return_run_info": True, "store_in_file": False, "add_monitor": False,
                         "skip_feedback": skip_feedback, "stable_system": stable_system, "custom_xinit": x_init,
                         "simulation_length": simulation_length, "x_desired": x_desired, "problem_setup_fn": problem_setup_fn,
                         "ignore_init_constr_check": ignore_init_constr_check, "satis_prob": satisfaction_prob}
        curr_init_info_dict = {"gp_ds_inst": gp_ds_inst, "u_start_feas_gen": u_start_feas_gen, "u_end_feas_gen": u_end_feas_gen,
                               "piecewise_warmstart": piecewise_warmstart, "global_warmstart": global_warmstart,
                               "warmstart_dict_piecewise": warmstart_dict_piecewise, "warmstart_dict_global": warmstart_dict_global,
                               "test_config_opts": common_kwargs, "system_setup_fn": problem_setup_fn, "test_softplus": test_softplus
                               }
        all_accd_info["run_info"].append(curr_init_info_dict)

        print()
        curr_init_info_dict["run_accumulated_info_dicts"] = []
        print("Random init attempt: %s with x_init = %s" % (setup_idx, x_init.squeeze()))
        for run_idx in range(num_runs_per_init):
            print("Run %s" % run_idx, end="\t")
            run_data_dict = {"start_time": dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}
            print("Starting piecewise run.", end=" ")
            controller_inst_piecewise, X_test, U_test,\
            data_dict_cl_piecewise, sol_stats_cl_piecewise = GPR_test_hardcoded_allfeatsactive_piecewise(**common_kwargs,
                                                                                                         warmstart=piecewise_warmstart,
                                                                                                         warmstart_dict=warmstart_dict_piecewise,
                                                                                                         gp_fns=piecewise_gp_wrapped,
                                                                                                         show_plot=False,
                                                                                                         verbose=verbose,
                                                                                                         test_softplus=test_softplus)
            print("Piecewise run solved", end=" ")
            if not ignore_global:
                print("Starting global run.", end=" ")
                controller_inst_global, X_test, U_test,\
                data_dict_cl_global, sol_stats_cl_global = GPR_test_hardcoded_allfeatsactive_global(**common_kwargs,
                                                                                                    warmstart=global_warmstart,
                                                                                                    warmstart_dict=warmstart_dict_global,
                                                                                                    gp_fns=global_gp_wrapped,
                                                                                                    verbose=verbose)
                print("Global run solved", end="\n")
            run_data_dict.update({"X_test": X_test, "U_test": U_test,
                                  "data_dict_cl_piecewise": data_dict_cl_piecewise, "sol_stats_cl_piecewise": sol_stats_cl_piecewise})
            if not ignore_global:
                run_data_dict.update({"data_dict_cl_global": data_dict_cl_global, "sol_stats_cl_global": sol_stats_cl_global})
            curr_init_info_dict["run_accumulated_info_dicts"].append(run_data_dict)

            with open(store_dir+datastore_file, "wb") as pklfile:
                existing_data[-1] = all_accd_info
                pkl.dump(existing_data, pklfile)



def retrieve_vals_from_setup_fn(reqd_vars, problem_setup_fn):
    retrieved_vars = {}
    for var_name in reqd_vars:
        retrieved_vars[var_name] = problem_setup_fn()[var_name]
    return retrieved_vars


def fwd_sim_true_means(gp_ds_inst: GP_DS, mu_x_ol, mu_u_ol, problem_setup_fn, N):
    reqd_vars = ["gp_input_mask", "delta_input_mask", "gp_inputs", "delta_control_variables",
                 "n_d", "A_test", "B_test", "Bd_test"]
    mu_x_init = mu_x_ol[:, [0]]
    retrieved_vars = retrieve_vals_from_setup_fn(reqd_vars, problem_setup_fn)
    condn1 = (retrieved_vars["delta_input_mask"] == retrieved_vars["gp_input_mask"]).all()
    # Roundabout way of showing that the variables involved are only the state variables i.e. applying input mask to joint
    # state-input vector yields out only the state vector
    condn2 = (retrieved_vars["gp_input_mask"] @ (np.vstack([mu_x_init, mu_u_ol[:, [0]]])) == mu_x_ol[:, [0]]).all()
    assert condn1 and condn2, "Implementation currently assumes the state variables are involved in both" \
                              " region specifications and gp inputs"
    mu_d_ol = np.zeros([retrieved_vars["n_d"], N])
    true_mu_x_ol = np.zeros([*mu_x_ol.shape])
    true_mu_x_ol[:, [0]] = mu_x_init

    for k in range(N):
        mu_x_k, mu_u_k = true_mu_x_ol[:, [k]], mu_u_ol[:, [k]]
        mu_z_k = np.vstack([mu_x_k, mu_u_k])
        gp_inp = torch.from_numpy((retrieved_vars["gp_input_mask"] @ mu_z_k).astype(np.float32))

        # Generate the residual term directly from the true underlying residual dynamics function.
        mu_d_ol[:, [k]] = gp_ds_inst.generate_outputs(input_arr=gp_inp,
                                                      no_noise=True,
                                                      return_op=True)
        mu_d_k = mu_d_ol[:, [k]]
        true_mu_x_ol[:, [k+1]] = retrieved_vars["A_test"] @ mu_x_k + retrieved_vars["B_test"] @ mu_u_k + retrieved_vars["Bd_test"] @ mu_d_k
    return true_mu_x_ol


def get_true_res_cov(gp_ds_inst: GP_DS, true_mu_x_ol, mu_u_ol, problem_setup_fn, N):
    reqd_vars = ["gp_input_mask", "delta_input_mask", "gp_inputs", "delta_control_variables",
                 "regions", "n_d"]
    retrieved_vars = retrieve_vals_from_setup_fn(reqd_vars, problem_setup_fn)
    mu_z_ol = np.vstack([true_mu_x_ol[:, :-1], mu_u_ol])
    region_masks = gp_ds_inst._generate_regionspec_mask(input_arr=(retrieved_vars["delta_input_mask"] @ mu_z_ol))
    regionwise_sample_idxs = gp_ds_inst.get_region_idxs(mask_input=region_masks)
    Sigma_d_ol = [np.zeros([retrieved_vars["n_d"], retrieved_vars["n_d"]]) for _ in range(N)]
    for region_idx, region in enumerate(retrieved_vars["regions"]):
        region_noise_var = gp_ds_inst.noise_vars[region_idx]
        # Doesn't necessarily need to be 1 mean in every region so some region_idxs won't have sample_idxs associated
        # with them and can be ignored.
        try:
            for idx in regionwise_sample_idxs[region_idx][0]:
                Sigma_d_ol[idx.item()] = np.array(region_noise_var, ndmin=2)
        except KeyError:
            continue

    return Sigma_d_ol


def fwd_sim_true_shrunk(gp_ds_inst: GP_DS, true_mu_x_ol, mu_u_ol, problem_setup_fn, N):
    Sigma_d_ol = get_true_res_cov(gp_ds_inst, true_mu_x_ol, mu_u_ol, problem_setup_fn, N)
    # print("True res cov finished")
    reqd_vars = ["A_test", "B_test", "Bd_test", "Q_test", "R_test", "n_d", "n_x", "n_u", "X_test", "U_test"]
    retrieved_vars = retrieve_vals_from_setup_fn(reqd_vars, problem_setup_fn)
    K, _ = setup_terminal_costs(retrieved_vars["A_test"], retrieved_vars["B_test"],
                                retrieved_vars["Q_test"], retrieved_vars["R_test"])
    computesigma_wrapped = computeSigma_meaneq('Sigma',
                                               feedback_mat=K,
                                               residual_dim=retrieved_vars["n_d"],
                                               opts={"enable_fd": True})
    affine_transform = np.concatenate((retrieved_vars["A_test"], retrieved_vars["B_test"], retrieved_vars["Bd_test"]), axis=1)
    Sigma_x = [np.zeros([retrieved_vars["n_x"], retrieved_vars["n_x"]])]
    Sigma_u = []
    X, U = retrieved_vars["X_test"], retrieved_vars["U_test"]
    b_shr_x_true, b_shr_u_true = np.zeros([2*retrieved_vars["n_x"], N+1]), np.zeros([2*retrieved_vars["n_u"], N])
    b_shr_x_true[:, [0]], b_shr_u_true[:, [0]] = X.b_np, U.b_np
    sqrt_const = 1e-4
    inverse_cdf_x = get_inv_cdf(retrieved_vars["n_x"], problem_setup_fn().get("satisfaction_prob", setup_problem_basic()["satisfaction_prob"]))
    inverse_cdf_u = get_inv_cdf(retrieved_vars["n_u"], problem_setup_fn().get("satisfaction_prob", setup_problem_basic()["satisfaction_prob"]))
    for k in range(N):
        Sigma_x_k = Sigma_x[k]
        Sigma_d_k = Sigma_d_ol[k]
        Sigma_u_k = K @ Sigma_x_k @ K.T
        Sigma_u.append(Sigma_u_k)
        Sigma_i = computesigma_wrapped(Sigma_x_k, Sigma_u_k, Sigma_d_k)
        Sigma_x.append(affine_transform @ Sigma_i @ affine_transform.T)
        if k > 0:
            b_shr_x_true[:, [k]] = X.b_np - (cs.fabs(X.H_np) @ (cs.sqrt(cs.diag(Sigma_x_k)) * inverse_cdf_x))
            b_shr_u_true[:, [k]] = U.b_np - (cs.fabs(U.H_np) @ (cs.sqrt(cs.diag(Sigma_u_k)) * inverse_cdf_u))
    b_shr_x_true[:, [-1]] = X.b_np - (cs.fabs(X.H_np) @ (cs.sqrt(cs.diag(Sigma_x[-1])) * inverse_cdf_x))
    return b_shr_x_true, b_shr_u_true


def postproc_genddata(datastore_file='init_20_runs_5_samples_100.pkl', shrunk_errors=True):
    with open("C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\"+datastore_file, "rb") as pklfile:
        stored_data = pkl.load(pklfile)
    # print(stored_data)
    run_retrieved_info = stored_data[-1]["run_info"]
    # print(run_retrieved_info)
    pw_errors, glob_errors = [], []
    for init_idx in range(len(run_retrieved_info)):
        curr_init_info = run_retrieved_info[init_idx]
        gp_ds_inst = curr_init_info["gp_ds_inst"]
        problem_setup_fn = curr_init_info["system_setup_fn"]
        N = curr_init_info["test_config_opts"]["N"]
        simulation_length = curr_init_info["test_config_opts"]["simulation_length"]
        # piecewise_gp_wrapped, global_gp_wrapped, _ = gen_ds_and_train(hardcoded=False, fixed_numeric_means=False, viz=False, verbose=False,
        #                                                               problem_setup_fn=curr_init_info["system_setup_fn"], ds_inst_in=gp_ds_inst)
        # print("Global and piecewise models trained for outer run %s" % init_idx)
        curr_init_run_accd = curr_init_info["run_accumulated_info_dicts"]
        for inner_run_idx in range(len(curr_init_run_accd)):
            pw_cl_data = curr_init_run_accd[inner_run_idx]["data_dict_cl_piecewise"]
            glob_cl_data = curr_init_run_accd[inner_run_idx]["data_dict_cl_global"]
            accd_run_data_pw, accd_run_data_glob = {}, {}
            for var_name in pw_cl_data[0].keys():
                accd_run_data_pw[var_name] = [pw_ol_data[var_name] for pw_ol_data in pw_cl_data]
            # print(pw_cl_data)
            # print()
            # print(accd_run_data_pw)
            for var_name in glob_cl_data[0].keys():
                accd_run_data_glob[var_name] = [glob_ol_data[var_name] for glob_ol_data in glob_cl_data]
            # print(glob_cl_data)
            # print()
            # print(accd_run_data_glob)

            # print(accd_run_data_pw)
            pw_error, glob_error = 0, 0
            print("Init iter: %s Run: %s " % (init_idx, inner_run_idx), end=" ")
            for t in range(simulation_length):
                # Note: mu_x_ol doesn't need to be pw_mu_x_ol since only the initial state information is extracted which
                # is common for both the global and piecewise case but we just do so for intuition
                pw_mu_x_ol = np.array(accd_run_data_pw["mu_x"][t], ndmin=2)
                glob_mu_x_ol = np.array(accd_run_data_glob["mu_x"][t], ndmin=2)
                pw_mu_u_ol = np.array(accd_run_data_pw["mu_u"][t], ndmin=2)
                glob_mu_u_ol = np.array(accd_run_data_glob["mu_u"][t], ndmin=2)
                # print("Start pw fwd sim")
                pw_true_mu_x_ol = fwd_sim_true_means(gp_ds_inst, pw_mu_x_ol, pw_mu_u_ol, problem_setup_fn, N)
                if shrunk_errors:
                    # print("Mean sim finished")
                    pw_b_shr_x_true, pw_b_shr_u_true = fwd_sim_true_shrunk(gp_ds_inst, pw_true_mu_x_ol, pw_mu_u_ol, problem_setup_fn, N)
                    # print("Shrunk sim finished")
                    # print("Piecewise fwd sim complete")
                    glob_true_mu_x_ol = fwd_sim_true_means(gp_ds_inst, glob_mu_x_ol, glob_mu_u_ol, problem_setup_fn, N)
                    glob_b_shr_x_true, glob_b_shr_u_true = fwd_sim_true_shrunk(gp_ds_inst, glob_true_mu_x_ol, glob_mu_u_ol, problem_setup_fn, N)
                    # print("Global fwd sim complete")
                    for k in range(N+1):
                        # print(pw_b_shr_x_true[:, [k]], accd_run_data_pw["b_shrunk_x"])
                        # print(accd_run_data_pw["b_shrunk_x"][t])
                        pw_error += np.linalg.norm(pw_b_shr_x_true[:, [k]] - accd_run_data_pw["b_shrunk_x"][t][k])
                        glob_error += np.linalg.norm(glob_b_shr_x_true[:, [k]] - accd_run_data_glob["b_shrunk_x"][t][k])
                    print("t=%s complete" % t, end=" ")
            if shrunk_errors:
                print("Current accd Error: (pw_error, global_error) = (%s, %s)" % (pw_error, glob_error))
                if pw_error < glob_error:
                    pw_errors.append(pw_error)
                    glob_errors.append(glob_error)
                    # print(pw_error, glob_error)
                # print(b_shr_x_true)
                # print(b_shr_u_true)
                # print(accd_run_data_pw["b_shrunk_x"][t])
                # print(accd_run_data_pw["b_shrunk_u"][t])
                # print(accd_run_data_glob["b_shrunk_x"][t])
                # print(accd_run_data_glob["b_shrunk_u"][t])

                # break
            # break
        # break
        with open("C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\shrunk_metric.pkl", "wb") as pklfile:
            pkl.dump([pw_errors, glob_errors], pklfile)


def verify_infeas_problem(retrieval_file='subset_stabil_small_ctrl_stable_100samples.pkl',
                          datastore_file=None):
    store_dir = "C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\"
    with open(store_dir+retrieval_file, "rb") as pklfile:
        stored_data = pkl.load(pklfile)
    if datastore_file is None:
        datastore_file = "rerun_"+retrieval_file
    run_retrieved_info = stored_data[-1]["run_info"]
    for init_idx in range(len(run_retrieved_info)):
        curr_init_info = run_retrieved_info[init_idx]
        gp_ds_inst = curr_init_info["gp_ds_inst"]
        global_warmstart = curr_init_info["global_warmstart"]
        warmstart_dict_global = curr_init_info["warmstart_dict_global"]
        problem_setup_fn = curr_init_info["system_setup_fn"]
        print("Training models")
        piecewise_gp_wrapped, global_gp_wrapped, _ = gen_ds_and_train(hardcoded=False, fixed_numeric_means=False, viz=False, verbose=False,
                                                                      problem_setup_fn=problem_setup_fn, ds_inst_in=gp_ds_inst)
        print("Global and piecewise models trained for outer run %s" % init_idx)
        run_accumulated_info_dicts = curr_init_info["run_accumulated_info_dicts"]
        common_kwargs = curr_init_info["test_config_opts"]
        common_kwargs["minimal_print"] = False
        # print(run_accumulated_info_dicts)
        for run_data_dict in run_accumulated_info_dicts:
            sol_stats_cl_global = run_data_dict["sol_stats_cl_global"]
            data_dict_cl_global = run_data_dict["data_dict_cl_global"]
            data_dict_cl_pw = run_data_dict["data_dict_cl_piecewise"]

            problematic_timesteps = []
            infeas_issue = False
            for timestep, sol_stats_ol_global in enumerate(sol_stats_cl_global):
                # checking sol_stats_ol_global["success"] False doesn't necessarily give bad runs since it includes ones
                # that converge to a point of local infeasibility even if the primal has low constraint violation.
                if sol_stats_ol_global["return_status"] == "Restoration_Failed":
                    problematic_timesteps.append(timestep)
                    infeas_issue = True
                print(sol_stats_ol_global["return_status"])

            accd_run_data_glob = {"mu_u": [glob_ol_data["mu_u"] for glob_ol_data in data_dict_cl_global]}
            accd_run_data_pw = {"mu_u": [pw_ol_data["mu_u"] for pw_ol_data in data_dict_cl_pw]}
            mu_u_cl_glob = np.hstack([mu_u_ol[0] for mu_u_ol in accd_run_data_glob["mu_u"]])
            mu_u_cl_pw = np.hstack([mu_u_ol[0] for mu_u_ol in accd_run_data_pw["mu_u"]])
            print(mu_u_cl_glob)
            print(mu_u_cl_pw)

            if infeas_issue:
                print("Found run with infeas issues failed at timestep(s): %s" % problematic_timesteps)
                controller_inst_global, X_test, U_test, \
                data_dict_cl_global, sol_stats_cl_global = GPR_test_hardcoded_allfeatsactive_global(**common_kwargs,
                                                                                                    warmstart=global_warmstart,
                                                                                                    warmstart_dict=warmstart_dict_global,
                                                                                                    gp_fns=global_gp_wrapped,
                                                                                                    verbose=True)
                run_data_dict.update({"data_dict_cl_global": data_dict_cl_global, "sol_stats_cl_global": sol_stats_cl_global})
                break
    with open(store_dir+datastore_file, "wb") as pklfile:
        pkl.dump(stored_data, pklfile)


def viz_stabilizatn_traj(datastore_file='final_edgexdes_stabil_small_ctrl_stable_100samples.pkl',
                         ignore_global=False, check_viol=False, compare_cost=False, idx=-1,
                         plot_regions=True, outer_idx=0, inner_idx=0, y_lim=(0, 2.2), x_lim=(-0.75, 2.5),
                         set_point=(2, 2), pw_fig_file_name="cl_pw_edgetrack", glob_fig_file_name="cl_glob_edgetrack",
                         verbose=True):
    with open("C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\"+datastore_file, "rb") as pklfile:
        stored_data = pkl.load(pklfile)
    # print(len(stored_data))
    run_retrieved_info = stored_data[idx]["run_info"]
    # reqd_args = ["noise_std_devs", "x_init_limits", "only_neg_xinit", "only_pos_xinit"]
    # for arg in reqd_args:
    #     print(arg, stored_data[idx]["config_info"][arg])
    # print("N", run_retrieved_info[0]["test_config_opts"]["N"])
    # noise_std_devs = stored_data[idx]["config_info"]["noise_std_devs"]
    # print(noise_std_devs)
    # print(stored_data[-1]["config_info"]["x_init_limits"])
    for init_idx in range(outer_idx, len(run_retrieved_info)):
        curr_init_info = run_retrieved_info[init_idx]
        problem_setup_fn = curr_init_info["system_setup_fn"]
        regions = problem_setup_fn()["regions"]
        curr_init_run_accd = curr_init_info["run_accumulated_info_dicts"]
        try:
            x_desired = curr_init_info["test_config_opts"]["x_desired"]
        except KeyError:
            x_desired = None
        simulation_length = curr_init_info["test_config_opts"]["simulation_length"]
        if x_desired is None:
            x_desired = np.zeros([problem_setup_fn()["n_x"], simulation_length])
        for inner_run_idx in range(inner_idx, len(curr_init_run_accd)):
            pw_cl_data = curr_init_run_accd[inner_run_idx]["data_dict_cl_piecewise"]
            X_test: box_constraint = curr_init_run_accd[inner_run_idx]["X_test"]
            if not ignore_global:
                glob_cl_data = curr_init_run_accd[inner_run_idx]["data_dict_cl_global"]
            accd_run_data_pw, accd_run_data_glob = {}, {}
            for var_name in pw_cl_data[0].keys():
                accd_run_data_pw[var_name] = [pw_ol_data[var_name] for pw_ol_data in pw_cl_data]
            if not ignore_global:
                for var_name in glob_cl_data[0].keys():
                    accd_run_data_glob[var_name] = [glob_ol_data[var_name] for glob_ol_data in glob_cl_data]

            fig, axes = plt.subplots(1, (4 if not ignore_global else 2), figsize=(50, 12))
            colours = ['r', 'g', 'b']
            if verbose:
                print(accd_run_data_pw["mu_u"])
                if not ignore_global:
                    print(accd_run_data_glob["mu_u"])
                print(accd_run_data_pw["mu_x"])

            if plot_regions:
                for ax_idx in range(len(axes)):
                    ax = axes[ax_idx]
                    facecolours = ["lightsalmon", "cyan", "limegreen"]
                    edgecolours = ["orangered", "blue", "green"]
                    for region_idx, region in enumerate(regions):
                        region: box_constraint
                        x1, x2 = gen_square_coords2(region.lb, region.ub)
                        ax.fill(x1, x2, facecolor=facecolours[region_idx], edgecolor=edgecolours[region_idx], alpha=0.6,
                                linewidth=0)
                        ax.set_ylim(y_lim)
                        ax.set_xlim(x_lim)
                    ax.plot(*(set_point), color='g', marker='o', markersize=12, label='Set-point to track')
                    ax.set_xlabel("State 1", fontsize=25)
                    ax.set_ylabel("State 2", fontsize=25)
                    x1, x2 = gen_square_coords2(X_test.lb, X_test.ub)
                    ax.fill(x1, x2, facecolor='none', edgecolor='magenta', alpha=0.6,
                            linewidth=2, linestyle='dashed', label='State constraint set')
                    if ax_idx in [1, 3]:
                        ax.xaxis.set_tick_params(labelsize=25)
                        ax.yaxis.set_tick_params(labelsize=25)

            # print(problem_setup_fn()["A_test"])
            for timestep, mu_x_ol in enumerate(accd_run_data_pw["mu_x"]):
                mu_x_ol = np.array(mu_x_ol, ndmin=2)
                axes[0].plot(mu_x_ol[0, :].squeeze(), mu_x_ol[1, :].squeeze(), color=colours[timestep % 3], marker='x',
                             linestyle='dashed', linewidth=2, markersize=12, label='OL output: Timestep: %s' % timestep)
                axes[0].legend(loc='upper center', fontsize=15)
            if not ignore_global:
                for timestep, mu_x_ol in enumerate(accd_run_data_glob["mu_x"]):
                    mu_x_ol = np.array(mu_x_ol, ndmin=2)
                    axes[2].plot(mu_x_ol[0, :].squeeze(), mu_x_ol[1, :].squeeze(), color=colours[timestep % 3], marker='x',
                                 linestyle='dashed', linewidth=2, markersize=12, label='OL output: Timestep: %s' % timestep)
                    axes[2].legend(loc='upper center')
            mu_x_cl_pw = np.hstack([mu_x_ol[:, [0]] for mu_x_ol in accd_run_data_pw["mu_x"]])
            mu_u_cl_pw = np.hstack([mu_u_ol[0] for mu_u_ol in accd_run_data_pw["mu_u"]])
            axes[1].plot(mu_x_cl_pw[0, :].squeeze(), mu_x_cl_pw[1, :].squeeze(), color='b', marker='x',
                         markevery=range(1, mu_x_cl_pw.shape[-1]),
                         linestyle='solid', linewidth=1.5, markersize=10, label='Piecewise CL trajectory')
            axes[1].plot(mu_x_cl_pw[0, 0], mu_x_cl_pw[1, 0], color='r', marker='o', markersize=10, label='Initial state')
            axes[1].legend(loc='upper left', fontsize=17)

            if not ignore_global:
                mu_x_cl_glob = np.hstack([mu_x_ol[:, [0]] for mu_x_ol in accd_run_data_glob["mu_x"]])
                mu_u_cl_glob = np.hstack([mu_u_ol[0] for mu_u_ol in accd_run_data_glob["mu_u"]])
                colours = ['r'] + ['b']*(mu_x_cl_glob.shape[-1]-1)
                markers = ['o'] + ['x']*(mu_x_cl_glob.shape[-1]-1)
                axes[3].plot(mu_x_cl_glob[0, :].squeeze(), mu_x_cl_glob[1, :].squeeze(), color='b', marker='x',
                             markevery=range(1, mu_x_cl_glob.shape[-1]),
                             linestyle='solid', linewidth=1.5, markersize=10, label='Global CL trajectory')
                axes[3].plot(mu_x_cl_glob[0, 0], mu_x_cl_glob[1, 0], color='r', marker='o', markersize=10, label='Initial state')
                axes[3].legend(loc='upper left', fontsize=17)

            save_subplot(axes[1], fig, fig_name=pw_fig_file_name, sub_path="planar_plots\\")
            save_subplot(axes[3], fig, fig_name=glob_fig_file_name, sub_path="planar_plots\\")

            if check_viol:
                pw_viols, glob_viols = 0, 0
                viol_amt_pw, viol_amt_glob = 0, 0
                for t in range(mu_x_cl_pw.shape[-1]):
                    if not ignore_global:
                        mu_x_t_glob = mu_x_cl_glob[:, [t]]
                    mu_x_t_pw = mu_x_cl_pw[:, [t]]
                    for dim in range(mu_x_t_pw.shape[0]):
                        abs_dim_limit = np.abs(X_test.b_np[dim, :])
                        if not ignore_global:
                            abs_dim_val_glob = np.abs(mu_x_t_glob[dim, 0])
                            if abs_dim_val_glob > abs_dim_limit:
                                viol_amt_glob += (abs_dim_val_glob - abs_dim_limit)
                        abs_dim_val_pw = np.abs(mu_x_t_pw[dim, 0])
                        if abs_dim_val_pw > abs_dim_limit:
                            viol_amt_pw += (abs_dim_val_pw - abs_dim_limit)
                    # print(mu_x_t_pw)
                    # print(X_test.check_satisfaction(mu_x_t_pw.T))
                    # # print(X_test)
                    # print(mu_x_t_glob)
                    # print(X_test.check_satisfaction(mu_x_t_glob.T))
                    pw_viols += (1 if X_test.check_satisfaction(mu_x_t_pw.T).item() is False else 0)
                    if not ignore_global:
                        glob_viols += (1 if X_test.check_satisfaction(mu_x_t_glob.T).item() is False else 0)
                print(pw_viols, glob_viols, viol_amt_pw, viol_amt_glob)

            if compare_cost:
                pw_cost, glob_cost = 0, 0
                Q, R = problem_setup_fn()["Q_test"], problem_setup_fn()["R_test"]
                for t in range(mu_x_cl_glob.shape[-1]):
                    mu_x_t_glob = mu_x_cl_glob[:, [t]]
                    mu_x_t_glob_dev = mu_x_t_glob - x_desired[:, [t]]
                    mu_u_t_glob = mu_u_cl_glob[t]
                    mu_x_t_pw = mu_x_cl_pw[:, [t]]
                    mu_x_t_pw_dev = mu_x_t_pw - x_desired[:, [t]]
                    mu_u_t_pw = mu_u_cl_pw[t]
                    pw_cost += (mu_x_t_pw_dev.T @ Q @ mu_x_t_pw_dev + mu_u_t_pw[None].T @ R @ mu_u_t_pw[None])
                    glob_cost += (mu_x_t_glob_dev.T @ Q @ mu_x_t_glob_dev + mu_u_t_glob[None].T @ R @ mu_u_t_glob[None])
                print(glob_cost, pw_cost)
            break
        break
    if compare_cost:
        try:
            if check_viol:
                return glob_cost, pw_cost, pw_viols, glob_viols, viol_amt_pw, viol_amt_glob
            else:
                return glob_cost, pw_cost
        except Exception as e:
            print(e)
            return False
    else:
        return None


def metric_evaluation(idxs=(-1,), datastore_file="stabilizatn_cost_comp.pkl"):
    pw_costs, glob_costs = [], []
    for idx in idxs:
        outer_idx, inner_idx = 0, 0
        no_terminate = True
        while no_terminate:
            print(outer_idx, inner_idx)
            error = False
            try:
                costs = viz_stabilizatn_traj(ignore_global=False, check_viol=False, compare_cost=True, datastore_file=datastore_file,
                                             idx=idx, outer_idx=outer_idx, inner_idx=inner_idx, verbose=False)
                print(costs)
            except Exception as e:
                print(e)
                error = True
                costs = 1
            if costs is False:
                if inner_idx == 0:
                    no_terminate = False
                else:
                    inner_idx = 0
                    outer_idx += 1
            else:
                inner_idx += 1
                if error:
                    continue
                else:
                    pw_costs.append(costs[1])
                    glob_costs.append(costs[0])
    pw_costs = np.array(pw_costs)
    glob_costs = np.array(glob_costs)
    print(np.mean(pw_costs), np.sqrt(np.sum((np.mean(pw_costs[np.nonzero(pw_costs < 70)]) - pw_costs[np.nonzero(pw_costs < 70)])**2)))
    print(np.mean(glob_costs), np.sqrt(np.sum((np.mean(glob_costs) - glob_costs)**2)))


def constr_viol_evaluation(idxs=(-1,), datastore_file="stabilizatn_cost_comp.pkl"):
    pw_costs, glob_costs = [], []
    pw_viols, glob_viols, viol_amts_pw, viol_amts_glob = [], [], [], []
    for idx in idxs:
        outer_idx, inner_idx = 0, 0
        no_terminate = True
        while no_terminate:
            print(outer_idx, inner_idx)
            error = False
            try:
                metric_ops = viz_stabilizatn_traj(ignore_global=False, check_viol=True,
                                                  compare_cost=True, datastore_file=datastore_file,
                                                  idx=idx, outer_idx=outer_idx, inner_idx=inner_idx, verbose=False)
                if metric_ops is not False:
                    glob_cost, pw_cost, pw_viol, glob_viol, viol_amt_pw, viol_amt_glob = metric_ops
                # print("here")
                # print(costs)
            except Exception as e:
                print(e)
                error = True
                metric_ops = 1
            if metric_ops is False:
                if inner_idx == 0:
                    no_terminate = False
                else:
                    inner_idx = 0
                    outer_idx += 1
            else:
                inner_idx += 1
                if error:
                    continue
                else:
                    pw_costs.append(pw_cost)
                    glob_costs.append(glob_cost)
                    pw_viols.append(pw_viol)
                    glob_viols.append(glob_viol)
                    viol_amts_pw.append(viol_amt_pw)
                    viol_amts_glob.append(viol_amt_glob)
    pw_costs_np = np.array(pw_costs)
    glob_costs_np = np.array(glob_costs)
    print(np.mean(pw_costs_np), np.sqrt(pw_costs_np.var()))
    print(np.mean(glob_costs_np), np.sqrt(glob_costs_np.var()))
    for viol_metric in [pw_viols, glob_viols, viol_amts_pw, viol_amts_glob]:
        print(np.mean(viol_metric), np.sqrt(np.array(viol_metric).var()))


def constr_viol_evaluation2(idxs=(-1,), datastore_file="stabilizatn_cost_comp.pkl"):
    pw_costs, glob_costs = [], []
    pw_viols, glob_viols, viol_amts_pw, viol_amts_glob = [], [], [], []
    for idx in idxs:
        outer_idx, inner_idx = 0, 0
        inner_idx_lim = 2
        no_terminate = True
        while no_terminate:
            print(outer_idx, inner_idx)
            error = False
            try:
                metric_ops = viz_stabilizatn_traj(ignore_global=False, check_viol=True,
                                                  compare_cost=True, datastore_file=datastore_file,
                                                  idx=idx, outer_idx=outer_idx, inner_idx=inner_idx, verbose=False)
                if metric_ops is not False:
                    glob_cost, pw_cost, pw_viol, glob_viol, viol_amt_pw, viol_amt_glob = metric_ops
                # print("here")
                # print(costs)
            except Exception as e:
                print(e)
                error = True
            else:
                inner_idx += 1
                if inner_idx == inner_idx_lim:
                    inner_idx = 0
                    outer_idx += 1
                if outer_idx == 3:
                    no_terminate = False
            if not error:
                pw_costs.append(pw_costs)
                glob_costs.append(glob_costs)
                pw_viols.append(pw_viol)
                glob_viols.append(glob_viol)
                viol_amts_pw.append(viol_amt_pw)
                viol_amts_glob.append(viol_amt_glob)
            pw_costs = np.array(pw_costs)
            glob_costs = np.array(glob_costs)
            print(np.mean(pw_costs), np.sqrt(np.sum((np.mean(pw_costs[np.nonzero(pw_costs < 70)]) - pw_costs[np.nonzero(pw_costs < 70)])**2)))
            print(np.mean(glob_costs), np.sqrt(np.sum((np.mean(glob_costs) - glob_costs)**2)))
            for viol_metric in [pw_viols, glob_viols, viol_amts_pw, viol_amts_glob]:
                print(np.mean(viol_metric), np.sqrt(np.sum((viol_metric - np.mean(viol_metric))**2)))


def gen_square_coords(lbs):
    dim_0_coord, dim_1_coord = lbs[0, :].item(), lbs[1, :].item()
    return (dim_0_coord, dim_0_coord, -dim_0_coord, -dim_0_coord), (dim_1_coord, -dim_1_coord, -dim_1_coord, dim_1_coord)


def gen_square_coords2(lbs, ubs):
    dim_1_ub, dim_1_lb, dim_0_ub, dim_0_lb = ubs[1, :].item(), lbs[1, :].item(), ubs[0, :].item(), lbs[0, :].item(),
    return (dim_0_ub, dim_0_ub, dim_0_lb, dim_0_lb), (dim_1_ub, dim_1_lb, dim_1_lb, dim_1_ub)


def viz_set_shrinking(datastore_file='unstable_init_1_runs_1_samples_100_nofeedback.pkl'):
    with open("C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\"+datastore_file, "rb") as pklfile:
        stored_data = pkl.load(pklfile)
    run_retrieved_info = stored_data[-1]["run_info"]
    for init_idx in range(len(run_retrieved_info)):
        curr_init_info = run_retrieved_info[init_idx]
        N = curr_init_info["test_config_opts"]["N"]
        simulation_length = curr_init_info["test_config_opts"]["simulation_length"]
        curr_init_run_accd = curr_init_info["run_accumulated_info_dicts"]
        for inner_run_idx in range(len(curr_init_run_accd)):
            pw_cl_data = curr_init_run_accd[inner_run_idx]["data_dict_cl_piecewise"]
            glob_cl_data = curr_init_run_accd[inner_run_idx]["data_dict_cl_global"]
            accd_run_data_pw, accd_run_data_glob = {}, {}
            for var_name in pw_cl_data[0].keys():
                accd_run_data_pw[var_name] = [pw_ol_data[var_name] for pw_ol_data in pw_cl_data]
            for var_name in glob_cl_data[0].keys():
                accd_run_data_glob[var_name] = [glob_ol_data[var_name] for glob_ol_data in glob_cl_data]

            plt.figure()
            colours = ['r', 'g', 'b', 'cyan']
            print(accd_run_data_pw["b_shrunk_x"][0])
            for t in range(simulation_length):
                for k in range(N+1):
                    b_shrunk_ol = np.array(accd_run_data_pw["b_shrunk_x"][t][k], ndmin=2).T
                    dim_lbs = b_shrunk_ol[:2, :]
                    x1, x2 = gen_square_coords(dim_lbs)
                    plt.fill(x1, x2, color=colours[k], alpha=0.2*(k+1))
                break
            break
        break


def run_constr_viol_test(uni_comp=True):
    path_kwarg = {}
    if uni_comp:
        base_path = "/home/leroydsouza/PycharmProjects/GP_libraries/src"
        path_kwarg = {"base_path": base_path}
    data_gen_for_metrics(stable_system=True, num_initializations=5, num_runs_per_init=2, datastore_file='final_constr_viol_stable_200samples.pkl',
                         num_samples=200, simulation_length=20, edge_case_xdes=True, problem_setup_fn=setup_constr_viol_test,
                         N=3, x_init_limits=(0, 0.5), only_run_with_warmstart=True, test_softplus=True, skip_feedback=False,
                         noise_std_devs=(0.05, 0.2, 0.025), ignore_init_constr_check=True, ignore_global=False, **path_kwarg)


def run_stabilizatn_test(uni_comp=True):
    path_kwarg = {}
    if uni_comp:
        base_path = "/home/leroydsouza/PycharmProjects/GP_libraries/src"
        path_kwarg = {"base_path": base_path}
    data_gen_for_metrics(stable_system=True, num_initializations=5, num_runs_per_init=2, datastore_file='stabilizatn_cost_comp.pkl',
                         num_samples=100, simulation_length=15, edge_case_xdes=False, problem_setup_fn=cost_comp_stable,
                         N=3, x_init_limits=(1.25, 1.75), only_run_with_warmstart=True, test_softplus=True,
                         noise_std_devs=(0.05, 0.5, 0.15), only_neg_xinit=True, **path_kwarg)


def run_edge_tracking_test(uni_comp=True):
    path_kwarg = {}
    if uni_comp:
        base_path = "/home/leroydsouza/PycharmProjects/GP_libraries/src"
        path_kwarg = {"base_path": base_path}
    data_gen_for_metrics(stable_system=True, num_initializations=5, num_runs_per_init=2, datastore_file='boundary_cost_comp.pkl',
                         num_samples=150, simulation_length=15, edge_case_xdes=True, problem_setup_fn=cost_comp_boundary,
                         N=3, x_init_limits=(1.4, 1.5), only_run_with_warmstart=True, test_softplus=True,
                         noise_std_devs=(0.4, 0.05, 0.55), only_neg_xinit=False, only_pos_xinit=True, ignore_init_constr_check=True,
                         **path_kwarg)


def run_approx_tests(retrieval_file='subset_stabil_small_ctrl_stable_100samples.pkl',
                     datastore_file=None):
    store_dir = "C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\data\\"
    with open(store_dir+retrieval_file, "rb") as pklfile:
        stored_data = pkl.load(pklfile)
    if datastore_file is None:
        datastore_file = "pw_approx"+retrieval_file
    for run_idx, run in enumerate(stored_data):
        run_retrieved_info = run["run_info"]
        # satisfaction_prob = run["config_info"]["satisfaction_prob"]
        print("Processing approximation cases for run: %s / %s" % (run_idx+1, len(stored_data)))
        for init_idx in range(len(run_retrieved_info)):
            curr_init_info = run_retrieved_info[init_idx]
            gp_ds_inst = curr_init_info["gp_ds_inst"]
            piecewise_warmstart = curr_init_info["piecewise_warmstart"]
            warmstart_dict_piecewise = curr_init_info["warmstart_dict_piecewise"]
            problem_setup_fn = curr_init_info["system_setup_fn"]
            test_softplus = curr_init_info["test_softplus"]
            print("Training models")
            piecewise_gp_wrapped, global_gp_wrapped, _ = gen_ds_and_train(hardcoded=False, fixed_numeric_means=False, viz=False, verbose=False,
                                                                          problem_setup_fn=problem_setup_fn, ds_inst_in=gp_ds_inst)
            print("Global and piecewise models trained for outer run %s" % init_idx)
            run_accumulated_info_dicts = curr_init_info["run_accumulated_info_dicts"]
            common_kwargs = curr_init_info["test_config_opts"]
            common_kwargs["minimal_print"] = False
            # print(run_accumulated_info_dicts)
            for run_data_dict in run_accumulated_info_dicts:
                controller_inst_pw_approx, X_test, U_test,\
                data_dict_cl_pw_approx, sol_stats_cl_pw_approx = GPR_test_hardcoded_allfeatsactive_piecewise(**common_kwargs,
                                                                                                             warmstart=piecewise_warmstart,
                                                                                                             warmstart_dict=warmstart_dict_piecewise,
                                                                                                             gp_fns=piecewise_gp_wrapped,
                                                                                                             show_plot=False,
                                                                                                             verbose=False,
                                                                                                             test_softplus=test_softplus,
                                                                                                             approxd_shrinking=True)
                run_data_dict.update({"data_dict_cl_pw_approx": data_dict_cl_pw_approx, "sol_stats_cl_pw_approx": sol_stats_cl_pw_approx})
                del run_data_dict["sol_stats_cl_global"]
                del run_data_dict["data_dict_cl_global"]
            break
        with open(store_dir+datastore_file, "wb") as pklfile:
            pkl.dump(stored_data, pklfile)


def run_test(loops=1, uni_comp=True):
    for _ in range(loops):
        try:
            run_constr_viol_test(uni_comp=uni_comp)
        except Exception as e:
            print("Constr viol test errored with: ")
            print(e)
        try:
            run_edge_tracking_test(uni_comp=uni_comp)
        except Exception as e:
            print("Edge tracking test errored with: ")
            print(e)
        # try:
        #     run_approx_tests(retrieval_file='final_constr_viol_stable_200samples.pkl',
        #                      datastore_file='final_pw_approx_tests.pkl')
        # except Exception as e:
        #     print("Approx test errored with: ")
        #     print(e)

    for _ in range(loops):
        try:
            run_stabilizatn_test(uni_comp=uni_comp)
        except Exception as e:
            print("Stabilization test errored with: ")
            print(e)
