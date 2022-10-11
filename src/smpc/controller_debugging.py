import numpy as np
import casadi as cs
from .utils import np2bmatrix

from IPython.display import display, Math


class OptiDebugger:
    def __init__(self, controller_inst):
        self.controller_inst = controller_inst

    def get_vals_from_opti_debug(self, var_name):
        assert var_name in self.controller_inst.opti_dict.keys(), "Controller's opti_dict has no key: %s . Add it to the dictionary within the O.L. setups" % var_name
        if type(self.controller_inst.opti_dict[var_name]) in [list, tuple]:
            return [self.controller_inst.opti.debug.value(var_k) for var_k in self.controller_inst.opti_dict[var_name]]
        else:
            return self.controller_inst.opti.debug.value(self.controller_inst.opti_dict[var_name])


def retrieve_controller_results(controller_inst, X_test, U_test,
                                testing_validity=False, ignore_covs=False, return_data_dict=False, verbose=True):
    debugger_inst = OptiDebugger(controller_inst)
    data_dict = {}
    try:
        data_dict["mu_x"] = debugger_inst.get_vals_from_opti_debug("mu_x")
    except RuntimeError:
        data_dict["mu_x"] = None
        data_dict["run_failed"] = True
        return False, data_dict
    data_dict["mu_u"] = debugger_inst.get_vals_from_opti_debug("mu_u")
    data_dict["mu_d"] = debugger_inst.get_vals_from_opti_debug("mu_d")
    if not ignore_covs:
        data_dict["Sigma_x"] = debugger_inst.get_vals_from_opti_debug("Sigma_x")
    data_dict["b_shrunk_x"] = debugger_inst.get_vals_from_opti_debug("b_shrunk_x")
    data_dict["b_shrunk_u"] = debugger_inst.get_vals_from_opti_debug("b_shrunk_u")
    if verbose:
        display_controller_results(controller_inst, data_dict, ignore_covs, testing_validity, X_test, U_test)
    if return_data_dict:
        return debugger_inst, data_dict
    else:
        return debugger_inst


def display_controller_results(controller_inst, data_dict, ignore_covs, testing_validity, X_test, U_test):
    print("mu_x")
    print(data_dict["mu_x"])
    print("mu_u")
    print(data_dict["mu_u"])
    print("mu_d")
    print(data_dict["mu_d"])
    if not ignore_covs:
        print("Sigma_x")
        print(data_dict["Sigma_x"])
    print("X constraint b vector")
    print(X_test.b)
    print("Shrunk X b vectors over O.L. horizon")
    print(data_dict["b_shrunk_x"])
    if testing_validity:
        print("Testing symbolic method validity")
        print("Sigma_x at timestep 1")
        print(data_dict["Sigma_x"][1])
        # Note difference between output shape of np.diag and cs.diag
        # print("np.diag(Sigma_x[1]")
        # print(np.diag(Sigma_xval[1]))
        # print("cs.sqrt(cs.diag(Sigma_x[1]))")
        # print(cs.sqrt(cs.diag(Sigma_xval[1])))
        # print("shape of above matrix")
        # print(cs.sqrt(cs.diag(Sigma_xval[1])).shape)
        b_shrunk_x1 = X_test.b_np - (np.abs(X_test.H_np) @ (
                cs.sqrt(cs.diag(np.array(data_dict["Sigma_x"][1], ndmin=2))) * controller_inst.inverse_cdf_x))
        print("b_shrunk_x[1]")
        print(b_shrunk_x1)
    print("U constraint b vector")
    print(U_test.b)
    print("Shrunk U b vectors over O.L. horizons")
    print(data_dict["b_shrunk_u"])


def retrieve_controller_results_piecewise(controller_inst, X_test, U_test,
                                          ignore_lld=False, ignore_covs=False, return_data_dict=False, verbose=True):
    global_returns = retrieve_controller_results(controller_inst, X_test, U_test,
                                                 ignore_covs=ignore_covs, return_data_dict=return_data_dict, verbose=verbose)
    if return_data_dict:
        debugger_inst, data_dict = global_returns
    else:
        debugger_inst, data_dict = global_returns, {}
    if not ignore_lld:
        data_dict["lld"] = debugger_inst.get_vals_from_opti_debug("lld")
    try:
        data_dict["hld"] = debugger_inst.get_vals_from_opti_debug("hld")
    # Attribute Error results from data_dict being bool due to the global controller results yielded False
    except AttributeError:
        print("Errored out (most probably with MINLP Error)")
        return debugger_inst, data_dict
    if not ignore_covs:
        data_dict["Sigma_d"] = debugger_inst.get_vals_from_opti_debug("Sigma_d")
    if verbose:
        display_controller_results_piecewise(data_dict, ignore_lld, ignore_covs)
    if return_data_dict:
        return debugger_inst, data_dict
    else:
        return debugger_inst


def display_controller_results_piecewise(data_dict, ignore_lld, ignore_covs):
    if not ignore_lld:
        print("lld")
        print(data_dict["lld"])
    print("hld")
    print(data_dict["hld"])
    if not ignore_covs:
        print("Sigma d")
        print(data_dict["Sigma_d"])


def forward_simulate_test_global(controller_inst):
    N = controller_inst.N
    print("System Matrices")
    A, B, Bd, input_mask, K = controller_inst.A, controller_inst.B, controller_inst.Bd, controller_inst.input_mask, controller_inst.K
    affine_transform = controller_inst.affine_transform
    system_matrices = np2bmatrix([A, B, Bd, K], return_list=True)
    display(Math(r'A={}\,\,\;\,\,B={}\,\,\;\,\,Bd={}\,\,\;\,\,K={}'.format(*system_matrices)))
    # affine_transform = controller_inst.affine_transform
    debugger_inst = OptiDebugger(controller_inst)
    mu_x = np.array(debugger_inst.get_vals_from_opti_debug("mu_x"), ndmin=2)
    mu_u = np.array(debugger_inst.get_vals_from_opti_debug("mu_u"), ndmin=2)
    mu_d = np.array(debugger_inst.get_vals_from_opti_debug("mu_d"), ndmin=2)
    n_x, n_d, n_u = controller_inst.n_x, controller_inst.n_d, controller_inst.n_u
    Sigma = debugger_inst.get_vals_from_opti_debug("Sigma")
    Sigma_x = debugger_inst.get_vals_from_opti_debug("Sigma_x")
    Sigma_d = [np.array(Sigma_d_k, ndmin=2) for Sigma_d_k in debugger_inst.get_vals_from_opti_debug("Sigma_d")]
    print("State mean propagation equation")
    if controller_inst.gp_inputs == "state_only":
        display(Math(r'\mu^x_{}={}\mu^x_{} + {}\mu^u_{} + {}g(\mu^x_{})'.format("k+1", "A", "k", "B", "k", "B_d", "k")))
    else:
        display(Math(
            r'\mu^x_{}={}*\mu^x_{} + {}*\mu^u_{} + {}*g(\mu^x_{}, \mu^u_{})'.format("k+1", "A", "k", "B", "k", "B_d",
                                                                                    "k", "k")))
    print("State covariance propagation equation")
    display(Math(r'\Sigma^x_{k+1}=[A\,\, B\,\, B_d]\,\, \Sigma_k \,\,[A\,\, B\,\, B_d]^T'))
    print("where")
    display(Math(r'\Sigma=\begin{bmatrix}'
                 r'\Sigma^x&\Sigma^{xu}&\Sigma^{xd}\\\Sigma^{ux}&\Sigma^{u}&\Sigma^{ud}\\\Sigma^{dx}&\Sigma^{du}&\Sigma^{d}\\'
                 r'\end{bmatrix}'))
    for k in range(N):
        print("Step: %s" % k)
        matrices = [np2bmatrix(array) for array in [[A, mu_x[:, k]], [B, mu_u[:, k]], [Bd, mu_d[:, k]]]]
        # print(' '.join(matrices))
        print("Mean prop.")
        display(Math(r'\mu^x_{} = {}'.format(k + 1, '\,\,+\,\,'.join(matrices))))
        state_contrib, input_contrib, resid_contrib = A @ mu_x[:, k], B @ mu_u[:, k], Bd @ mu_d[:, k]
        display(Math(r'\,\,\,\,\,\,\, = {} + {} + {}'.format(np2bmatrix([state_contrib]),
                                                             np2bmatrix([input_contrib]),
                                                             np2bmatrix([resid_contrib]))))
        display(Math(r'\,\,\,\,\,\,\, = {}'.format(np2bmatrix([state_contrib + input_contrib + resid_contrib]))))
        print("Check against next state mean from Opti output")
        display(Math(r'\mu^x_{} = {}'.format(k + 1, np2bmatrix([mu_x[:, k + 1]]))))
        display(Math("\Sigma_k \,\,\\text{computation}"))
        Sigma_xu, Sigma_u_k = Sigma_x[k] @ K.T, K @ Sigma_x[k] @ K.T
        if controller_inst.gp_approx == "mean_eq":
            Sigma_xd, Sigma_ud = np.zeros((n_x, n_d)), np.zeros((n_u, n_d))
            display(Math(r'\Sigma^{}_{} = {}'.format("{xu}", k, np2bmatrix([Sigma_x[k], K.T]))))
            display(Math(r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '.format("{xu}", k, np2bmatrix([Sigma_xu]),
                                                               "{xd}", k, np2bmatrix([Sigma_xd]),
                                                               "{ud}", k, np2bmatrix([Sigma_ud]),
                                                               "{x}", k, np2bmatrix([Sigma_x[k]]),
                                                               "{u}", k, np2bmatrix([Sigma_u_k]),
                                                               "{d}", k, np2bmatrix([Sigma_d[k]]))))
            Sigma_computed = np.vstack([np.hstack([Sigma_x[k], Sigma_xu, Sigma_xd]),
                                        np.hstack([Sigma_xu.T, Sigma_u_k, Sigma_ud]),
                                        np.hstack([Sigma_xd.T, Sigma_ud.T, Sigma_d[k]])])
        display(Math(r"\Sigma_{} = {}".format(k, np2bmatrix([Sigma_computed]))))
        display(Math("\\text{Check against}\,\, \Sigma_k \,\,\\text{from Opti output}"))
        display(Math(r'\Sigma_{} = {}'.format(k, np2bmatrix([Sigma[k]]))))
        print("Covariance Prop.")
        display(
            Math(r'\Sigma^x_{} = {}'.format(k + 1, np2bmatrix([affine_transform, Sigma_computed, affine_transform.T]))))
        display(
            Math(r'\,\,\,\,\,\,\, = {}'.format(np2bmatrix([affine_transform @ Sigma_computed, affine_transform.T]))))
        display(
            Math(r'\,\,\,\,\,\,\, = {}'.format(np2bmatrix([affine_transform @ Sigma_computed @ affine_transform.T]))))
        display(Math("\\text{Check against}\,\, \Sigma^x_k \,\,\\text{from Opti output}"))
        display(Math(r'\Sigma^x_{} = {}'.format(k + 1, np2bmatrix([Sigma_x[k + 1]]))))
        print("Constraint shrinking on state")


def forward_simulate_test_piecewise(controller_inst):
    N = controller_inst.N
    print("System Matrices")
    A, B, Bd, input_mask, K = controller_inst.A, controller_inst.B, controller_inst.Bd, controller_inst.input_mask, controller_inst.K
    affine_transform = controller_inst.affine_transform
    system_matrices = np2bmatrix([A, B, Bd, K], return_list=True)
    display(Math(r'A={}\,\,\;\,\,B={}\,\,\;\,\,Bd={}\,\,\;\,\,K={}'.format(*system_matrices)))
    # affine_transform = controller_inst.affine_transform
    debugger_inst = OptiDebugger(controller_inst)
    mu_x = debugger_inst.get_vals_from_opti_debug("mu_x")
    mu_u = debugger_inst.get_vals_from_opti_debug("mu_u")
    mu_d = debugger_inst.get_vals_from_opti_debug("mu_d")
    n_x, n_d, n_u = controller_inst.n_x, controller_inst.n_d, controller_inst.n_u
    Sigma = debugger_inst.get_vals_from_opti_debug("Sigma")
    Sigma_x = debugger_inst.get_vals_from_opti_debug("Sigma_x")
    Sigma_d = debugger_inst.get_vals_from_opti_debug("Sigma_d")
    print("State mean propagation equation")
    if controller_inst.gp_inputs == "state_only":
        display(Math(r'\mu^x_{}={}\mu^x_{} + {}\mu^u_{} + {}g(\mu^x_{})'.format("k+1", "A", "k", "B", "k", "B_d", "k")))
    else:
        display(Math(
            r'\mu^x_{}={}*\mu^x_{} + {}*\mu^u_{} + {}*g(\mu^x_{}, \mu^u_{})'.format("k+1", "A", "k", "B", "k", "B_d",
                                                                                    "k", "k")))
    print("State covariance propagation equation")
    display(Math(r'\Sigma^x_{k+1}=[A\,\, B\,\, B_d]\,\, \Sigma_k \,\,[A\,\, B\,\, B_d]^T'))
    print("where")
    display(Math(r'\Sigma=\begin{bmatrix}'
                 r'\Sigma^x&\Sigma^{xu}&\Sigma^{xd}\\\Sigma^{ux}&\Sigma^{u}&\Sigma^{ud}\\\Sigma^{dx}&\Sigma^{du}&\Sigma^{d}\\'
                 r'\end{bmatrix}'))
    for k in range(N):
        print("Step: %s" % k)
        matrices = [np2bmatrix(array) for array in [[A, mu_x[:, k]], [B, mu_u[:, k]], [Bd, mu_d[k]]]]
        # print(' '.join(matrices))
        print("Mean prop.")
        display(Math(r'\mu^x_{} = {}'.format(k + 1, '\,\,+\,\,'.join(matrices))))
        state_contrib, input_contrib, resid_contrib = A @ mu_x[:, k], B @ mu_u[:, k], Bd @ mu_d[k]
        display(Math(r'\,\,\,\,\,\,\, = {} + {} + {}'.format(np2bmatrix([state_contrib]),
                                                             np2bmatrix([input_contrib]),
                                                             np2bmatrix([resid_contrib]))))
        display(Math(r'\,\,\,\,\,\,\, = {}'.format(np2bmatrix([state_contrib + input_contrib + resid_contrib]))))
        print("Check against next state mean from Opti output")
        display(Math(r'\mu^x_{} = {}'.format(k + 1, np2bmatrix([mu_x[:, k + 1]]))))
        display(Math("\Sigma_k \,\,\\text{computation}"))
        Sigma_xu, Sigma_u_k = Sigma_x[k] @ K.T, K @ Sigma_x[k] @ K.T
        if controller_inst.gp_approx == "mean_eq":
            Sigma_xd, Sigma_ud = np.zeros((n_x, n_d)), np.zeros((n_u, n_d))
            display(Math(r'\Sigma^{}_{} = {}'.format("{xu}", k, np2bmatrix([Sigma_x[k], K.T]))))
            display(Math(r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '
                         r'\Sigma^{}_{} = {}\,\,;\,\, '.format("{xu}", k, np2bmatrix([Sigma_xu]),
                                                               "{xd}", k, np2bmatrix([Sigma_xd]),
                                                               "{ud}", k, np2bmatrix([Sigma_ud]),
                                                               "{x}", k, np2bmatrix([Sigma_x[k]]),
                                                               "{u}", k, np2bmatrix([Sigma_u_k]),
                                                               "{d}", k, np2bmatrix([Sigma_d[k]]))))
            Sigma_computed = np.vstack([np.hstack([Sigma_x[k], Sigma_xu, Sigma_xd]),
                                        np.hstack([Sigma_xu.T, Sigma_u_k, Sigma_ud]),
                                        np.hstack([Sigma_xd.T, Sigma_ud.T, Sigma_d[k]])])
        display(Math(r"\Sigma_{} = {}".format(k, np2bmatrix([Sigma_computed]))))
        display(Math("\\text{Check against}\,\, \Sigma_k \,\,\\text{from Opti output}"))
        display(Math(r'\Sigma_{} = {}'.format(k, np2bmatrix([Sigma[k]]))))
        print("Covariance Prop.")
        display(
            Math(r'\Sigma^x_{} = {}'.format(k + 1, np2bmatrix([affine_transform, Sigma_computed, affine_transform.T]))))
        display(
            Math(r'\,\,\,\,\,\,\, = {}'.format(np2bmatrix([affine_transform @ Sigma_computed, affine_transform.T]))))
        display(
            Math(r'\,\,\,\,\,\,\, = {}'.format(np2bmatrix([affine_transform @ Sigma_computed @ affine_transform.T]))))
        display(Math("\\text{Check against}\,\, \Sigma^x_k \,\,\\text{from Opti output}"))
        display(Math(r'\Sigma^x_{} = {}'.format(k + 1, np2bmatrix([Sigma_x[k + 1]]))))
        print("Constraint shrinking on state")

