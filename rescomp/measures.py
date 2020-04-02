# -*- coding: utf-8 -*-
""" Measures and other analysis functions useful for RC """

import numpy as np
import scipy
import matplotlib.pyplot as plt


def nrmse_over_time(pred_time_series, meas_time_series):
    """ Calculates the NRME between to time series for each time step

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)

    Returns:
        np.ndarray: NRMSE for each time step, shape (T,)

    """
    pred = pred_time_series
    meas = meas_time_series

    nrmse_list = []

    for i in range(0, meas.shape[0]):
        local_nrmse = nrmse(pred[i: i+1], meas[i: i+1])
        nrmse_list.append(local_nrmse)

    return np.array(nrmse_list)


def nrmse(pred_time_series, meas_time_series):
    """ Calculates the NRME between two time series

    Internally just calls rmse with normalized=True

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)

    Returns:
        float: NRMSE
    """
    return rmse(pred_time_series, meas_time_series, normalized=True)


def rmse(pred_time_series, meas_time_series, normalized=False):
    """ Calculates the root mean squared error between two time series

    The time series must be of equal length and dimension

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
        normalized (bool): If False, normalizes the result w.r.t. to the length
            T of the time series.
            If True normalizes the result w.r.t. the length T and the std. of
            the meas_time_series.

    Returns:
        float: RMSE or NRMSE

    """
    pred = pred_time_series
    meas = meas_time_series

    # if normalized: norm = (y_real_cut ** 2).sum()
    # else: norm = y_real_cut.shape[0]
    # error = np.sqrt(((y_pred_cut - y_real_cut) ** 2).sum() / norm)
    #
    # if normalized: norm = np.linalg.norm(y_real_cut) ** 2
    # else: norm = y_real_cut.shape[0]
    # error = np.sqrt(np.linalg.norm(y_pred_cut - y_real_cut) ** 2 / norm)

    if normalized:
        error = np.linalg.norm(pred - meas) \
                / np.linalg.norm(meas)
    else:
        error = np.linalg.norm(pred - meas) \
                / np.sqrt(meas.shape[0])

    return error


def demerge_time(pred_time_series, meas_time_series, epsilon):
    """ Synonym for the divergence_time fct. """

    return divergence_time(pred_time_series, meas_time_series, epsilon)


def divergence_time(pred_time_series, meas_time_series, epsilon):
    """ Calculates how long it takes for measurement and prediction to diverge

    Measure for the quality of the predicted trajectory

    The divergence time refers to the number of time_steps it takes for the
    predicted trajectory to diverge from the measured trajectory by more than a
    given distance in one or more dimensions.
    The distance measure is the supremum norm, NOT the euclidean one.

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
        epsilon (float or np.ndarray): Distance threshold, above which the two
            time series count as diverged. Either float or 1D-array with length d.

    Returns:
        int: divergence_time, the number of time steps for which
            meas_time_series and pred_time_series are separated by less than
            epsilon in each dimension.

    """
    pred = pred_time_series
    meas = meas_time_series

    delta = np.abs(meas - pred)
    
    div_bool = (delta > epsilon).any(axis=1)
    div_time = np.argmax(np.append(div_bool,True))+1

    return div_time


def dimension(time_series, r_min=1.5, r_max=5., nr_steps=2,
              plot=False):
    """ Calculates correlation dimension using
    the algorithm by Grassberger and Procaccia.
     
    First we calculate a sum over all points within a given radius, then
    average over all basis points and vary the radius
    (grassberger, procaccia).

    parameters depend on timesteps and the system itself!

    Args:
        time_series (np.ndarray): time series to calculate dimension of, shape (T, d)
        r_min (float): minimum radius
        r_max (float): maximum radius
        nr_steps (int): number of steps in radius, if r_min and r_max are chosen
            properly, then 2 is enough.
        plot (boolean): flag for plotting loglog plot

    Returns: dimension: slope of the log.log plot assumes:
        N_r(radius) ~ radius**dimension
    """
    # TODO: write method to automatically find good parameters of r_min and
    #       r_max for a given system. This method will probably be slow and 
    #       should not be called everytime dimension is called. 
    
    nr_points = float(time_series.shape[0])
    radii = np.logspace(np.log10(r_min), np.log10(r_max), nr_steps)

    tree = scipy.spatial.cKDTree(time_series)
    N_r = np.array(tree.count_neighbors(tree, radii), dtype=float) / nr_points
    N_r = np.vstack((radii, N_r))
    
    if nr_steps > 2:
        # linear fit based on loglog scale, to get slope/dimension:
        slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)[0:2]
        dimension = slope
    elif nr_steps is 2:
        slope = (N_r[1,1]-N_r[1,0])/(N_r[0,1]-N_r[0,0])
        dimension = slope

    ###plotting
    if plot:
        plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.)
        plt.title('loglog plot of the N_r(radius), slope/dim = ' + str(slope))
        plt.show()
    return dimension

    

# def return_map(self, axis=2):
#     """
#     Shows the recurrence plot of the maxima of a given axis
#     """
#     max_pred = self.y_pred[scipy.signal.argrelextrema(self.y_pred[:,2],
#         np.greater, order = 5),axis]
#     max_test = self.y_test[scipy.signal.argrelextrema(self.y_test[:,2],
#         np.greater, order=5),axis]
#     plt.plot(max_pred[0,:-1:2], max_pred[0,1::2],
#              '.', color='red', alpha=0.5, label='predicted y')
#     plt.plot(max_test[0,:-1:2], max_test[0,1::2],
#              '.', color='green', alpha=0.5, label='test y')
#     plt.legend(loc=2, fontsize=10)
#     plt.show()


# def dimension(reservoir, r_min=0.5, r_max=5., r_steps=0.15,
#               plot=False, test_measure=False):
#     """ Calculates correlation dimension
#
#     for reservoir.y_pred (or reservoir.y_test) using
#     the algorithm by Grassberger and Procaccia and returns dimension.
#     traj: trajectory of an attractor, whos correlation dimension is returned
#     First we calculate a sum over all points within a given radius, then
#     average over all basis points and vary the radius
#     (grassberger, procaccia).
#
#     parameters depend on reservoir.dt and the system itself!
#
#     N_r: list of tuples: (radius, average number of neighbours within all
#         balls)
#
#     Args:
#         reservoir ():
#         r_min ():
#         r_max ():
#         r_steps ():
#         plot ():
#         test_measure ():
#
#     Returns: dimension: slope of the log.log plot assumes:
#         N_r(radius) ~ radius**dimension
#
#     """
#     if test_measure:
#         traj = reservoir.y_test  # for measure assessing
#     else:
#         traj = reservoir.y_pred  # for evaluating prediction
#
#     # TODO: This rescale factor only works for the 3D Lorenz-63 System and has
#     # TODO: to be changed for all other Systems! just plot the log-log plot and
#     # TODO: then change the rest of the code accordingly
#     lorenz_rescale_factor = 8.5
#
#     # adapt parameters to input size:
#     r_min *= traj.std(axis=0).mean() / lorenz_rescale_factor
#     r_max *= traj.std(axis=0).mean() / lorenz_rescale_factor
#     r_steps *= traj.std(axis=0).mean() / lorenz_rescale_factor
#
#     nr_points = float(traj.shape[0])
#     radii = np.arange(r_min, r_max, r_steps)
#
#     tree = scipy.spatial.cKDTree(traj)
#     N_r = np.array(tree.count_neighbors(tree, radii), dtype=float) / nr_points
#     N_r = np.vstack((radii, N_r))
#
#     # linear fit based on loglog scale, to get slope/dimension:
#     slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)[0:2]
#     dimension = slope
#
#     ###plotting
#     if plot:
#         plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.)
#         plt.title('loglog plot of the N_r(radius), slope/dim = ' + str(slope))
#         plt.show()
#     return dimension


# def lyapunov(reservoir, threshold=int(10),
#              plot=False, print_switch=False, test_measure=False):
#     """
#     Calculates the maximal Lyapunov Exponent of reservoir.y_pred (or reservoir.y_test),
#     by estimating the time derivative of the mean logarithmic distances of
#     former next neighbours. Stores it in reservoir.lyapunov (reservoir.lyapunov_test)
#     Only values for tau_min/max are used for calculating the slope!
#
#     Since the attractor has a size of roughly 20 [log(20) ~ 3.] this
#     distance reaches a maximum after a certain time, approximately
#     after 4. time units [time_units = dt*steps]
#     Therefore the default values are choosen to be dt dependent as in
#     ###Definition of taus:
#
#     tau_min/max are given in units of steps
#     plot to check for correct average
#     """
#     """
#     REMINDER:
#     remove the loop over taus, since the slope is calculated with single
#     values only
#     """
#     ###Definition of taus:
#     tau_min = int(0.5 / reservoir.dt)
#     tau_max = int(3.8 / reservoir.dt)
#     taus = np.arange(tau_min, tau_max,
#                      10)  # taus = np.array([tau_min, tau_max])
#
#     if test_measure:
#         traj = reservoir.y_test  # for measure assessing
#     else:
#         traj = reservoir.y_pred  # for evaluating prediction
#
#     tree = scipy.spatial.cKDTree(traj)
#     nn_index = tree.query(traj, k=2)[1]
#
#     # drop all elements in nn_index lists where the neighbour is:
#     # 1. less than threshold time_steps away
#     # 2. where we cannot calculate the neighbours future in tau_max time_steps:
#
#     # contains indices of points and the indices of their nn:
#     reservoir.nn_index = nn_index[
#         np.abs(nn_index[:, 0] - nn_index[:, 1]) > threshold]
#
#     nn_index = nn_index[nn_index[:, 1] + tau_max < traj.shape[0]]
#     nn_index = nn_index[nn_index[:, 0] + tau_max < traj.shape[0]]
#
#     # Calculate the largest Lyapunov exponent:
#     # for storing the results:
#     Sum = []
#     # loop over differnt tau, to get a functional dependence:
#     for tau in taus:
#         # print(tau)
#
#         S = []  # the summed values for all basis points
#
#         # loop over every point in the trajectory, where we can calclutate
#         # the future in tau_max time_steps:
#         for point, nn in nn_index:
#             S.append(np.log(np.linalg.norm(traj[point + tau] - traj[
#                 nn + tau])))  # add one points average s to S
#
#             # since there is no else, we only avg over the points that have
#             # points in their epsilon environment
#         Sum.append((tau * reservoir.dt, np.array(S).mean()))
#     Sum = np.array(Sum)
#
#     slope = (Sum[-1, 1] - Sum[0, 1]) / (Sum[-1, 0] - Sum[0, 0])
#     if plot:
#         plt.title('slope: ' + str(slope))
#         plt.plot(Sum[:, 0], Sum[:, 1])
#         plt.plot(Sum[:, 0], Sum[:, 0] * slope)
#         plt.xlabel('time dt*tau[steps]')
#         plt.ylabel('log_dist_former_neighbours')
#         # plt.plot(Sum[:,0], Sum[:,0]*slope + Sum[0,0])
#         plt.show()
#
#     return slope


# def W_out_distr(self):
#     """
#     Shows a histogram of the fitted parameters of self.w_out, each output
#     dimension in an other color
#     """
#     f = plt.figure(figsize=(10, 10))
#     for i in np.arange(self.y_dim):
#         plt.hist(self.w_out[i], bins=30, alpha=0.5, label='w_out[' + str(i) + ']')
#     plt.legend(fontsize=10)
#     f.show()

# def calc_strength(self):
#     """
#     Calculate the absolute in and out strength of nodes in self.network
#     and its respective average.
#     Stores them in :self.in_strength, self.avg_in_strength, self.out_strength, 
#     self.avg_out_strength
#     """
#     self.in_strength = np.abs(self.network).sum(axis=0)
#     self.avg_in_strength = self.in_strength.mean()
#     self.out_strength = np.abs(self.network).sum(axis=1)
#     self.avg_out_strength = self.out_strength.mean()

# def clustering_coeff(reservoir):
#     """
#     clustering coefficient for each node and returns it.
#     """
#     reservoir.calc_binary_network()
#     network = reservoir.binary_network
#     k = network.sum(axis=0)
#     C = np.diag(network @ network @ network) / k * (k - 1)
#     reservoir.clustering_coeff = C


# def calc_tt(reservoir, flag='bool', split=0.1):
#     """
#     selects depending on if the abs(entry) of reservoir.w_out is one of the
#     largest, depending on split.
#     If split is negative the abs(entry) smallest are selected depending
#     on flag:
#     - 'bool': reservoir.w_out.shape with True/False
#     - 'bool_1d': is a projection to 1d
#     - 'arg': returns args of the selection
#
#     """
#     if reservoir.r_squared:
#         print('no tt_calc for r_squared implemented yet')
#     else:
#         absolute = int(reservoir.ndim * split)
#         n = reservoir.ydim * reservoir.ndim  # dof in w_out
#         top_ten_bool = np.zeros(n, dtype=bool)  # False array
#         arg = np.argsort(
#             np.reshape(np.abs(reservoir.W_out), -1))  # order of abs(w_out)
#         if absolute > 0:
#             top_ten_bool[arg[-absolute:]] = True  # set largest entries True
#             top_ten_arg = np.argsort(np.max(np.abs(reservoir.W_out), axis=0))[
#                           -absolute:]
#         elif absolute < 0:
#             top_ten_bool[arg[:-absolute]] = True  # set largest entries True
#             top_ten_arg = np.argsort(np.max(np.abs(reservoir.W_out), axis=0))[
#                           :-absolute]
#         else:
#             top_ten_arg = np.empty(0)
#
#         top_ten_bool = np.reshape(top_ten_bool,
#                                   reservoir.W_out.shape)  # reshape to original shape
#         top_ten_bool_1d = np.array(top_ten_bool.sum(axis=0),
#                                    dtype=bool)  # project to 1d
#
#         if flag == 'bool':
#             return top_ten_bool
#         elif flag == 'bool_1d':
#             return top_ten_bool_1d
#         elif flag == 'arg':
#             return top_ten_arg


# def weighted_clustering_coeff_onnela(reservoir):
#     """
#     Calculates the weighted clustering coefficient of abs(self.network)
#     according to Onnela paper from 2005.
#     Replacing NaN (originating from division by zero (degree = 0,1)) with 0.
#     Returns weighted_cc.
#     """
#     k = reservoir.binary_network.sum(axis=0)
#     # print(k)
#     network = abs(reservoir.network) / abs(reservoir.network).max()
#
#     network_cbrt = np.cbrt(network)
#     weighted_cc = np.diag(network_cbrt @ network_cbrt @ network_cbrt) / \
#                   (k * (k - 1))
#     # assign 0. to infinit values:
#     weighted_cc[np.isnan(weighted_cc)] = 0.
#     return weighted_cc


#    def calc_covar_rank(reservoir, flag='train'):
#        """
#        Calculated the covarianc rank of the squared network dynamics matrix self.r
#        (or self.r_pred) and stores it in self.covar_rank
#        """
#        """
#        Does not calculate the actual covariance matrix!! Fix befor using
#        """
#        if flag == 'train':
#            res_dyn = self.r
#        elif flag == 'pred':
#            res_dyn = self.r_pred
#        else:
#            raise Exception("wrong covariance flag")
#        covar = np.matmul(res_dyn.T, res_dyn)
#        #self.covar_rank = np.linalg.matrix_rank(covar)
#        print(np.linalg.matrix_rank(covar))

# TODO: Add to ESNWrapper
# def remove_nodes(reservoir, split):
#     """
#     This method removes nodes from the network and w_in according to split,
#     updates avg_degree, spectral_radius,
#     This new reservoir is returned
#     split should be given as a list of two values or a float e [-1. and 1.]
#     example: split = [-.3, 0.3]
#     """
#     if type(split) == list:
#         if len(split) < 3:
#             pass
#         else:
#             raise Exception('too many entries in split. length: ', len(split))
#     elif type(split) == float and split >= -1. and split <= 1.:
#         split = [split]
#     else:
#         raise Exception('values in split not between -1. and 1., type: ',
#                         type(split))
#
#     remaining_size = sum(np.abs(split))
#
#     new = ESN(sys_flag=reservoir.sys_flag,
#               network_dimension=int(
#                           round(reservoir.ndim * (1 - remaining_size))),
#               input_dimension=3, output_dimension=3,
#               type_of_network=reservoir.type, dt=reservoir.dt,
#               training_steps=reservoir.training_steps,
#               prediction_steps=reservoir.prediction_steps,
#               discard_steps=reservoir.discard_steps,
#               regularization_parameter=reservoir.reg_param,
#               spectral_radius=reservoir.spectral_radius,
#               avg_degree=reservoir.avg_degree,
#               epsilon=reservoir.epsilon,
#               # activation_function_flag=reservoir.activation_function_flag,
#               w_in_sparse=reservoir.W_in_sparse,
#               w_in_scale=reservoir.W_in_scale,
#               bias_scale=reservoir.bias_scale,
#               normalize_data=reservoir.normalize_data,
#               r_squared=reservoir.r_squared)
#     # gather to be removed nodes arguments in rm_args:
#     rm_args = np.empty(0)
#     for s in split:
#         rm_args = np.append(calc_tt(reservoir, flag='arg', split=s), rm_args)
#         # print(s, rm_args.shape)
#
#     # rows and columns of network are deleted according to rm_args:
#     new.network = np.delete(np.delete(reservoir.network, rm_args, 0), rm_args,
#                             1)
#     # the new average degree is calculated:
#     new.calc_binary_network()
#     new.avg_degree = new.binary_network.sum(axis=0).mean(axis=0)
#     # the new spectral radius is calculated:
#     new.network = scipy.sparse.csr_matrix(new.network)
#     try:
#         eigenvals = scipy.sparse.linalg.eigs(new.network,
#                                              k=1,
#                                              v0=np.ones(new.n_dim),
#                                              maxiter=1e3*new.n_dim)[0]
#         new.spectral_radius = np.absolute(eigenvals).max()
#
#         # try:
#         #     eigenvals = scipy.sparse.linalg.eigs(new.network, k=1, which='LM')[0]
#         #     new.spectral_radius = np.absolute(eigenvals).max()
#         # except:
#         #     print('eigenvalue calculation failed!, no spectral_radius assigned')
#
#         new.network = new.network.toarray()
#
#     except ArpackNoConvergence:
#         print('Eigenvalue in remove_nodes could not be calculated!')
#         raise
#
#     # Adjust w_in
#     new._w_in = np.delete(reservoir.W_in, rm_args, 0)
#     # pass x,y to new_reservoir
#     new.x_train = reservoir.x_train
#     new.x_discard = reservoir.x_discard
#     new.y_test = reservoir.y_test
#     new.y_train = reservoir.y_train
#
#     return new
