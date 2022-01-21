import os
from datetime import date
import time
import yaml
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import re
import rescomp

def compare_ks_sims(create_ks_sim_dict_func, create_starting_points_func,
                    ks_sims, parameter_dict, return_qr=False, T_qr=1, return_traj=False, T_traj=10,
                    save=False, prepath="", exp_name=""):

    start = time.time()

    seperator = "--------------------"
    if save:
        if not exp_name == "":
            prepath = os.path.join(prepath, exp_name)
            os.makedirs(prepath)

    # select ks sims:
    possible_ks_sims = list(create_ks_sim_dict_func(0, 0))
    if ks_sims is None:
        ks_sims = possible_ks_sims
    else:
        if type(ks_sims) == str:
            ks_sims = [ks_sims, ]
        for ks_sim in ks_sims:
            if ks_sim not in possible_ks_sims:
                raise Exception(f"ks_sim {ks_sim} is not available")

    # print ks sims to test
    print(seperator)
    print("KS SIMULATIONS TO TEST: ")
    print("ks_sims ", ks_sims)

    # make function out of non-iterable parameters
    for key, val in parameter_dict.items():
        if type(val) in (list, tuple):
            pass
        else:
            parameter_dict[key] = [val, ]

    # print parameters:
    print(seperator)
    print("PARAMETERS: ")
    for key, val in parameter_dict.items():
        if len(val) == 1:
            val = val[0]
        print(key, val)
    print(seperator)

    # parse the dict parameters to python variables:
    system_sizes = parameter_dict["system_size"]
    dimensionss = parameter_dict["dimensions"]
    dts = parameter_dict["dt"]
    taus = parameter_dict["tau"]
    Ts = parameter_dict["T"]
    epss = parameter_dict["eps"]
    N_dimss = parameter_dict["N_dims"]
    N_enss = parameter_dict["N_ens"]

    # calculate number of calculations needed:
    nr_calcs = len(ks_sims)
    for val in parameter_dict.values():
        nr_calcs *= len(val)

    # start the calculations
    print(seperator)
    print(f"STARTING CALCULATIONS ({nr_calcs}): ")
    print(seperator)

    # to save results which are outputted:
    if return_traj:
        traj_res_dict = {}
    if return_qr:
        qr_res_dict = {}
    div_res_dict = {}

    c = 1 # counter for printing
    for i_s, system_size in enumerate(system_sizes):
        # print("......")
        # print(f"system_size: {system_size}")

        for i_d, dimensions in enumerate(dimensionss):
            # print(f"dimensions: {dimensions}")
            ks_sim_dict = create_ks_sim_dict_func(dimensions, system_size)
            ks_sim_dict = {key: val for key, val in ks_sim_dict.items() if key in ks_sims}

            for i_N_ens, N_ens in enumerate(N_enss):
                # print(f"N_ens: {N_ens}")
                starting_points = create_starting_points_func(dimensions, N_ens)

                for i_k, (ks_sim, data_creation_function) in enumerate(ks_sim_dict.items()):
                    # print(f"ks_sim: {ks_sim}")
                    for i_dt, dt in enumerate(dts):
                        # print(f"dt: {dt}")

                        # define the iterator f
                        def f(x):
                            return data_creation_function(time_steps=2, dt=dt, starting_point=x)[-1]

                        dt_str = str(round(dt, 5)).replace(".", "p")
                        name = f"L{i_s}_{system_size}__dim{i_d}_{dimensions}__ks{i_k}_{ks_sim}__" \
                               f"Nens{i_N_ens}_{N_ens}__dt{i_dt}_{dt_str}"

                        if return_traj:
                            traj_name = "traj__" + name + f"__Ttraj_{T_traj}"
                            traj_time_steps = int(T_traj / dt)
                            traj_results = np.zeros((N_ens, traj_time_steps + 1, dimensions))
                            for i in range(N_ens):
                                starting_point = starting_points[i, :]
                                traj = data_creation_function(traj_time_steps, dt, starting_point)
                                traj_results[i, :, :] = traj

                            traj_res_dict[traj_name] = traj_results

                            if save:
                                path = os.path.join(prepath, traj_name)
                                np.save(path, traj_results)

                        for i_tau, tau in enumerate(taus): # TODO create all starting points already here
                            # print(f"tau: {tau}")

                            for i_eps, eps in enumerate(epss):
                                # print(f"eps: {eps}")
                                eps_str = "{:.1e}".format(eps).replace(".", "p")
                                name2 = name + f"__tau{i_tau}_{tau}__eps{i_eps}_{eps_str}"

                                if return_qr:
                                    qr_name = "qr__" + name2 + f"__Tqr_{T_qr}"
                                    qr_time_steps = int(T_qr / dt)
                                    largest_LE_qr = rescomp.measures.iterator_based_lyapunov_spectrum(f,
                                                                                                      starting_points,
                                                                                                      T=T_qr,
                                                                                                      tau=tau, eps=eps,
                                                                                                      nr_of_lyapunovs=1,
                                                                                                      nr_steps=1)
                                    qr_res_dict[qr_name] = largest_LE_qr
                                    if save:
                                        path = os.path.join(prepath, qr_name)
                                        np.save(path, largest_LE_qr)
                                for i_T, T in enumerate(Ts):
                                    # print(f"T: {T}")

                                    for i_N_dims, N_dims in enumerate(N_dimss):
                                        # print(f"N_dims: {N_dims}")
                                        print("....")
                                        print(f"CALC: {c}/{nr_calcs}")

                                        out = rescomp.measures.largest_LE_simple(f, starting_points=starting_points,
                                                                                 T=T,
                                                                                 tau=tau, dt=dt, eps=eps, N_dims=N_dims,
                                                                                 agg=None)

                                        out_name ="div__"+  name2 + f"__T{i_T}_{T}__Ndims{i_N_dims}_{N_dims}"

                                        print(f"{out_name}")

                                        div_res_dict[out_name] = out

                                        if save:
                                            path = os.path.join(prepath, out_name)
                                            np.save(path, out)
                                    c += 1
    # aggregate data in a info_dict and return (or save) it
    info_dict = {}
    if not exp_name == "":
        info_dict["exp_name"] = exp_name
    info_dict["ks_sims"] = ks_sims
    info_dict["parameter_dict"] = parameter_dict
    info_dict["return_qr"] = return_qr
    if return_qr:
        info_dict["T_qr"] = T_qr
    info_dict["return_traj"] = return_traj
    if return_traj:
        info_dict["T_traj"] = T_traj
    info_dict["date"] = date.today().strftime("%d/%m/%Y")
    info_dict["time[s]"] = str(round(time.time() - start, 3))

    if save:
        info_name = "info"
        if not exp_name == "":
            info_name = info_name + "_" + exp_name + ".yml"
        path = os.path.join(prepath, info_name)
        with open(path, 'w') as outfile:
            yaml.dump(info_dict, outfile, default_flow_style=False)

    to_return = [info_dict, div_res_dict]
    if return_traj:
        to_return.append(traj_res_dict)
    if return_qr:
        to_return.append(qr_res_dict)

    return to_return


def subplot_creator(params_dict, data, plot_func, zs=[], xs=[], ys=[]):
    '''
    Args:
        params_dict: a dictionary of the kind {"param1": [param1_val1, param1_val2], "param2": ...}
        data: a list of the kind: [(specific_parameter_dict_1, data_1), (..), ..].
            specific_parameter_dict_1 = {"param1": param1_val2, "param2": param2_val1, ..}
            data_1 = some kind of array or similar that is used by plot_func
        plot_func: takes axis as input and plots onto it using also specific_parameter_dict_1, data_1 as input
    Returns:
        figure (?)
    '''
    fig_size_x, fig_size_y = 9, 9

    if type(xs) == str:
        xs = [xs, ]
    if type(ys) == str:
        ys = [ys, ]
    if type(zs) == str:
        zs = [zs, ]

    variables_sweeped = {key: len(val) for (key, val) in params_dict.items()}
    # nr_sweeped_variables = sum([1 for x in variables_sweeped.values() if x!=1])

    nr_cols = 1
    for x in [variables_sweeped[x] for x in xs]:
        nr_cols *= x

    nr_rows = 1
    for y in [variables_sweeped[y] for y in ys]:
        nr_rows *= y

    nr_lines = 1
    for z in [variables_sweeped[z] for z in zs]:
        nr_lines *= z

    print(f"lines: {nr_lines}, cols: {nr_cols}, rows: {nr_rows}")

    i = 0
    x_index_to_sweep = {}
    x_arg_list = [params_dict[x] for x in xs]
    for items in product(*x_arg_list):
        x_index_to_sweep[i] = items
        i += 1

    i = 0
    y_index_to_sweep = {}
    y_arg_list = [params_dict[y] for y in ys]
    for items in product(*y_arg_list):
        y_index_to_sweep[i] = items
        i += 1

    i = 0
    z_index_to_sweep = {}
    z_arg_list = [params_dict[z] for z in zs]
    for items in product(*z_arg_list):
        z_index_to_sweep[i] = items
        i += 1

    # nr_lines_on_plot = len(params_dict[sweep_on_plot])
    # params_for_line_on_plot = params_dict[sweep_on_plot]

    fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(fig_size_x*nr_cols, fig_size_y*nr_rows))

    for i_x in range(nr_cols):
        for i_y in range(nr_rows):
            if nr_cols == 1:
                if nr_rows == 1:
                    ax = axs
                else:
                    ax = axs[i_x]
            else:
                if nr_rows == 1:
                    ax = axs[i_y]
                else:
                    ax = axs[i_y, i_x]

            # ax = axs[i_y, i_x]

            x_params = x_index_to_sweep[i_x]
            y_params = y_index_to_sweep[i_y]

            x_params_key = dict(list(zip(xs, x_params)))
            y_params_key = dict(list(zip(ys, y_params)))

            xy_params_key = {**x_params_key, **y_params_key}
            # print(xy_all_params_key)

            for i_z in range(nr_lines):
                z_params = z_index_to_sweep[i_z]
                z_params_key = dict(list(zip(zs, z_params)))
                all_params_key = {**xy_params_key, **z_params_key}
                label = ", ".join([f"{key}: {val}" for key, val in z_params_key.items()])
                plot_func(ax, data, all_params_key, label=label)

            title_string = ", ".join([f"{key}: {val}" for key, val in xy_params_key.items()])
            ax.set_title(title_string)

    # print(x_index_to_sweep)
    # print(y_index_to_sweep)
    # print(nr_cols, nr_rows)
    # print(variables_sweeped)
    # print(nr_sweeped_variables)


def plot_div_file(ax, file_name, prepath="", exp_name="", mean_axs="all", dim_index=None,
                  ens_index=None, show_error=True,
                  rel_dist=True, include_fit=True, t_min=0, t_max=5):
    '''
    Args:
        ax:
        file_name:
        prepath:
        exp_name:
        mean_axs: "no", "dim", "ens", "all"
        show_error:
        rel_dist:
        include_fit:
    Returns:
    '''

    path = os.path.join(prepath, exp_name)
    files = os.listdir(path)

    file_list = [x for x in files if x.startswith(file_name)]
    if len(file_list) == 0:
        raise Exception(f"no file found under {path}, that starts with {file_name}")
    elif len(file_list) > 1:
        raise Exception(f"multiple files found under {path}, that starts with {file_name}")
    else:
        file = file_list[0]

    params = file_name_to_params(file_name)
    T = params["T"]
    dt = params["dt"]

    print(params)

    array = np.load(os.path.join(path, file))
    array_shape = array.shape
    print(f"shape of file: {array_shape}")

    if rel_dist:
        array = array/array[0, :, :]

    if mean_axs == "no":
        dist = array
        error = np.zeros(array_shape)
    elif mean_axs == "dim":
        dist = np.mean(array, axis=-2)[:, np.newaxis, :]
        error = np.std(array, axis=-2)[:, np.newaxis, :]
    elif mean_axs == "ens":
        dist = np.mean(array, axis=-1)[:, :, np.newaxis]
        error = np.std(array, axis=-1)[:, :, np.newaxis]
    elif mean_axs == "all":
        dist = np.mean(array, axis=(-1, -2))[:, np.newaxis, np.newaxis]
        error = np.std(array, axis=(-1, -2))[:, np.newaxis, np.newaxis]

    x_steps, n_dims_mod, n_ens_mod = dist.shape

    if not show_error:
        error = np.zeros(dist.shape)

    if dim_index is None:
        dim_index = np.arange(n_dims_mod)
    elif type(dim_index) == int:
        dim_index = [dim_index, ]
    if ens_index is None:
        ens_index = np.arange(n_ens_mod)
    elif type(ens_index) == int:
        ens_index = [ens_index, ]

    x = np.arange(x_steps)*dt
    for i_dim in range(n_dims_mod):
        for i_ens in range(n_ens_mod):
            if i_dim in dim_index and i_ens in ens_index:
                data = dist[:, i_dim, i_ens]
                data_error = error[:, i_dim, i_ens]
                data_error_high, data_error_low = data + data_error, data - data_error

                # transform to logarithmic
                data = np.log(data)
                data_error_high = np.log(data_error_high)
                data_error_low = np.log(data_error_low)

                # linear fit
                label = ""
                if include_fit:
                    x_fit, y_fit, coef = linear_fit(data, dt, t_min=t_min, t_max=t_max)
                    label += f"LLE: {coef[0].round(3)}"
                p = ax.errorbar(x=x, y=data, yerr=(data_error_high - data, data - data_error_low), label=label)
                c = p[0].get_color()
                if include_fit:
                    ax.plot(x_fit, y_fit, linestyle="--", c=c)

    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_title(", ".join([f"{key}: {val}" for key,val in params.items()]))

    ax.set_ylabel(r"$\ln d/d_0$")
    ax.set_xlabel("T")

def file_name_to_params(file_name):
    transform_dict = {"L": "system_size",
                      "dim": "dimensions",
                      "ks": "ks_sims",
                      "Nens": "N_ens",
                      "dt": "dt",
                      "tau": "tau",
                      "eps": "eps",
                      "T": "T",
                      "Ndims": "N_dims",
                      }

    if "." in file_name:
        file_name = file_name.split(".")[0]

    params = file_name.split("__")[1:]

    params_keys = [x.split("_")[0] for x in params]
    params_vals = [x.split("_")[1] for x in params]

    params = dict([(transform_dict[re.search("\D+", key).group()], val) for (key, val) in zip(params_keys, params_vals)])

    for key, val in params.items():
        if re.search("[0-9]p", val) is not None: # floating point values
            params[key] = float(params[key].replace("p", "."))
        elif re.search("\D", val) is None: # other numerical values
            params[key] = int(params[key])

    return params

def linear_fit(y, dt, t_min=0, t_max=5):
    i_min, i_max = int(t_min/dt), int(t_max/dt)
    y = y[i_min: i_max+1]
    x_fit = np.arange(i_min, i_max+1)*dt
    coef = np.polyfit(x_fit,y,1)
    poly1d_fn = np.poly1d(coef)
    y_fit = poly1d_fn(x_fit)
    return x_fit, y_fit, coef
