import argparse
import matplotlib.pyplot as plt
import numpy as np
import LIB_convergent_summation_heads as lib
import torch
import copy
import logging


def normal_integral(x_0):
    return 0.5*torch.special.erfc(x_0/torch.sqrt(torch.tensor(2.)))


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', "-f", type=str)
parser.add_argument('--dataset_location', type=str,
                    default="/home/user/datasets",
                    help='where the training datasets are stored')

parser.add_argument("--number_test_examples", default=1000000000000, type=int)
# we set a huge default number, which means all test examples are used

# FLAGS
parser.add_argument("--forced_temperature", default=None, type=float)
parser.add_argument("--plot_training_results", action="store_true",
                    help="Plots the training results (e.g. loss vs epochs, etc...")

parser.add_argument("--test_predictor", action="store_true",
                    help="This computes the predictor statistics")
parser.add_argument("--force_unit_variance_gp", action="store_true",
                    help="forces unit variance when testing the predictor in the GP limit")

parser.add_argument("--test_sampled_predictor", action="store_true",
                    help="This evaluates the sampled predictor statistics")
parser.add_argument("--examples_chunk_size", default=100, type=int)

parser.add_argument("--plot_pre_kernels", action="store_true",
                    help="This skips the training and just plots the pre-kernels")
parser.add_argument("--plot_pre_kernels_style", type=str, default="all",
                    help="all, diagonal. Whether to plot only the diagonal or also the off-diagonal pre-kernels")
parser.add_argument("--plot_pre_kernels_with_temperature", action="store_true",
                    help="plots the pre_kernels with the addition of temperature*identity")

parser.add_argument("--plot_kernel", action="store_true")
parser.add_argument("--plot_kernel_with_temperature", action="store_true",
                    help="plots the kernel with the addition of temperature*identity")

parser.add_argument("--plot_order_parameter", action="store_true")
parser.add_argument("--plot_order_parameter_file_name", default=None,
                    help="If not None, explicitly specifies output path/plot file name")
parser.add_argument("--plot_sampled_order_parameter", action="store_true")
parser.add_argument("--disable_figure_visualization", action="store_true")

parser.add_argument('--output_log_file_name', default='./view.log',
                    help='Specifies output path/log file name')

args = parser.parse_args()

# FORCE FLOAT64
torch.set_default_dtype(torch.float64)

# Logging
log_file_name = args.output_log_file_name
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

# load results
results = torch.load(args.file_name, map_location=torch.device('cpu'))
# we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
dataset_info_train = copy.deepcopy(results["dataset_info"])
model = results["model"]
train_results = results["train_results"]

# <editor-fold desc="PRINT MODEL INFO">
loginf("\n")
loginf("MODEL INFO: START")
loginf("\n")
model.print_architecture()
lib.print_dataset_info(dataset_info_train)
loginf("\n")
loginf("MODEL INFO: END")
loginf("\n")
# </editor-fold>

# <editor-fold desc="PLOT TRAINING RESULTS">
if args.plot_training_results:
    # <editor-fold desc="make time axis into numpy array">
    train_results["loss_history"] = np.array(train_results["loss_history"])
    train_results["loss_energy_history"] = np.array(train_results["loss_energy_history"])
    train_results["loss_entropy_history"] = np.array(train_results["loss_entropy_history"])
    train_results["time_points_history"] = np.array(train_results["time_points_history"])

    for l in range(model.number_attention_layers):
        train_results["max_abs_gradient_history"][l] = np.array(train_results["max_abs_gradient_history"][l])
        train_results["avg_abs_gradient_history"][l] = np.array(train_results["avg_abs_gradient_history"][l])
        train_results["energy_max_abs_gradient_history"][l] = np.array(train_results["energy_max_abs_gradient_history"][l])
        train_results["energy_avg_abs_gradient_history"][l] = np.array(train_results["energy_avg_abs_gradient_history"][l])
        train_results["entropy_max_abs_gradient_history"][l] = (
            np.array(train_results["entropy_max_abs_gradient_history"][l]))
        train_results["entropy_avg_abs_gradient_history"][l] = np.array(train_results["entropy_avg_abs_gradient_history"][l])

    for l in range(model.number_attention_layers + 1):
        # the +1 is because we also want Ua
        train_results["order_parameters_history"][l] = np.array(train_results["order_parameters_history"][l])
    # </editor-fold>

    # <editor-fold desc="define colors and line-styles">
    color_tot = "tab:green"
    color_energy = "tab:red"
    color_entropy = "tab:blue"
    linestyle_max = "-"
    linestyle_avg = ":"
    # </editor-fold>

    # <editor-fold desc="plot loss">
    fig1, ax1 = plt.subplots(1, 1)
    fig1.suptitle("Loss vs epochs")
    ax1.plot(train_results["time_points_history"], train_results["loss_history"], label="total loss", color=color_tot)
    ax1.plot(train_results["time_points_history"], train_results["loss_energy_history"], label="energy", color=color_energy)
    ax1.plot(train_results["time_points_history"], train_results["loss_entropy_history"],
             label="entropy", color=color_entropy)
    ax1.legend()
    # </editor-fold>

    # <editor-fold desc="plot gradients">
    for l in range(model.number_attention_layers):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"Summary: gradient vs epochs, layer {model.number_attention_layers} - {l}")
        label = f"avg abs gradient"
        ax.plot(train_results["time_points_history"], train_results["avg_abs_gradient_history"][l], label=label,
                color=color_tot, linestyle=linestyle_avg)
        label = f"max abs gradient"
        ax.plot(train_results["time_points_history"], train_results["max_abs_gradient_history"][l], label=label,
                color=color_tot, linestyle=linestyle_max)
        label = f"avg abs gradient, energy"
        ax.plot(train_results["time_points_history"], train_results["energy_avg_abs_gradient_history"][l], label=label,
                color=color_energy, linestyle=linestyle_avg)
        label = f"max abs gradient, energy"
        ax.plot(train_results["time_points_history"], train_results["energy_max_abs_gradient_history"][l], label=label,
                color=color_energy, linestyle=linestyle_max)
        label = f"avg abs gradient, entropy"
        ax.plot(train_results["time_points_history"], train_results["entropy_avg_abs_gradient_history"][l], label=label,
                color=color_entropy, linestyle=linestyle_avg)
        label = f"max abs gradient, entropy"
        ax.plot(train_results["time_points_history"], train_results["entropy_max_abs_gradient_history"][l],
                label=label, color=color_entropy, linestyle=linestyle_max)
        ax.legend()
        plt.yscale('log')
    # </editor-fold>

    # <editor-fold desc="plot order parameters">
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f"Scalar order parameter")
    ax.plot(train_results["time_points_history"], train_results["order_parameters_history"][0])
    total_variance = model.variances[-1]
    # Add a vertical for the GP limit
    ax.axhline(y=total_variance, color='k', linestyle='--', label='GP limit')
    # now do the other order parameters
    for l in range(model.number_attention_layers):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"Order parameter, layer {model.number_attention_layers}  - {l}")
        tot_heads = np.shape(train_results["order_parameters_history"][l+1])[1]
        total_variance *= model.variances[-2-l]
        # Add a vertical for the GP limit
        ax.axhline(y=total_variance, color='k', linestyle='--', label='GP limit (diagonals)')
        ax.axhline(y=0, color='tab:red', linestyle='--', label='GP limit (off-diagonals)')
        if tot_heads > 1:  # this if simply ensures that we do not have an error if we have a single head
            for j in range(tot_heads):
                for i in range(j+1):
                    if i == j:
                        if i == 0:
                            label = "diagonal"
                            ax.plot(train_results["time_points_history"],
                                    train_results["order_parameters_history"][l+1][:, i, j], label=label, linestyle='-')
                        else:
                            ax.plot(train_results["time_points_history"],
                                    train_results["order_parameters_history"][l + 1][:, i, j], linestyle='-')
                    if i != j:
                        if i == 0 and j == 1:
                            label = "off-diagonal"
                            ax.plot(train_results["time_points_history"],
                                    train_results["order_parameters_history"][l + 1][:, i, j], label=label,
                                    linestyle=':')
                        else:
                            ax.plot(train_results["time_points_history"],
                                    train_results["order_parameters_history"][l + 1][:, i, j], linestyle=':')
        else:
            label = "diagonal"
            ax.plot(train_results["time_points_history"], train_results["order_parameters_history"][l + 1][:, 0, 0],
                    label=label, linestyle='-')

        ax.legend()
    # </editor-fold>
# </editor-fold>

# TEST PREDICTOR
# <editor-fold desc="TEST PREDICTOR, SAMPLED (optional)">
if args.test_sampled_predictor:
    loginf("\nPREDICTOR STATISTICS, SAMPLED: START\n")

    # retrieve test data
    dataset_info_test = copy.deepcopy(results["dataset_info"])
    dataset_info_test["number_examples"] = args.number_test_examples
    test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test, train=False)
    # sample predictor statistics
    predictor_mean_sampled, predictor_var_sampled = (
        model.evaluate_sampled_predictor_statistics(test_input, examples_chunk_size=args.examples_chunk_size))

    # compute errors and accuracy:
    bias_error = torch.mean(torch.pow(predictor_mean_sampled - test_labels, 2))
    variance_error = torch.mean(predictor_var_sampled)

    # # DEBUG START:
    # loginf("predictor mean:")
    # loginf(predictor_mean[:])
    # loginf("predictor var:")
    # loginf(predictor_var[:])
    # # DEBUG END

    loginf(f"bias error: {bias_error}")
    loginf(f"variance_error: {variance_error}")
    loginf(f"total_error: {bias_error + variance_error}")
    # compute the classification accuracy:
    # transform the predictor mean to just -1 and +1 entries
    thresholded_predictor_mean = predictor_mean_sampled / torch.abs(predictor_mean_sampled)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)
    loginf(f"classification accuracy (mean predictor only): {100*accuracy_mean_predictor} %")
    # compute the classification accuracy:
    x_0 = -1.0*test_labels*predictor_mean_sampled/torch.sqrt(predictor_var_sampled)
    accuracy = torch.mean(normal_integral(x_0))
    loginf(f"classification accuracy: {100*accuracy} %")

    loginf("\nPREDICTOR STATISTICS, SAMPLED: END")
# </editor-fold>

# <editor-fold desc="TEST PREDICTOR (optional)">
if args.test_predictor:
    # retrieve training data
    train_input, train_labels, dataset_info_train = lib.prepare_dataset(
        args.dataset_location, dataset_info_train, train=True)

    # load the model for testing
    model.load(train_input, dataset_info_train)

    # basic params for printing
    P = model.number_training_examples
    Nmax = model.max_model_width
    Temp = model.temperature

    # retrieve test data
    dataset_info_test = copy.deepcopy(results["dataset_info"])
    dataset_info_test["number_examples"] = args.number_test_examples
    test_input, test_labels, dataset_info_test = lib.prepare_dataset(
        args.dataset_location, dataset_info_test, train=False)
    has_extra_testsets = ("extra_testsets" in dataset_info_test)
    if has_extra_testsets:
        test_input, extra_test_input = test_input
        test_labels, extra_test_labels = test_labels

    # # DEBUG START:
    # loginf("test labels:")
    # loginf(test_labels[:])
    # # DEBUG END

    loginf("\nPREDICTOR STATISTICS, THEORY: START\n")

    # compute predictor statistics
    predictor_mean_theory, predictor_var_theory = model.compute_predictor_statistics(
        test_input, train_labels, forced_temperature=args.forced_temperature)

    # # DEBUG START:
    # loginf("predictor mean:")
    # loginf(predictor_mean[:])
    # loginf("predictor var:")
    # loginf(predictor_var[:])
    # # DEBUG END

    bias_error = torch.mean(torch.pow(predictor_mean_theory - test_labels, 2))
    variance_error = torch.mean(predictor_var_theory)
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) Renorm, bias error: {bias_error}")
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) Renorm, variance_error: {variance_error}")
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) Renorm, total_error: {bias_error + variance_error}")
    # compute the classification accuracy:
    # transform the predictor mean to just -1 and +1 entries
    thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) Renorm, classification accuracy (mean predictor only): {100*accuracy_mean_predictor} %")
    # compute the classification accuracy:
    x_0 = -1.0*test_labels*predictor_mean_theory/torch.sqrt(predictor_var_theory)
    accuracy = torch.mean(normal_integral(x_0))
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) Renorm, classification accuracy: {100*accuracy} %")

    # compute predictor statistics in the GP limit
    predictor_mean_gp, predictor_var_gp = (
        model.compute_predictor_statistics(test_input, train_labels, gp_limit=True,
                                           force_unit_variance_gp=args.force_unit_variance_gp,
                                           forced_temperature=args.forced_temperature))
    bias_error = torch.mean(torch.pow(predictor_mean_gp - test_labels, 2))
    variance_error = torch.mean(predictor_var_gp)
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) GP, bias error: {bias_error}")
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) GP, variance_error: {variance_error}")
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) GP, total_error: {bias_error + variance_error}")
    # compute the classification accuracy:
    # transform the predictor mean to just -1 and +1 entries
    thresholded_predictor_mean = predictor_mean_gp / torch.abs(predictor_mean_gp)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) GP, classification accuracy (mean predictor only): {100*accuracy_mean_predictor} %")
    # compute the classification accuracy:
    x_0 = -1.0*test_labels*predictor_mean_gp/torch.sqrt(predictor_var_gp)
    accuracy = torch.mean(normal_integral(x_0))
    loginf(f"(P={P}, Nmax={Nmax}, T={Temp}) GP, classification accuracy: {100*accuracy} %")

    if has_extra_testsets:
        loginf("\nEXTRA DATASETS: START\n")
        for extra_data in dataset_info_test["extra_testsets"]:
            loginf(f"=== {extra_data} ===")
            test_input = extra_test_input[extra_data]
            test_labels = extra_test_labels[extra_data]

            predictor_mean_theory, predictor_var_theory = model.compute_predictor_statistics(
                test_input, train_labels, forced_temperature=args.forced_temperature)

            bias_error = torch.mean(torch.pow(predictor_mean_theory - test_labels, 2))
            variance_error = torch.mean(predictor_var_theory)
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) Renorm, bias error: {bias_error}")
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) Renorm, variance_error: {variance_error}")
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) Renorm, total_error: {bias_error + variance_error}")
            # compute the classification accuracy:
            # transform the predictor mean to just -1 and +1 entries
            thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) Renorm, classification accuracy (mean predictor only): {100*accuracy_mean_predictor} %")
            # compute the classification accuracy:
            x_0 = -1.0*test_labels*predictor_mean_theory/torch.sqrt(predictor_var_theory)
            accuracy = torch.mean(normal_integral(x_0))
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) Renorm, classification accuracy: {100*accuracy} %")

            # compute predictor statistics in the GP limit
            predictor_mean_gp, predictor_var_gp = (
                model.compute_predictor_statistics(test_input, train_labels, gp_limit=True,
                                                force_unit_variance_gp=args.force_unit_variance_gp,
                                                forced_temperature=args.forced_temperature))
            bias_error = torch.mean(torch.pow(predictor_mean_gp - test_labels, 2))
            variance_error = torch.mean(predictor_var_gp)
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) GP, bias error: {bias_error}")
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) GP, variance_error: {variance_error}")
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) GP, total_error: {bias_error + variance_error}")
            # compute the classification accuracy:
            # transform the predictor mean to just -1 and +1 entries
            thresholded_predictor_mean = predictor_mean_gp / torch.abs(predictor_mean_gp)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) GP, classification accuracy (mean predictor only): {100*accuracy_mean_predictor} %")
            # compute the classification accuracy:
            x_0 = -1.0*test_labels*predictor_mean_gp/torch.sqrt(predictor_var_gp)
            accuracy = torch.mean(normal_integral(x_0))
            loginf(f"[{extra_data}] (P={P}, Nmax={Nmax}, T={Temp}) GP, classification accuracy: {100*accuracy} %")

        loginf("\nEXTRA DATASETS: END\n")

    loginf("\nPREDICTOR STATISTICS, THEORY: END")

    if args.test_sampled_predictor:
        average_difference = torch.mean(torch.abs(predictor_mean_theory - predictor_mean_sampled))
        loginf("Average absolute error in predictor (theory vs samples):")
        loginf(average_difference)
# </editor-fold>

# <editor-fold desc="PRINT HEAD STYLE INFO">
if model.heads_style_info is not None:
    loginf("\nHEADS STYLE INFO: START")
    heads_style_info = model.heads_style_info
    for l, layer in enumerate(heads_style_info):
        loginf(f"\nLAYER {l + 1}:")
        for h, head in enumerate(layer):
            loginf(f"head {h}: " + head)

    loginf("\nHEADS STYLE INFO: END\n")
# </editor-fold>

# <editor-fold desc="PLOT KERNELS">
if args.plot_pre_kernels:
    model.plot_pre_kernels(style=args.plot_pre_kernels_style, with_temperature=args.plot_pre_kernels_with_temperature)
if args.plot_kernel:
    model.plot_kernel(with_temperature=args.plot_kernel_with_temperature)


# </editor-fold>

# PLOT ORDER PARAMETER
# <editor-fold desc="PLOT ORDER PARAMETER">
if args.plot_order_parameter:
    # order_parameter = model.compute_symmetrized_order_parameter_largest().detach().clone().cpu().numpy()
    # plt.figure()
    # plt.imshow(order_parameter, cmap='viridis')
    # plt.colorbar()  # Add a colorbar to show the scale
    # plt.title(f'order_parameter')
    model.plot_order_parameter(plot_order_parameter_file_name=args.plot_order_parameter_file_name)
    plt.title("Order parameter, theory")
# </editor-fold>

# <editor-fold desc="PLOT SAMPLED ORDER PARAMETER">
if args.plot_sampled_order_parameter:
    sampled_order_parameter = model.evaluate_sampled_order_parameter()
    model.plot_order_parameter(
        order_parameter=sampled_order_parameter,
        plot_order_parameter_file_name=args.plot_order_parameter_file_name)
    plt.title("Order parameter, sampled")
# </editor-fold>

# SHOW PLOTS
if not args.disable_figure_visualization:
    if (args.plot_training_results or args.plot_pre_kernels or args.plot_kernel or args.plot_order_parameter
            or args.plot_sampled_order_parameter):
        plt.show()
