import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import LIB_convergent_summation_heads as lib
import torch
import copy
import os
import time


# We do everything under no_grad(), so graphs are not computed by pytorch, which uses memory and computing power
with torch.no_grad():

    # <editor-fold desc="ARGUMENT PARSER">
    parser = argparse.ArgumentParser()
    # STORE
    parser.add_argument("--save_figure", action="store_true")
    parser.add_argument('--figure_id', type=str)
    # LOCATIONS
    parser.add_argument('--filenames', nargs='+')
    parser.add_argument('--dataset_location', type=str,
                        default="./",
                        help='where the training datasets are stored')
    # TEST PARAMETERS
    parser.add_argument("--number_test_examples", default=1000, type=int)
    # COMPUTATION PARAMETERS
    parser.add_argument("--examples_chunk_size", default=100, type=int)
    args = parser.parse_args()
    # </editor-fold>

    # <editor-fold desc="UTILITIES">
    # FORCE FLOAT64
    torch.set_default_dtype(torch.float64)

    # create folder where to store image
    folder_path = "./classification_accuracy_mean_predictor/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    date = time.strftime("%d%m%Y-%H%M%S")
    if args.save_figure:
        subfolder_path = folder_path + f"{date}_" + args.figure_id + "/"
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    # </editor-fold>

    # <editor-fold desc="LOOP THROUGH ALL THE RESULT FILES">
    model_widths = []
    mean_predictor_classification_accuracies = []
    mean_predictor_classification_accuracies_mnist = []
    mean_predictor_classification_accuracies_fashion = []
    # below is a different list because some widths may not be sampled
    model_widths_sampled = []
    mean_predictor_classification_accuracies_sampled = []
    mean_predictor_classification_accuracies_mnist_sampled = []
    mean_predictor_classification_accuracies_fashion_sampled = []
    for file_name in args.filenames:
        # <editor-fold desc="LOAD RESULTS">
        results = torch.load(file_name, map_location=torch.device('cpu'))
        # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
        dataset_info_train = copy.deepcopy(results["dataset_info"])
        model = results["model"]
        # </editor-fold>

        # <editor-fold desc="RETRIEVE WIDTH">
        width = model.model_widths[0]  # all widths are the same in this experiment, so we take the first
        model_widths.append(width)
        # </editor-fold>

        # <editor-fold desc="LOAD MODEL WITH TRAIN DATASET">
        # retrieve training data
        train_input, train_labels, dataset_info_train = lib.prepare_dataset(args.dataset_location, dataset_info_train,
                                                                            train=True)

        # load the model for testing
        model.load(train_input, dataset_info_train)
        # </editor-fold>

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (THEORY, OMNIGLOT)">
        # retrieve test data
        dataset_info_test = copy.deepcopy(results["dataset_info"])
        dataset_info_test["number_examples"] = args.number_test_examples
        test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                         train=False)

        # compute predictor statistics
        predictor_mean_theory, _ = model.compute_predictor_statistics(test_input, train_labels)
        # compute the classification accuracy:
        thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
        # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
        # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
        accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels) / 2)
        mean_predictor_classification_accuracies.append(accuracy_mean_predictor.item())
        # </editor-fold>

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (THEORY, MNIST)">
        # retrieve test data
        dataset_info_test = copy.deepcopy(results["dataset_info"])
        dataset_info_test["test_set"] = "mnist"
        dataset_info_test["number_examples"] = args.number_test_examples
        test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                         train=False)

        # compute predictor statistics
        predictor_mean_theory, _ = model.compute_predictor_statistics(test_input, train_labels)
        # compute the classification accuracy:
        thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
        # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
        # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
        accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels) / 2)
        mean_predictor_classification_accuracies_mnist.append(accuracy_mean_predictor.item())
        # </editor-fold>

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (THEORY, FASHION MNIST)">
        # retrieve test data
        dataset_info_test = copy.deepcopy(results["dataset_info"])
        dataset_info_test["test_set"] = "fashion"
        dataset_info_test["number_examples"] = args.number_test_examples
        test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                         train=False)

        # compute predictor statistics
        predictor_mean_theory, _ = model.compute_predictor_statistics(test_input, train_labels)
        # compute the classification accuracy:
        thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
        # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
        # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
        accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels) / 2)
        mean_predictor_classification_accuracies_fashion.append(accuracy_mean_predictor.item())
        # </editor-fold>

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (SAMPLED)">
        if model.posterior_samples is not None:

            print(f"START TESTING SAMPLED PREDICTOR, N={width}")
            start_time = time.time()

            # append width
            model_widths_sampled.append(width)

            # <editor-fold desc="OMNIGLOT">
            # retrieve test data
            dataset_info_test = copy.deepcopy(results["dataset_info"])
            dataset_info_test["number_examples"] = args.number_test_examples
            test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test, train=False)
            # sample predictor statistics
            predictor_mean_sampled, _ = (
                model.evaluate_sampled_predictor_statistics(test_input, examples_chunk_size=args.examples_chunk_size))

            thresholded_predictor_mean_sampled = predictor_mean_sampled / torch.abs(predictor_mean_sampled)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy_mean_predictor_sampled = torch.mean(torch.abs(thresholded_predictor_mean_sampled + test_labels) / 2)
            mean_predictor_classification_accuracies_sampled.append(accuracy_mean_predictor_sampled.item())
            # </editor-fold>

            # <editor-fold desc="MNIST">
            # retrieve test data
            dataset_info_test = copy.deepcopy(results["dataset_info"])
            dataset_info_test["test_set"] = "mnist"
            dataset_info_test["number_examples"] = args.number_test_examples
            test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test, train=False)
            # sample predictor statistics
            predictor_mean_sampled, _ = (
                model.evaluate_sampled_predictor_statistics(test_input, examples_chunk_size=args.examples_chunk_size))

            thresholded_predictor_mean_sampled = predictor_mean_sampled / torch.abs(predictor_mean_sampled)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy_mean_predictor_sampled = torch.mean(torch.abs(thresholded_predictor_mean_sampled + test_labels) / 2)
            mean_predictor_classification_accuracies_mnist_sampled.append(accuracy_mean_predictor_sampled.item())
            # </editor-fold>

            # <editor-fold desc="FASHION">
            # retrieve test data
            dataset_info_test = copy.deepcopy(results["dataset_info"])
            dataset_info_test["test_set"] = "fashion"
            dataset_info_test["number_examples"] = args.number_test_examples
            test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test, train=False)
            # sample predictor statistics
            predictor_mean_sampled, _ = (
                model.evaluate_sampled_predictor_statistics(test_input, examples_chunk_size=args.examples_chunk_size))

            thresholded_predictor_mean_sampled = predictor_mean_sampled / torch.abs(predictor_mean_sampled)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy_mean_predictor_sampled = torch.mean(
                torch.abs(thresholded_predictor_mean_sampled + test_labels) / 2)
            mean_predictor_classification_accuracies_fashion_sampled.append(accuracy_mean_predictor_sampled.item())
            # </editor-fold>

            print(f"END TESTING SAMPLED PREDICTOR, N={width}")
            time_elapsed = time.time() - start_time
            print(f"total running time (mins): {time_elapsed / 60}")
        # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="COMPUTE GP LIMIT (finding optimal temperature)">
    file_name = args.filenames[0]  # any result file at any width is fine for computing the GP limit
    # <editor-fold desc="LOAD RESULTS">
    results = torch.load(file_name, map_location=torch.device('cpu'))
    # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
    dataset_info_train = copy.deepcopy(results["dataset_info"])
    model = results["model"]
    # </editor-fold>

    # <editor-fold desc="LOAD MODEL WITH TRAIN DATASET">
    # retrieve training data
    train_input, train_labels, dataset_info_train = lib.prepare_dataset(args.dataset_location, dataset_info_train,
                                                                        train=True)

    # load the model for testing
    model.load(train_input, dataset_info_train)
    # </editor-fold>

    temperatures_for_loop = [0.25, 0.10, 0.075, 0.050, 0.025]

    optimal_temperature_gp = None
    optimal_accuracy_gp = 0.0

    for temperature in temperatures_for_loop:

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (GP LIMIT, OMNIGLOT)">
        # retrieve test data
        dataset_info_test = copy.deepcopy(results["dataset_info"])
        dataset_info_test["number_examples"] = args.number_test_examples
        test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                         train=False)

        # compute predictor statistics
        predictor_mean_theory_gp, _ = model.compute_predictor_statistics(test_input, train_labels, gp_limit=True,
                                                                         forced_temperature=temperature)

        # compute the classification accuracy:
        thresholded_predictor_mean_gp = predictor_mean_theory_gp / torch.abs(predictor_mean_theory_gp)
        # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
        # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
        mean_predictor_classification_accuracy_gp = torch.mean(
            torch.abs(thresholded_predictor_mean_gp + test_labels) / 2).item()
        # </editor-fold>

        # check if accuracy has improved
        if mean_predictor_classification_accuracy_gp > optimal_accuracy_gp:
            optimal_temperature_gp = temperature
            optimal_accuracy_gp = mean_predictor_classification_accuracy_gp

    # set accuracy to optimal value
    mean_predictor_classification_accuracy_gp = optimal_accuracy_gp
    # print found optimal temperature
    print(f"optimal temperature: {optimal_temperature_gp}")
    print(f"found among temperatures: {temperatures_for_loop}")

    # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (GP LIMIT, MNIST)">
    # retrieve test data
    dataset_info_test = copy.deepcopy(results["dataset_info"])
    dataset_info_test["test_set"] = "mnist"
    dataset_info_test["number_examples"] = args.number_test_examples
    test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                     train=False)

    # compute predictor statistics
    predictor_mean_theory_gp, _ = model.compute_predictor_statistics(test_input, train_labels, gp_limit=True,
                                                                     forced_temperature=optimal_temperature_gp)

    # compute the classification accuracy:
    thresholded_predictor_mean_gp = predictor_mean_theory_gp / torch.abs(predictor_mean_theory_gp)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    mean_predictor_classification_accuracy_mnist_gp = torch.mean(
        torch.abs(thresholded_predictor_mean_gp + test_labels) / 2).item()
    # </editor-fold>

    # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (GP LIMIT, FASHION)">
    # retrieve test data
    dataset_info_test = copy.deepcopy(results["dataset_info"])
    dataset_info_test["test_set"] = "fashion"
    dataset_info_test["number_examples"] = args.number_test_examples
    test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                     train=False)

    # compute predictor statistics
    predictor_mean_theory_gp, _ = model.compute_predictor_statistics(test_input, train_labels, gp_limit=True,
                                                                     forced_temperature=optimal_temperature_gp)

    # compute the classification accuracy:
    thresholded_predictor_mean_gp = predictor_mean_theory_gp / torch.abs(predictor_mean_theory_gp)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    mean_predictor_classification_accuracy_fashion_gp = torch.mean(
        torch.abs(thresholded_predictor_mean_gp + test_labels) / 2).item()
    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc="SORT BY INCREASING WIDTH">
    # Convert lists to NumPy arrays and overwrite the original lists (and sort the array for increasing width)
    model_widths = np.array(model_widths)
    sorted_indices = np.argsort(model_widths)
    model_widths = model_widths[sorted_indices]
    mean_predictor_classification_accuracies = np.array(mean_predictor_classification_accuracies)[sorted_indices]
    mean_predictor_classification_accuracies_mnist = np.array(mean_predictor_classification_accuracies_mnist)[sorted_indices]
    mean_predictor_classification_accuracies_fashion = np.array(mean_predictor_classification_accuracies_fashion)[sorted_indices]

    # same for sampled points
    model_widths_sampled = np.array(model_widths_sampled)
    sorted_indices_sampled = np.argsort(model_widths_sampled)
    model_widths_sampled = model_widths_sampled[sorted_indices_sampled]
    mean_predictor_classification_accuracies_sampled = np.array(
        mean_predictor_classification_accuracies_sampled)[sorted_indices_sampled]
    mean_predictor_classification_accuracies_mnist_sampled = np.array(
        mean_predictor_classification_accuracies_mnist_sampled)[sorted_indices_sampled]
    mean_predictor_classification_accuracies_fashion_sampled = np.array(
        mean_predictor_classification_accuracies_fashion_sampled)[sorted_indices_sampled]
    # </editor-fold>

    # titles/labels parameters
    x_label = 'N'
    y_label = 'A (%)'
    title_general = 'A vs N'
    title_omniglot = title_general + ', in distribution'
    title_mnist = title_general + ', ood (MNIST)'
    title_fashion = title_general + ', ood (fashionMNIST)'
    label_GP = "GP"
    label_rn_theory = "RN, theory"
    label_rn_sampled = "RN, sampled"

    # scale parameters
    x_scale = 'log'

    # size constants (inches)
    text_width = 6.9
    text_height = 10

    # Size parameters
    fontsize_axis_labels = 8
    fontsize_title = 8
    fontsize_legend = 6
    line_thickness = 2
    marker_size = 5
    marker_size_sampled = 10
    # height fraction of figure w.r.t. to the text height
    height_fraction = 1/6
    # width fraction of figure w.r.t. to the text width
    width_fraction = 3/8 * 0.9
    figure_size = (text_width*width_fraction, text_height*height_fraction)  # Adjust figure size here
    print("Figure size (inches):")
    print(figure_size)

    # colors/linestyles/markers parameters
    # renormalized parameters
    color_renormalized = "tab:blue"
    marker_renormalized = "x"
    linestyle_renormalized = '-'
    # renormalized (sampled) parameters
    color_renormalized_sampled = "k"
    marker_renormalized_sampled = "o"
    # gp all paths parameters
    color_gp = "tab:red"
    linestyle_gp = '-'

    # <editor-fold desc="OMNIGLOT">
    # Create the plot
    plt.figure(figsize=figure_size)  # Set figure size

    # plot the Renormalized (theory)
    plt.plot(model_widths, 100*mean_predictor_classification_accuracies, color=color_renormalized,
             marker=marker_renormalized, linestyle=linestyle_renormalized, linewidth=line_thickness, markersize=marker_size, label=label_rn_theory)
    # plot the Renormalized (samples)
    plt.scatter(model_widths_sampled, 100*mean_predictor_classification_accuracies_sampled, marker=marker_renormalized_sampled,
                color=color_renormalized_sampled, s=marker_size_sampled, label=label_rn_sampled, zorder=10)
    # plot GP
    plt.axhline(y=100*mean_predictor_classification_accuracy_gp, color=color_gp, linestyle=linestyle_gp,
                linewidth=line_thickness, label=label_GP)

    # Set x-axis to log scale
    plt.xscale(x_scale)

    # Add labels and title
    plt.xlabel(x_label, fontsize=fontsize_axis_labels)
    plt.ylabel(y_label, fontsize=fontsize_axis_labels)
    plt.title(title_omniglot, fontsize=fontsize_title)

    # Set tick font size
    plt.xticks(fontsize=fontsize_axis_labels)
    plt.yticks(fontsize=fontsize_axis_labels)

    plt.legend(fontsize=fontsize_legend)

    # call tight layout
    plt.tight_layout()
    # plt.grid(True) # this is to plot a grid (not sure if I like it)

    # Save the figure
    if args.save_figure:
        plt.savefig(subfolder_path + "omniglot.svg", format='svg')
    # </editor-fold>

    # <editor-fold desc="MNIST">
    # Create the plot
    plt.figure(figsize=figure_size)  # Set figure size

    # plot the Renormalized (theory)
    plt.plot(model_widths, 100*mean_predictor_classification_accuracies_mnist, color=color_renormalized,
             marker=marker_renormalized, linestyle=linestyle_renormalized, linewidth=line_thickness, markersize=marker_size, label=label_rn_theory)
    # plot the Renormalized (samples)
    plt.scatter(model_widths_sampled, 100*mean_predictor_classification_accuracies_mnist_sampled, marker=marker_renormalized_sampled,
                color=color_renormalized_sampled, s=marker_size_sampled, label=label_rn_sampled, zorder=10)
    # plot GP
    plt.axhline(y=100*mean_predictor_classification_accuracy_mnist_gp, color=color_gp, linestyle=linestyle_gp,
                linewidth=line_thickness, label=label_GP)

    # Set x-axis to log scale
    plt.xscale(x_scale)

    # Add labels and title
    plt.xlabel(x_label, fontsize=fontsize_axis_labels)
    plt.ylabel(y_label, fontsize=fontsize_axis_labels)
    plt.title(title_mnist, fontsize=fontsize_title)

    # Set tick font size
    plt.xticks(fontsize=fontsize_axis_labels)
    plt.yticks(fontsize=fontsize_axis_labels)

    plt.legend(fontsize=fontsize_legend)

    # call tight layout
    plt.tight_layout()
    # plt.grid(True) # this is to plot a grid (not sure if I like it)

    # Save the figure
    if args.save_figure:
        plt.savefig(subfolder_path + "mnist.svg", format='svg')
    # </editor-fold>

    # <editor-fold desc="FASHION">
    # Create the plot
    plt.figure(figsize=figure_size)  # Set figure size

    # plot the Renormalized (theory)
    plt.plot(model_widths, 100*mean_predictor_classification_accuracies_fashion, color=color_renormalized,
             marker=marker_renormalized, linestyle=linestyle_renormalized, linewidth=line_thickness, markersize=marker_size, label=label_rn_theory)
    # plot the Renormalized (samples)
    plt.scatter(model_widths_sampled, 100*mean_predictor_classification_accuracies_fashion_sampled, marker=marker_renormalized_sampled,
                color=color_renormalized_sampled, s=marker_size_sampled, label=label_rn_sampled, zorder=10)
    # plot GP
    plt.axhline(y=100*mean_predictor_classification_accuracy_fashion_gp, color=color_gp, linestyle=linestyle_gp,
                linewidth=line_thickness, label=label_GP)

    # Set x-axis to log scale
    plt.xscale(x_scale)

    # Add labels and title
    plt.xlabel(x_label, fontsize=fontsize_axis_labels)
    plt.ylabel(y_label, fontsize=fontsize_axis_labels)
    plt.title(title_fashion, fontsize=fontsize_title)

    # Set tick font size
    plt.xticks(fontsize=fontsize_axis_labels)
    plt.yticks(fontsize=fontsize_axis_labels)

    plt.legend(fontsize=fontsize_legend)

    # call tight layout
    plt.tight_layout()
    # plt.grid(True) # this is to plot a grid (not sure if I like it)

    # Save the figure
    if args.save_figure:
        plt.savefig(subfolder_path + "fashion.svg", format='svg')
    # </editor-fold>

    # Show the plot
    plt.show()
