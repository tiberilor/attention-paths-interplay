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
    # create name of image
    date = time.strftime("%d%m%Y-%H%M%S")
    image_name = folder_path + f"{date}_Ptest{args.number_test_examples}_" + args.figure_id
    # </editor-fold>

    # <editor-fold desc="LOOP THROUGH ALL THE RESULT FILES">
    model_widths = []
    mean_predictor_classification_accuracies = []
    mean_predictor_classification_accuracies_diagonals_only = []
    mean_predictor_classification_accuracies_evened_diagonals = []
    # below is a different list because some widths may not be sampled
    model_widths_sampled = []
    mean_predictor_classification_accuracies_sampled = []
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

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (THEORY)">
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

        # compute the classification accuracy (oparam diagonals only):
        # retrieve order parameter
        o_param = model.compute_symmetrized_order_parameter_largest().detach().clone()
        # extract the diagonal only
        o_param = torch.diag(torch.diag(o_param))
        # compute predictor statistics
        predictor_mean_theory, _ = model.compute_predictor_statistics(test_input, train_labels, order_param=o_param)
        # compute the classification accuracy:
        thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
        # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
        # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
        accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels) / 2)
        mean_predictor_classification_accuracies_diagonals_only.append(accuracy_mean_predictor.item())

        # compute the classification accuracy (oparam evened diagonals):
        # retrieve order parameter
        o_param = model.compute_symmetrized_order_parameter_largest().detach().clone()
        # extract the diagonal
        diagonal = torch.diag(o_param)
        # construct a diagonal with all entries having the same value as the first entry in the original diagonal
        new_diagonal = torch.eye(o_param.size()[0]) * diagonal[0]
        # rewrite the diagonal in the order parameter
        o_param = o_param - torch.diag(diagonal) + new_diagonal
        # compute predictor statistics
        predictor_mean_theory, _ = model.compute_predictor_statistics(test_input, train_labels, order_param=o_param)
        # compute the classification accuracy:
        thresholded_predictor_mean = predictor_mean_theory / torch.abs(predictor_mean_theory)
        # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
        # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
        accuracy_mean_predictor = torch.mean(torch.abs(thresholded_predictor_mean + test_labels) / 2)
        mean_predictor_classification_accuracies_evened_diagonals.append(accuracy_mean_predictor.item())
        # </editor-fold>

        # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (SAMPLED)">
        if model.posterior_samples is not None:

            print(f"START TESTING SAMPLED PREDICTOR, N={width}")
            start_time = time.time()

            # append width
            model_widths_sampled.append(width)

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

            print(f"END TESTING SAMPLED PREDICTOR, N={width}")
            time_elapsed = time.time() - start_time
            print(f"total running time (mins): {time_elapsed / 60}")
        # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="COMPUTE GP LIMIT">
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

    # <editor-fold desc="TEST CLASSIFICATION ACCURACY MEAN PREDICTOR (GP LIMIT)">
    # retrieve test data
    dataset_info_test = copy.deepcopy(results["dataset_info"])
    dataset_info_test["number_examples"] = args.number_test_examples
    test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test,
                                                     train=False)

    # compute predictor statistics
    predictor_mean_theory_gp, _ = model.compute_predictor_statistics(test_input, train_labels, gp_limit=True)

    # compute the classification accuracy:
    thresholded_predictor_mean_gp = predictor_mean_theory_gp / torch.abs(predictor_mean_theory_gp)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    mean_predictor_classification_accuracy_all_paths = torch.mean(
        torch.abs(thresholded_predictor_mean_gp + test_labels) / 2).item()
    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc="SORT BY INCREASING WIDTH">
    # Convert lists to NumPy arrays and overwrite the original lists (and sort the array for increasing width)
    model_widths = np.array(model_widths)
    sorted_indices = np.argsort(model_widths)
    model_widths = model_widths[sorted_indices]
    mean_predictor_classification_accuracies = np.array(mean_predictor_classification_accuracies)[sorted_indices]
    mean_predictor_classification_accuracies_diagonals_only = np.array(mean_predictor_classification_accuracies_diagonals_only)[sorted_indices]
    mean_predictor_classification_accuracies_evened_diagonals = np.array(mean_predictor_classification_accuracies_evened_diagonals)[sorted_indices]

    # same for sampled points
    model_widths_sampled = np.array(model_widths_sampled)
    sorted_indices_sampled = np.argsort(model_widths_sampled)
    model_widths_sampled = model_widths_sampled[sorted_indices_sampled]
    mean_predictor_classification_accuracies_sampled = np.array(
        mean_predictor_classification_accuracies_sampled)[sorted_indices_sampled]
    # </editor-fold>

    # legend parameters
    legend_location = "center left"
    legend_anchor = (0.01, 0.4)

    # titles/labels parameters
    x_label = 'N'
    y_label = 'A (%)'
    title = 'Classification accuracy, noisier data'

    # scale parameters
    x_scale = 'log'

    # size constants (inches)
    text_width = 6.9
    text_height = 10

    # Size parameters
    fontsize_legend = 5
    fontsize_axis_labels = 8
    fontsize_title = 8
    line_thickness = 1
    marker_size = 5
    marker_size_sampled = 10
    # height fraction of figure w.r.t. to the text height
    height_fraction = 1/6
    # width fraction of figure w.r.t. to the text width
    width_fraction = 1/2
    figure_size = (text_width*width_fraction, text_height*height_fraction)  # Adjust figure size here
    print("Figure size (inches):")
    print(figure_size)

    # colors/linestyles/markers parameters
    # renormalized parameters
    label_rn_theory = "paths g+d, theory"
    color_renormalized = "tab:blue"
    marker_renormalized = "x"
    linestyle_renormalized = 'dotted'
    # renormalized (sampled) parameters
    label_rn_sampled = "paths g+d, sampled"
    color_renormalized_sampled = "k"
    marker_renormalized_sampled = "o"
    # PLOT MANIPULATED OPARAM CASES?
    plot_manipulated_oparam = False
    # renormalized diagonals only parameter
    color_renormalized_diagonal_only = "tab:green"
    marker_renormalized_diagonals_only = "x"
    linestyle_renormalized_diagonal_only = '--'
    # renormalized evened diagonals parameter
    color_renormalized_evened_diagonals = "tab:orange"
    marker_renormalized_evened_diagonals = "x"
    linestyle_renormalized_evened_diagonals = '--'
    # gp all paths parameters
    label_gp = "paths g+d, GP"
    color_gp = "tab:red"
    linestyle_gp = 'dotted'

    # Create the plot
    plt.figure(figsize=figure_size)  # Set figure size

    # plot the Renormalized (theory)
    plt.plot(model_widths, 100*mean_predictor_classification_accuracies, color=color_renormalized,
             marker=marker_renormalized, linestyle=linestyle_renormalized, linewidth=line_thickness, markersize=marker_size, label=label_rn_theory)
    if plot_manipulated_oparam:
        # plot the Renormalized diagonals only (theory)
        plt.plot(model_widths, 100*mean_predictor_classification_accuracies_diagonals_only, color=color_renormalized_diagonal_only,
                 linestyle=linestyle_renormalized_diagonal_only, linewidth=line_thickness, marker=marker_renormalized_diagonals_only, markersize=marker_size)
        # plot the Renormalized evened diagonals (theory)
        plt.plot(model_widths, 100*mean_predictor_classification_accuracies_evened_diagonals, color=color_renormalized_evened_diagonals,
                 linestyle=linestyle_renormalized_evened_diagonals, linewidth=line_thickness, marker=marker_renormalized_evened_diagonals, markersize=marker_size)
    # plot the Renormalized (samples)
    plt.scatter(model_widths_sampled, 100*mean_predictor_classification_accuracies_sampled, marker=marker_renormalized_sampled,
                color=color_renormalized_sampled, s=marker_size_sampled, label=label_rn_sampled, zorder=10)
    # plot GP
    plt.axhline(y=100*mean_predictor_classification_accuracy_all_paths, color=color_gp, linestyle=linestyle_gp,
                linewidth=line_thickness, label=label_gp)

    # Set x-axis to log scale
    plt.xscale(x_scale)

    # Add labels and title
    plt.xlabel(x_label, fontsize=fontsize_axis_labels)
    plt.ylabel(y_label, fontsize=fontsize_axis_labels)
    plt.title(title, fontsize=fontsize_title)

    # Set tick font size
    plt.xticks(fontsize=fontsize_axis_labels)
    plt.yticks(fontsize=fontsize_axis_labels)

    # Adding legend
    plt.legend(loc=legend_location, fontsize=fontsize_legend, bbox_to_anchor=legend_anchor)

    # call tight layout
    plt.tight_layout()
    # plt.grid(True) # this is to plot a grid (not sure if I like it)

    # Save the figure
    if args.save_figure:
        plt.savefig(image_name + ".svg", format='svg')

    # Show the plot
    plt.show()
