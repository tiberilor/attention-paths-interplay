import argparse
import matplotlib.pyplot as plt
import numpy as np
import LIB_convergent_summation_heads as lib
import torch
import copy
from einops import rearrange, reduce, repeat, einsum
import os
import time

overwrite_with_fully_trained_drops = True


# Performance drops (manually inserted from results of fully trained model):
performances = np.array([81.8, 83.0, 79.5, 75.4, 76.0, 76.5, 74.3, 74.9])
performance_drops = 83.8 - performances

# FORCE FLOAT64
torch.set_default_dtype(torch.float64)

# We do everything under no_grad(), so graphs are not computed by pytorch, which uses memory and computing power
with torch.no_grad():

    # <editor-fold desc="ARGUMENT PARSER">
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_test_examples', type=int, default=1000)
    # STORE
    parser.add_argument("--save_figure", action="store_true")
    parser.add_argument('--figure_id', type=str)
    # LOCATIONS
    parser.add_argument('--filename', type=str)
    parser.add_argument('--dataset_location', type=str,
                        default="./",
                        help='where the training datasets are stored')
    args = parser.parse_args()
    # </editor-fold>
    number_test_examples = args.number_test_examples

    # <editor-fold desc="UTILITIES">
    # FORCE FLOAT64
    torch.set_default_dtype(torch.float64)

    # create folder where to store image
    folder_path = "./head_scores/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create name of image
    date = time.strftime("%d%m%Y-%H%M%S")
    image_name = folder_path + f"{date}_" + args.figure_id
    # </editor-fold>

    # load results
    results = torch.load(args.filename, map_location=torch.device('cpu'))
    # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
    dataset_info_train = copy.deepcopy(results["dataset_info"])
    model = results["model"]
    train_results = results["train_results"]

    # retrieve heads scores
    scores, heads_list, layers_list = model.compute_heads_score()

    # LOAD THE MODEL FOR TESTING
    train_input, train_labels, dataset_info_train = lib.prepare_dataset(args.dataset_location, dataset_info_train,
                                                                        train=True)
    # load the model for testing
    model.load(train_input, dataset_info_train)

    # EVALUATE BASE PERFORMANCE:

    # retrieve test data
    dataset_info_test = copy.deepcopy(results["dataset_info"])
    dataset_info_test["number_examples"] = number_test_examples

    test_input, test_labels, _ = lib.prepare_dataset(args.dataset_location, dataset_info_test, train=False)

    # compute predictor statistics (renormalized)
    predictor_mean, predictor_var = model.compute_predictor_statistics(test_input, train_labels, gp_limit=False)
    # compute the classification accuracy:
    # transform the predictor mean to just -1 and +1 entries
    thresholded_predictor_mean = predictor_mean / torch.abs(predictor_mean)
    # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
    # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
    base_accuracy = torch.mean(torch.abs(thresholded_predictor_mean + test_labels)/2)

    # LOOP TRHOUGH DIFFERENT HEADS, SHUT THE HEAD DOWN, COMPUTE PERFORMANCE DROP
    if not overwrite_with_fully_trained_drops:
        performance_drops = []
        oparam_unfolded = model.unfold_order_parameter()
        for head, layer in zip(heads_list, layers_list):
            # shut down the part of the order parameter associated with
            oparam_unfolded_silenced = oparam_unfolded.clone()
            slice = torch.select(oparam_unfolded_silenced, dim=layer, index=head)
            slice.zero_()
            slice = torch.select(oparam_unfolded_silenced, dim=(model.number_attention_layers + layer), index=head)
            slice.zero_()
            # refold the order parameter
            oparam_silenced = model.fold_order_parameter(oparam_unfolded_silenced)

            # compute predictor statistics with head shut down
            predictor_mean, predictor_var = model.compute_predictor_statistics(test_input, train_labels,
                                                                               order_param=oparam_silenced)
            # compute the classification accuracy:
            # transform the predictor mean to just -1 and +1 entries
            thresholded_predictor_mean = predictor_mean / torch.abs(predictor_mean)
            # add to test_labels (result will be +2 or -2 for correct classification, 0 otherwise). Take abs, divide by 2
            # and sum: the sum will be the number of correctly classified examples. (so taking the mean gives the accuracy in %)
            accuracy = torch.mean(torch.abs(thresholded_predictor_mean + test_labels) / 2)

            performance_drop = (base_accuracy - accuracy)*100
            performance_drops.append(performance_drop.item())

    # PLOT HEADS SCORES and PERFORMANCE DROP

    # Generate the x axis positions (simply, integers corresponding to the head position in the heads score rating)
    x = np.arange(len(scores))
    # create custom labels identifying each head
    custom_labels = []
    for head, layer in zip(heads_list, layers_list):
        # label = f"h={head+1}, l={layer+1}"  # the +1 is to label heads/layers starting from 1, rather than 0
        label = f"({layer+1}){head+1}"  # the +1 is to label heads/layers starting from 1, rather than 0
        custom_labels.append(label)


    # PLOT PARAMETERS
    x_labels_rotation = 45  # in degrees
    # SIZE CONSTANTS (inches)
    text_width = 6.9
    text_height = 10
    # SIZE PARAMETERS
    bar_width = 0.35  # max is 0.5: bars touching each other
    fontsize_axis_labels = 8
    fontsize_axis_labels_x = 6
    fontsize_title = 8
    fontsize_legend = 8
    # height fraction of figure w.r.t. to the text height
    height_fraction = 1/6 * 0.9
    # width fraction of figure w.r.t. to the text width
    width_fraction = 3/8 * 0.9
    figure_size = (text_width*width_fraction, text_height*height_fraction)
    print("Figure size (inches):")
    print(figure_size)
    # LABELS/TITLE
    x_axis_label = "Head ($\ell$)h"
    label_head_scores = "Score"
    color_head_scores = "tab:blue"
    label_performance_drop = 'Drop (%)'
    color_performance_drop = "tab:red"
    title = "Head scores and performance drop"

    # Plotting
    fig, ax1 = plt.subplots(figsize=figure_size)

    # Plotting scores_list bars on the left y-axis
    ax1.bar(x - bar_width/2, scores, width=bar_width, label=label_head_scores, color=color_head_scores)

    # Customize left y-axis
    ax1.set_ylabel(label_head_scores, color=color_head_scores, fontsize=fontsize_axis_labels)
    ax1.tick_params(axis='y', labelcolor=color_head_scores)

    # Create twin axis for the right y-axis
    ax2 = ax1.twinx()

    # Plotting performance_drop bars on the right y-axis
    ax2.bar(x + bar_width/2, performance_drops, width=bar_width, label=label_performance_drop, color=color_performance_drop)

    # Customize right y-axis
    ax2.set_ylabel(label_performance_drop, color=color_performance_drop, fontsize=fontsize_axis_labels)
    ax2.tick_params(axis='y', labelcolor=color_performance_drop)

    # Set custom x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(custom_labels, fontsize=fontsize_axis_labels, rotation=x_labels_rotation)

    # Adding labels and title
    ax1.set_xlabel(x_axis_label, fontsize=fontsize_axis_labels)
    ax1.set_title(title, fontsize=fontsize_title)

    # Set the fontsize of tick labels
    ax1.tick_params(axis='x', labelsize=fontsize_axis_labels_x)
    ax1.tick_params(axis='y', labelsize=fontsize_axis_labels)
    ax2.tick_params(axis='y', labelsize=fontsize_axis_labels)

    # Plot second grid
    # plt.grid(True)

    plt.tight_layout()

    # Save the figure
    if args.save_figure:
        plt.savefig(image_name + ".svg", format='svg')

    # model.plot_order_parameter()
    plt.show()
