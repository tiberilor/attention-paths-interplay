import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import LIB_convergent_summation_heads as lib
import torch
import copy
import os
import time
from itertools import product
from einops import rearrange, reduce, repeat, einsum


# We do everything under no_grad(), so graphs are not computed by pytorch, which uses memory and computing power
with torch.no_grad():

    # <editor-fold desc="ARGUMENT PARSER">
    parser = argparse.ArgumentParser()
    # STORE
    parser.add_argument("--save_figure", action="store_true")
    parser.add_argument('--figure_id', type=str)
    # LOCATIONS
    parser.add_argument('--filename_gibbs', type=str)
    parser.add_argument('--filename_gradient_descent', type=str)
    parser.add_argument('--dataset_location', type=str,
                        default="./",
                        help='where the training datasets are stored')
    args = parser.parse_args()
    # </editor-fold>

    # <editor-fold desc="UTILITIES">
    # FORCE FLOAT64
    torch.set_default_dtype(torch.float64)

    # create folder where to store image
    folder_path = "./order_parameter/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create name of image
    date = time.strftime("%d%m%Y-%H%M%S")
    if args.save_figure:
        subfolder_path = folder_path + f"{date}_" + args.figure_id + "/"
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    #</editor-fold>

    # PLOT PARAMETERS
    # Divergent colormap
    cmap = 'RdBu'
    # font sizes
    tick_labels_size = 6
    fontsize = 6
    title_fontsize = 8  # Define the fontsize for the title
    # title
    title_theory = "theory"
    title_sampled = "sampled"
    title_gradient_descent = "gradient descent"
    x_labels_rotation = 90  # in degrees
    # figure size
    colorbar_shrink = 1.0
    # size constants (inches)
    text_width = 6.9
    text_height = 10
    # height fraction of figure w.r.t. to the text height
    height_fraction = 1 / 4
    # width fraction of figure w.r.t. to the text width
    width_fraction = 1 / 4
    figure_size = (text_width * width_fraction, text_height * height_fraction)
    print("Figure size (inches):")
    print(figure_size)

    # <editor-fold desc="LOAD RESULTS GIBBS">
    results = torch.load(args.filename_gibbs, map_location=torch.device('cpu'))
    # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
    dataset_info_train = copy.deepcopy(results["dataset_info"])
    model = results["model"]
    # </editor-fold>

    # <editor-fold desc="CREATE CUSTOM PATH LABELS">
    indices = np.empty(model.numbers_heads, dtype=object)

    # Create a list of range objects based on the H values
    ranges = [range(H) for H in model.numbers_heads]

    # Use itertools.product to generate all combinations
    for combination in product(*ranges):
        # combination is a tuple containing values for h1, h2, ..., hL
        # You can access individual values like this:
        # h1, h2, h3, ..., hL = combination
        string = ""
        for i, index in enumerate(combination):
            if i == 0:  # do not put a "-" if this is the first index
                string = string + f"{index + 1}"   # note: the +1 is to start labeling heads from 1 rather than 0
            else:
                string = string + "," + f"{index + 1}"
        indices[combination] = string

    for l in range(len(model.numbers_heads) - 1):
        pre_arrangement = ""
        post_arrangement = ""
        for H in range(len(model.numbers_heads) - l):
            if H == 0:
                pre_arrangement = f"h{H}"
                post_arrangement = f"h{H})"
            elif H == 1:
                pre_arrangement = f"h{H} " + pre_arrangement
                post_arrangement = f"(h{H} " + post_arrangement
            else:
                pre_arrangement = f"h{H} " + pre_arrangement
                post_arrangement = f"h{H} " + post_arrangement
        indices = rearrange(indices, pre_arrangement + " -> " + post_arrangement)
    # </editor-fold>

    # <editor-fold desc="PLOT GIBBS (THEORY)">

    # retrieve width, in case we want it later
    width = model.model_widths[0]  # all widths are the same in this experiment, so we take the first

    order_parameter = model.compute_symmetrized_order_parameter_largest().detach().clone().cpu().numpy()

    # Create the plot
    plt.figure(figsize=figure_size)  # Adjust the figure size as needed

    # Create the plot with imshow
    img = plt.imshow(order_parameter, cmap=cmap)

    # Adjust left margin to prevent cutting off tick labels
    # plt.subplots_adjust(left=0.15)  # Adjust the value as needed

    # Add colorbar with adjusted height and fontsize of tick labels
    colorbar = plt.colorbar(img, shrink=colorbar_shrink)
    colorbar.ax.tick_params(labelsize=fontsize)  # Adjust fontsize of tick labels

    # Set custom labels for x and y axes with tilted x-axis labels
    plt.xticks(np.arange(len(indices)), indices, fontsize=tick_labels_size, rotation=x_labels_rotation)
    plt.yticks(np.arange(len(indices)), indices, fontsize=tick_labels_size)

    # Ensure the colormap is centered at 0
    plt.clim(-np.max(np.abs(order_parameter)), np.max(np.abs(order_parameter)))

    # Add title
    plt.title(title_theory, fontsize=title_fontsize)

    # call tight layout
    # plt.tight_layout()

    # Save the figure
    if args.save_figure:
        # stored image name
        image_name = subfolder_path + f"theory"
        plt.savefig(image_name + ".svg", format='svg', bbox_inches='tight')
    # </editor-fold>

    # <editor-fold desc="PLOT GIBBS (SAMPLED)">

    # retrieve width, in case we want it later
    width = model.model_widths[0]  # all widths are the same in this experiment, so we take the first

    order_parameter = model.evaluate_sampled_order_parameter().detach().clone().cpu().numpy()

    # Create the plot
    plt.figure(figsize=figure_size)  # Adjust the figure size as needed

    # Create the plot with imshow
    img = plt.imshow(order_parameter, cmap=cmap)

    # Adjust left margin to prevent cutting off tick labels
    # plt.subplots_adjust(left=0.15)  # Adjust the value as needed

    # Add colorbar with adjusted height and fontsize of tick labels
    colorbar = plt.colorbar(img, shrink=colorbar_shrink)
    colorbar.ax.tick_params(labelsize=fontsize)  # Adjust fontsize of tick labels

    # Set custom labels for x and y axes with tilted x-axis labels
    plt.xticks(np.arange(len(indices)), indices, fontsize=tick_labels_size, rotation=x_labels_rotation)
    plt.yticks(np.arange(len(indices)), indices, fontsize=tick_labels_size)

    # Ensure the colormap is centered at 0
    plt.clim(-np.max(np.abs(order_parameter)), np.max(np.abs(order_parameter)))

    # Add title
    plt.title(title_sampled, fontsize=title_fontsize)

    # call tight layout
    # plt.tight_layout()

    # Save the figure
    if args.save_figure:
        # stored image name
        image_name = subfolder_path + f"sampled"
        plt.savefig(image_name + ".svg", format='svg', bbox_inches='tight')
    # </editor-fold>

    # <editor-fold desc="PLOT GRADIENT DESCENT">
    data = torch.load(args.filename_gradient_descent,
                      map_location=torch.device('cpu'))

    # RETRIVE USEFUL MODEL INFO:
    value_weights = data["w_v_weigts"]
    number_attention_layers = value_weights.size()[0]
    number_heads = value_weights.size()[1]
    numbers_heads = []
    for l in range(number_attention_layers):
        numbers_heads.append(number_heads)

    V1 = data["w_v_weigts"][0]
    V2 = data["w_v_weigts"][1]
    a = data["readout_weights"][0]

    # do readout times value weight of second (last) layer
    effective_weight = einsum(a, V2, "i, h j i -> h j")
    # do effective weight times value weight of first layer
    effective_weight = einsum(effective_weight, V1, "H i , h j i -> h H j")
    effective_weight = rearrange(effective_weight, "h H i -> (h H) i")
    # compute the order parameter
    order_parameter = einsum(effective_weight, effective_weight, "H1 i, H2 i-> H1 H2").detach().clone().cpu().numpy()

    # Create the plot
    plt.figure(figsize=figure_size)  # Adjust the figure size as needed

    # Create the plot with imshow
    img = plt.imshow(order_parameter, cmap=cmap)

    # Adjust left margin to prevent cutting off tick labels
    # plt.subplots_adjust(left=0.15)  # Adjust the value as needed

    # Add colorbar with adjusted height and fontsize of tick labels
    colorbar = plt.colorbar(img, shrink=colorbar_shrink)
    colorbar.ax.tick_params(labelsize=fontsize)  # Adjust fontsize of tick labels

    # Set custom labels for x and y axes with tilted x-axis labels
    plt.xticks(np.arange(len(indices)), indices, fontsize=tick_labels_size, rotation=x_labels_rotation)
    plt.yticks(np.arange(len(indices)), indices, fontsize=tick_labels_size)

    # Ensure the colormap is centered at 0
    plt.clim(-np.max(np.abs(order_parameter)), np.max(np.abs(order_parameter)))

    # Add title
    plt.title(title_gradient_descent, fontsize=title_fontsize)

    # call tight layout
    # plt.tight_layout()

    # Save the figure
    if args.save_figure:
        # stored image name
        image_name = subfolder_path + f"gradient_descent"
        plt.savefig(image_name + ".svg", format='svg', bbox_inches='tight')

    # </editor-fold>


# show plots
plt.show()


















