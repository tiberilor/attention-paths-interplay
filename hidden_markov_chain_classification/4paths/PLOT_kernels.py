import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import LIB_convergent_summation_heads as lib
import torch
import copy
import os
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# PLOT PARAMETERS:
number_largest_pcs = 100

# We do everything under no_grad(), so graphs are not computed by pytorch, which uses memory and computing power
with torch.no_grad():

    # <editor-fold desc="ARGUMENT PARSER">
    parser = argparse.ArgumentParser()
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

    # <editor-fold desc="UTILITIES">
    # FORCE FLOAT64
    torch.set_default_dtype(torch.float64)

    # create folder where to store image
    folder_path = "./kernels/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create name of image
    date = time.strftime("%d%m%Y-%H%M%S")
    if args.save_figure:
        subfolder_path = folder_path + f"{date}_" + args.figure_id + "/"
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    # </editor-fold>

    # <editor-fold desc="LOAD RESULTS">
    results = torch.load(args.filename, map_location=torch.device('cpu'))
    # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
    dataset_info_train = copy.deepcopy(results["dataset_info"])
    model = results["model"]
    # </editor-fold>

    # <editor-fold desc="LOAD MODEL WITH TRAIN DATASET">
    # retrieve training data
    train_input, train_labels, dataset_info_train = lib.prepare_dataset(args.dataset_location, dataset_info_train,
                                                                        train=True)

    # load the model
    model.load(train_input, dataset_info_train)
    # </editor-fold>

    # <editor-fold desc="RETRIEVE KERNELS">
    gp_kernel = model.return_gp_kernel().detach().clone().cpu().numpy()
    rn_kernel = model.return_renormalized_kernel().detach().clone().cpu().numpy()
    pre_kernels = model.return_pre_kernels().detach().clone().cpu().numpy()
    pre_kernel_good_path = pre_kernels[0, 0, :, :]
    pre_kernel_bad_path1 = pre_kernels[1, 1, :, :]
    pre_kernel_denoising_path = pre_kernels[2, 2, :, :]
    pre_kernel_bad_path2 = pre_kernels[3, 3, :, :]
    # </editor-fold>

    # put kernels in a list and loop to plot:
    kernels = [pre_kernel_good_path, pre_kernel_denoising_path, pre_kernel_bad_path1, pre_kernel_bad_path2, gp_kernel, rn_kernel]

    # PLOT PARAMETERS
    # examples ranges
    examples_per_class = 25
    a1 = 0
    b1 = a1 + examples_per_class - 1
    a2 = 75 + 0
    b2 = a2 + examples_per_class - 1
    # titles
    titles = ["good", "denoising", "adversarial #1", "adversarial #2", "GP", "RN"]
    colors = ["tab:green", "tab:blue", "tab:red", "tab:purple", "k", "k"]
    # colormap
    cmap = 'viridis'
    # font sizes
    fontsize = 8
    title_fontsize = 8  # Define the fontsize for the title
    # figure size
    colorbar_shrink = 0.6
    # size constants (inches)
    text_width = 6.9
    text_height = 10
    # height fraction of figure w.r.t. to the text height
    height_fraction = 1 / 6
    # width fraction of figure w.r.t. to the text width
    width_fraction = 1 / 3
    figure_size = (text_width * width_fraction, text_height * height_fraction)
    print("Figure size (inches):")
    print(figure_size)

    for kernel, title, color in zip(kernels, titles, colors):
        # Create the plot
        fig, ax = plt.subplots(figsize=figure_size, dpi=300)  # Set high DPI

        # slice the kernel to a subset of examples
        # this is only to make the figure more readable. The examples can be chosen adequately,
        # so to get good representative examples for how the kernel looks like, and not outliers
        # Calculate the indices for the union of the ranges
        rows = np.concatenate((np.arange(a1, b1 + 1), np.arange(a2, b2 + 1)))
        cols = np.concatenate((np.arange(a1, b1 + 1), np.arange(a2, b2 + 1)))

        # Slice the kernel matrix to select the desired indices
        kernel_subset = kernel[np.ix_(rows, cols)]

        # Create the plot with imshow
        img = ax.imshow(kernel_subset, cmap=cmap)

        # Add title with bold font
        ax.set_title(title, fontsize=title_fontsize, color=color, weight='bold')

        # Set the fontsize for x and y axis labels
        ax.set_xlabel('Examples', fontsize=fontsize)
        ax.set_ylabel('Examples', fontsize=fontsize)

        # Set the fontsize for the axis tick labels
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        # Use make_axes_locatable to create an axes for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add colorbar with adjusted height and fontsize of tick labels
        colorbar = fig.colorbar(img, cax=cax)
        colorbar.ax.tick_params(labelsize=fontsize)  # Adjust fontsize of tick labels

        # Adjust layout to ensure nothing is cut off
        plt.tight_layout()

        # Save the figure
        if args.save_figure:
            # stored image name
            image_name = subfolder_path + title
            plt.savefig(image_name + ".svg", format='svg', bbox_inches='tight', dpi=300)  # Set high DPI

# show plots
plt.show()

