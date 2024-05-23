import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import LIB_convergent_summation_heads as lib
import torch
import copy
import os
import time

# PLOT PARAMETERS:
number_largest_pcs = 30


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
    folder_path = "./kernel_alignment/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create name of image
    date = time.strftime("%d%m%Y-%H%M%S")
    image_name = folder_path + f"{date}_nPCs{number_largest_pcs}_" + args.figure_id
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

    # <editor-fold desc="RETRIEVE KERNEL EIGENVECTORS">
    gp_kernel = model.return_gp_kernel()
    rn_kernel = model.return_renormalized_kernel()
    _, eigvecs_gp_kernel = torch.linalg.eigh(gp_kernel)
    _, eigvecs_rn_kernel = torch.linalg.eigh(rn_kernel)

    # convert to numpy, order from largest to smallest PC, extract only the largest PCs
    train_labels = train_labels.detach().cpu().numpy()
    eigvecs_gp_kernel = np.flip(eigvecs_gp_kernel.detach().cpu().numpy(), axis=1)[:, 0:number_largest_pcs]
    eigvecs_rn_kernel = np.flip(eigvecs_rn_kernel.detach().cpu().numpy(), axis=1)[:, 0:number_largest_pcs]
    # </editor-fold>

    # <editor-fold desc="COMPUTE EIGENVECTORS OVERLAP WITH LABELS">
    P = np.shape(train_labels)[0]
    overlaps_gp = np.abs(np.einsum("al,a->l", eigvecs_gp_kernel, train_labels) / np.sqrt(P))
    overlaps_rn = np.abs(np.einsum("al,a->l", eigvecs_rn_kernel, train_labels) / np.sqrt(P))
    pcs_rank = np.arange(1, number_largest_pcs+1)
    # </editor-fold>

    # PLOTTING

    # PLOT PARAMETERS

    # SIZE CONSTANTS (inches)
    text_width = 6.9
    text_height = 10
    # SIZE PARAMETERS
    bar_width = 0.35  # max is 0.5: bars touching each other
    fontsize_axis_labels = 8
    fontsize_title = 8
    # height fraction of figure w.r.t. to the text height
    height_fraction = 1/6 * 0.9
    # width fraction of figure w.r.t. to the text width
    width_fraction = 1/2 * 0.9
    figure_size = (text_width*width_fraction, text_height*height_fraction)
    print("Figure size (inches):")
    print(figure_size)
    # LABELS/TITLE
    xlabel = "PC rank"
    ylabel = "Overlap"
    title = " overlap of kernel PCs with task labels"
    # GP
    color_gp = "tab:orange"
    label_gp = "GP"
    # RN
    label_rn = "RN"
    color_rn = "tab:blue"

    # Generate the x-axis positions for the bars
    x_pos = np.arange(len(pcs_rank))

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=figure_size)

    # Plot the bars for overlaps_gp
    bars_gp = ax.bar(x_pos - bar_width / 2, overlaps_gp, bar_width, label=label_gp, color=color_gp)

    # Plot the bars for overlaps_rn
    bars_rn = ax.bar(x_pos + bar_width / 2, overlaps_rn, bar_width, label=label_rn, color=color_rn)

    # Set the x-axis ticks and labels
    ax.set_xticks(x_pos[np.concatenate(([0], np.arange(4, len(x_pos), 5)))])
    ax.set_xticklabels(pcs_rank[np.concatenate(([0], np.arange(4, len(pcs_rank), 5)))], fontsize=fontsize_axis_labels)

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize_axis_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis_labels)
    ax.set_title(title, fontsize=fontsize_title)
    ax.legend(fontsize=fontsize_axis_labels)

    plt.tight_layout()

    # Save the figure
    if args.save_figure:
        plt.savefig(image_name + ".svg", format='svg')

    # Show the plot
    plt.show()



