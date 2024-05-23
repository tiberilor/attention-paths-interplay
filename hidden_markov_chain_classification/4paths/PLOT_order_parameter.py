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
    # WHICH ORDER PARAMETERS
    parser.add_argument("--model_widths", "-N", nargs='+', type=int, default=[10],
                        help="single or multiple ints N1 Na")
    # STORE
    parser.add_argument("--save_figure", action="store_true")
    parser.add_argument('--figure_id', type=str)
    # LOCATIONS
    parser.add_argument('--filenames', nargs='+')
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

    # <editor-fold desc="LOOP THROUGH ALL THE RESULT FILES">
    for file_name in args.filenames:
        # <editor-fold desc="LOAD RESULTS">
        results = torch.load(file_name, map_location=torch.device('cpu'))
        # we convert what we loaded to cpu, so we make sure that even if we saved things on gpu, they are still properly loaded
        dataset_info_train = copy.deepcopy(results["dataset_info"])
        model = results["model"]
        # </editor-fold>

        # check if width is one we want
        width = model.model_widths[0]  # all widths are the same in this experiment, so we take the first
        if width in args.model_widths:
            # retrieve the order parameter
            order_parameter = model.compute_symmetrized_order_parameter_largest().detach().clone().cpu().numpy()

            # <editor-fold desc="SWAP DENOISING AND BAD PATH">
            # swap denoising and bad path, for nicer visualization of order parameter
            # given order_parameter[i,j]
            # Swap rows i=1 and i=2
            order_parameter[[1, 2]] = order_parameter[[2, 1]]
            # Swap columns j=1 and j=2
            order_parameter[:, [1, 2]] = order_parameter[:, [2, 1]]
            # </editor-fold>

            # PLOT PARAMETERS
            # Divergent colormap
            cmap = 'RdBu'
            # font sizes
            fontsize = 8
            title_fontsize = 8  # Define the fontsize for the title
            # title
            title = f"U at N={width}"
            # Custom path labels
            path_labels = ['g', 'd', 'a1', 'a2']
            x_labels_rotation = 90  # in degrees
            # Custom path colors
            label_colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:purple']
            # figure size
            colorbar_shrink = 0.6
            # size constants (inches)
            text_width = 6.9
            text_height = 10
            # height fraction of figure w.r.t. to the text height
            height_fraction = 1/8
            # width fraction of figure w.r.t. to the text width
            width_fraction = 1/8
            figure_size = (text_width * width_fraction, text_height * height_fraction)
            print("Figure size (inches):")
            print(figure_size)

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
            plt.xticks(np.arange(len(path_labels)), path_labels, fontsize=fontsize, rotation=x_labels_rotation)
            plt.yticks(np.arange(len(path_labels)), path_labels, fontsize=fontsize)

            # Modify tick labels colors one by one
            for i, label in enumerate(plt.gca().get_xticklabels()):
                label.set_color(label_colors[i])
            for i, label in enumerate(plt.gca().get_yticklabels()):
                label.set_color(label_colors[i])

            # Ensure the colormap is centered at 0
            plt.clim(-np.max(np.abs(order_parameter)), np.max(np.abs(order_parameter)))

            # Add title
            plt.title(title, fontsize=title_fontsize)

            # call tight layout
            # plt.tight_layout()

            # Save the figure
            if args.save_figure:
                # stored image name
                image_name = subfolder_path + f"N{width}"
                plt.savefig(image_name + ".svg", format='svg', bbox_inches='tight')


    # </editor-fold>

# show plots
plt.show()


















