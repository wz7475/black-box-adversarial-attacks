import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_data(input_dir):
    search_pattern = os.path.join(input_dir, "**", "log.csv")
    csv_files = sorted(glob.glob(search_pattern, recursive=True))

    l2_dists = []
    obj_values = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'success' not in df.columns:
                continue
            
            # Filter files which have at least one True in success column
            success_mask = df['success'].astype(str).str.lower() == 'true'
            if not success_mask.any():
                continue
            
            # Store values
            l2_dist_vals = df['l2_dist'].dropna().values
            obj_vals = df['obj_value'].dropna().values
            
            if len(l2_dist_vals) > 0 and len(obj_vals) > 0:
                l2_dists.append(l2_dist_vals)
                obj_values.append(obj_vals)
        except Exception as e:
            pass
            
    return obj_values, l2_dists

def main():
    parser = argparse.ArgumentParser(description="Create joint boxplots for two datasets.")
    parser.add_argument("--dir_cifar", type=str, default="outputs_grouped/stochastic_growth_alpha_0.1/", help="Path to CIFAR-10 dir")
    parser.add_argument("--dir_imagenet", type=str, default="outputs_grouped/stochastic_growth_imagenet_alpha_0.1", help="Path to ImageNet dir")
    parser.add_argument("--output_png", type=str, default="output_plot_joint.png", help="Path to output PNG file")
    args = parser.parse_args()

    cifar_obj, cifar_l2 = get_data(args.dir_cifar)
    imgnet_obj, imgnet_l2 = get_data(args.dir_imagenet)

    if not cifar_obj and not imgnet_obj:
        print("No valid 'success' data found to plot in either directory.")
        return

    # Draw 2 boxplots in one column
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), dpi=300)

    # Style props match previous
    boxprops = dict(facecolor='#4C72B0', color='black', linewidth=0.8)
    medianprops = dict(color='darkorange', linewidth=1.2)
    whiskerprops = dict(color='black', linewidth=0.8)
    capprops = dict(color='black', linewidth=0.8)
    flierprops = dict(marker='o', markerfacecolor='gray', markeredgecolor='none', markersize=2, alpha=0.7)

    def plot_box(ax, data, ylabel, xlabel=None):
        if data:
            ax.boxplot(data, patch_artist=True, boxprops=boxprops, medianprops=medianprops, 
                        whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)

    # CIFAR-10
    plot_box(axes[0], cifar_obj, "Objective Value\n(CIFAR-10)")
    
    # ImageNet
    plot_box(axes[1], imgnet_obj, "Objective Value\n(ImageNet)", xlabel="Images")

    fig.align_ylabels(axes)
    plt.tight_layout()
    plt.savefig(args.output_png, bbox_inches='tight', dpi=300)
    print(f"Saved joint plot for CIFAR-10 ({len(cifar_obj)} boxes) and ImageNet ({len(imgnet_obj)} boxes) to {args.output_png}")

if __name__ == "__main__":
    main()
