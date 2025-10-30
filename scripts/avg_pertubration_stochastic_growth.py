import os
import argparse
import numpy as np
from datasets import Dataset
from PIL import Image
 
from matplotlib import cm


def process_image_dir(hf_dataset_dir, output_path):
    # Load the dataset
    ds = Dataset.load_from_disk(hf_dataset_dir)
    # Compute average perturbation
    perturbations = np.stack(ds['perturbation'])  # shape: (N, 32, 32)
    avg_perturbation = np.mean(perturbations, axis=0)  # shape: (32, 32)
    # Scale from [0, 1] to [0, 255]
    avg_perturbation_scaled = (avg_perturbation * 255).clip(0, 255).astype(np.uint8) 
    one_chanel = np.expand_dims(avg_perturbation_scaled.mean(axis=0), axis=0) # shape: (32, 32)
    # Apply plasma colormap (matplotlib >=3.7 compatible)
    # Normalize to [0, 1] for colormap
    one_channel_norm = (one_chanel - one_chanel.min()) / (np.ptp(one_chanel) + 1e-8)
    # Squeeze to 2D
    one_channel_norm = one_channel_norm.squeeze()
    # Apply plasma colormap
    plasma_img = cm.plasma(one_channel_norm)
    # Convert to uint8 and RGB
    plasma_img_uint8 = (plasma_img[:, :, :3] * 255).astype(np.uint8)
    # Save image
    Image.fromarray(plasma_img_uint8).save(output_path)



def main():
    parser = argparse.ArgumentParser(description="Compute and save avg perturbation images.")
    parser.add_argument('--root', type=str, default="output_stochastic", help="Root output_stochastic dir")
    args = parser.parse_args()
    root = args.root
    for subdir in os.listdir(root):
        subdir_path = os.path.join(root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        # Discard dirs like cifar10_stochastic_alpha_0_test_img_115-120_iters_500
        if subdir.startswith('cifar10_'):
            # Look for image subdirs inside
            for img_dir in os.listdir(subdir_path):
                img_dir_path = os.path.join(subdir_path, img_dir)
                if not os.path.isdir(img_dir_path):
                    continue
                hf_dataset_dir = os.path.join(img_dir_path, 'hf_dataset')
                if os.path.isdir(hf_dataset_dir):
                    output_path = os.path.join(img_dir_path, 'avg_perturbation.png')
                    print(f"Processing {hf_dataset_dir} -> {output_path}")
                    process_image_dir(hf_dataset_dir, output_path)

if __name__ == "__main__":
    main()
