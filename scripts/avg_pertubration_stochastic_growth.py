import os
import argparse
import numpy as np
from datasets import Dataset
from PIL import Image
 
from matplotlib import cm


def process_image_dir(hf_dataset_dir, output_path_mean, output_path_std):
    # Load the dataset
    ds = Dataset.load_from_disk(hf_dataset_dir)
    # Compute average perturbation
    perturbations = np.stack(ds['perturbation'])  # shape: (N, 32, 32)
    avg_perturbation = np.mean(perturbations, axis=0)  # shape: (32, 32)
    std_perturbation = np.std(perturbations, axis=0)  # shape: (32, 32)
    # Scale from [0, 1] to [0, 255]
    avg_perturbation_scaled = (avg_perturbation * 255).clip(0, 255).astype(np.uint8) 
    std_perturbation_scaled = (std_perturbation * 255).clip(0, 255).astype(np.uint8)
    # Process mean perturbation
    one_channel_mean = np.expand_dims(avg_perturbation_scaled.mean(axis=0), axis=0)
    one_channel_mean_norm = (one_channel_mean - one_channel_mean.min()) / (np.ptp(one_channel_mean) + 1e-8)
    one_channel_mean_norm = one_channel_mean_norm.squeeze()
    plasma_img_mean = cm.plasma(one_channel_mean_norm)
    plasma_img_mean_uint8 = (plasma_img_mean[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(plasma_img_mean_uint8).save(output_path_mean)
    
    # Process std perturbation
    one_channel_std = np.expand_dims(std_perturbation_scaled.mean(axis=0), axis=0)
    one_channel_std_norm = (one_channel_std - one_channel_std.min()) / (np.ptp(one_channel_std) + 1e-8)
    one_channel_std_norm = one_channel_std_norm.squeeze()
    plasma_img_std = cm.plasma(one_channel_std_norm)
    plasma_img_std_uint8 = (plasma_img_std[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(plasma_img_std_uint8).save(output_path_std)



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
        if subdir.startswith('cifar10_') or subdir.startswith('imagenet_'):
            # Look for image subdirs inside
            for img_dir in os.listdir(subdir_path):
                img_dir_path = os.path.join(subdir_path, img_dir)
                if not os.path.isdir(img_dir_path):
                    continue
                hf_dataset_dir = os.path.join(img_dir_path, 'hf_dataset')
                if os.path.isdir(hf_dataset_dir):
                    output_path_mean = os.path.join(img_dir_path, 'avg_perturbation.png')
                    output_path_std = os.path.join(img_dir_path, 'std_perturbation.png')
                    print(f"Processing {hf_dataset_dir}")
                    print(f"  -> Mean: {output_path_mean}")
                    print(f"  -> Std: {output_path_std}")
                    process_image_dir(hf_dataset_dir, output_path_mean, output_path_std)

if __name__ == "__main__":
    main()
