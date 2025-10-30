import os
import shutil
import re

def collect_perturbation_images(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_idx = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'avg_perturbation.png':
                src_path = os.path.join(dirpath, filename)
                # Try to extract an image index from the parent directory name, fallback to counter
                parent = os.path.basename(dirpath)
                match = re.search(r'(\d+)$', parent)
                if match:
                    idx = match.group(1)
                else:
                    idx = str(img_idx)
                dst_filename = f'perturbation_{idx}.png'
                dst_path = os.path.join(output_dir, dst_filename)
                # If file exists, increment idx until unique
                while os.path.exists(dst_path):
                    img_idx += 1
                    dst_filename = f'perturbation_{img_idx}.png'
                    dst_path = os.path.join(output_dir, dst_filename)
                shutil.copy2(src_path, dst_path)
                print(f"Moved {src_path} -> {dst_path}")
                img_idx += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect all avg_perturbation.png files into one directory.")
    parser.add_argument('--root', type=str, default="output_stochastic", help="Root directory to search.")
    parser.add_argument('--output', type=str, default="perturbation_images", help="Output directory.")
    args = parser.parse_args()
    collect_perturbation_images(args.root, args.output)
