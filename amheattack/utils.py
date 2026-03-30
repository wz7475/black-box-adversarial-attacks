import os
import logging
import csv
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ResultLogger:
    def __init__(self, output_dir, args, include_seed=False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set up logging
        self.logger = logging.getLogger("ResultLogger")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.output_dir, "run.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(fh)
        # Save args
        with open(os.path.join(self.output_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
        # CSV header
        header = "idx,true_label,pred_label,success,queries,l2_dist,obj_value"
        if include_seed:
            header += ",seed"
        with open(os.path.join(self.output_dir, 'log.csv'), 'w') as f:
            f.write(header + "\n")
        self.include_seed = include_seed

    def add_result(self, idx, true_label, pred_label, success, queries, l2_dist, obj_value, seed=None):
        # Log to CSV
        line = f"{idx},{true_label},{pred_label},{int(success)},{queries},{l2_dist},{obj_value}"
        if self.include_seed and seed is not None:
            line += f",{seed}"
        with open(os.path.join(self.output_dir, 'log.csv'), 'a') as f:
            f.write(line + "\n")
        # Log info
        self.logger.info(
            f"idx={idx}, true_label={true_label}, pred_label={pred_label}, "
            f"success={success}, queries={queries}, l2_dist={l2_dist:.4f}, obj_value={obj_value}, seed={seed}"
        )


def get_mnist_loaders(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)
    train_set = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cifar_loaders(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = datasets.CIFAR10(root='data/cifar', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    train_set = datasets.CIFAR10(root='data/cifar', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_imagenet_loader(batch_size: int = 1, start_idx: int = 0, subset_size: int = 50):
    """Stream a contiguous slice [start_idx, start_idx + subset_size) from the
    ImageNet-1k validation split (ILSVRC/imagenet-1k on HuggingFace).

    Requires HuggingFace access to the gated ILSVRC/imagenet-1k dataset
    (run ``huggingface-cli login`` or set HF_TOKEN beforehand).
    """
    from datasets import load_dataset
    from torch.utils.data import Dataset as TorchDataset

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    hf_ds = (
        load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        .skip(start_idx)
        .take(subset_size)
    )

    class _ImageNetSubset(TorchDataset):
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __getitem__(self, i):
            item = self._items[i]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return preprocess(img), item["label"]

    return DataLoader(_ImageNetSubset(list(hf_ds)), batch_size=batch_size, shuffle=False)


def aggregate_log_csv(output_dir):
    """
    Reads log.csv in output_dir, computes success_ratio, averages and stddevs, and writes aggregation.csv.
    Handles optional 'seed' column.
    """
    log_path = os.path.join(output_dir, "log.csv")
    agg_path = os.path.join(output_dir, "aggregation.csv")
    rows = []
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Robust conversion for 'success'
            success_val = row['success']
            if success_val in ['True', 'true']:
                success_int = 1
            elif success_val in ['False', 'false']:
                success_int = 0
            else:
                success_int = int(success_val)
            r = {
                'success': success_int,
                'queries': int(row['queries']),
                'l2_dist': float(row['l2_dist']),
                'obj_value': float(row['obj_value'])
            }
            if 'seed' in row:
                r['seed'] = int(row['seed'])
            rows.append(r)
    if not rows:
        # No data, write empty aggregation
        with open(agg_path, 'w') as f:
            f.write("success_ratio,avg_queries_success,std_queries_success,avg_l2_dist_success,std_l2_dist_success,avg_obj_value,std_obj_value\n")
            f.write("0,0,0,0,0,0,0\n")
        return

    successes = [r for r in rows if r['success'] == 1]
    all_obj = [r['obj_value'] for r in rows]
    n_total = len(rows)
    n_success = len(successes)
    success_ratio = n_success / n_total if n_total else 0

    if n_success > 0:
        avg_queries_success = float(np.mean([r['queries'] for r in successes]))
        std_queries_success = float(np.std([r['queries'] for r in successes]))
        avg_l2_dist_success = float(np.mean([r['l2_dist'] for r in successes]))
        std_l2_dist_success = float(np.std([r['l2_dist'] for r in successes]))
    else:
        avg_queries_success = std_queries_success = avg_l2_dist_success = std_l2_dist_success = 0.0

    avg_obj_value = float(np.mean(all_obj)) if all_obj else 0.0
    std_obj_value = float(np.std(all_obj)) if all_obj else 0.0

    with open(agg_path, 'w') as f:
        f.write("success_ratio,avg_queries_success,std_queries_success,avg_l2_dist_success,std_l2_dist_success,avg_obj_value,std_obj_value\n")
        f.write(f"{success_ratio},{avg_queries_success},{std_queries_success},{avg_l2_dist_success},{std_l2_dist_success},{avg_obj_value},{std_obj_value}\n")
