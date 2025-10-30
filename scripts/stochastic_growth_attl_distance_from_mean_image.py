import os
import csv
import argparse
import numpy as np
import json
from collections import defaultdict

NUM_CLASSES_DEFAULT = 10

def process_log_csv(log_csv_path):
    preds = []
    l2_dists = []
    l2_by_class = defaultdict(list)
    seed_ids = []

    with open(log_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pred = int(row['pred'])
            l2 = float(row['l2_dist'])
            preds.append(pred)
            l2_dists.append(l2)
            l2_by_class[pred].append(l2)
            # record seed identifier if present, otherwise use row index-based id
            seed_id = row.get('seed') or row.get('seed_id') or str(len(seed_ids))
            seed_ids.append(seed_id)

    num_pred_classes = len(set(preds))
    avg_l2 = np.mean(l2_dists) if l2_dists else float('nan')
    std_l2 = np.std(l2_dists) if l2_dists else float('nan')
    avg_l2_per_class = {k: np.mean(v) for k, v in l2_by_class.items()}
    std_l2_per_class = {k: np.std(v) for k, v in l2_by_class.items()}

    # determine representative class for the image (most common prediction)
    rep_class = max(set(preds), key=preds.count) if preds else None

    # compute per-seed delta as (mean_l2 - per_seed_l2)
    mean_l2 = avg_l2
    # seed_deltas by seed id (mean - per_seed)
    seed_deltas = {sid: (mean_l2 - l2) for sid, l2 in zip(seed_ids, l2_dists)}

    # per-class lists: for each class, list of (mean - per_seed) for seeds that predicted that class
    class_deltas = defaultdict(list)
    for pred, l2 in zip(preds, l2_dists):
        class_deltas[pred].append(mean_l2 - l2)

    # return extended tuple including representative class, per-seed deltas, and per-class deltas
    return num_pred_classes, avg_l2, std_l2, avg_l2_per_class, std_l2_per_class, rep_class, seed_deltas, class_deltas

def main(base_dir, output_csv, output_json, num_classes=NUM_CLASSES_DEFAULT):
    results = []
    json_map = {}

    # Walk base_dir -> subdir -> img_dir, only process img_* dirs under each subdir
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for img_dir in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_dir)
            if not os.path.isdir(img_path):
                continue
            if not img_dir.startswith('img_'):
                continue

            log_csv = os.path.join(img_path, 'log.csv')
            if not os.path.isfile(log_csv):
                continue

            num_pred_classes, avg_l2, std_l2, avg_l2_per_class, std_l2_per_class, rep_class, seed_deltas, class_deltas = process_log_csv(log_csv)

            # For CSV include subdir and img_dir
            row = {
                'subdir': subdir,
                'img_dir': img_dir,
                'num_pred_classes': num_pred_classes,
                'avg_l2': avg_l2,
                'std_l2': std_l2,
            }
            # Add per-class stats
            for cls in sorted(avg_l2_per_class):
                row[f'avg_l2_class_{cls}'] = avg_l2_per_class[cls]
                row[f'std_l2_class_{cls}'] = std_l2_per_class[cls]
            results.append(row)

            # Build JSON entry keyed by img_dir only (no subdir included)
            classes_map = {}
            for c in range(num_classes):
                classes_map[f'class_{c}'] = class_deltas.get(c, [])

            json_map[img_dir] = {
                'rep_class': rep_class,
                'classes': classes_map
            }

    # Collect all possible class columns from results
    all_classes = set()
    for row in results:
        for k in row:
            if k.startswith('avg_l2_class_'):
                all_classes.add(int(k.split('_')[-1]))
    all_classes = sorted(all_classes)

    # Write CSV
    fieldnames = ['subdir', 'img_dir', 'num_pred_classes', 'avg_l2', 'std_l2']
    for cls in all_classes:
        fieldnames.append(f'avg_l2_class_{cls}')
        fieldnames.append(f'std_l2_class_{cls}')

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Write JSON mapping
    with open(output_json, 'w') as jf:
        json.dump(json_map, jf, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='Path to base directory containing img_* dirs', default="output_stochastic")
    parser.add_argument('--output_csv', help='Path to output CSV file', default="output.csv")
    parser.add_argument('--output_json', help='Path to output JSON file with per-image class deltas', default="output.json")
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES_DEFAULT, help='Number of classes to include in JSON (e.g., 10 for CIFAR-10)')
    args = parser.parse_args()
    main(args.base_dir, args.output_csv, args.output_json, args.num_classes)