import os
import csv
import argparse
import numpy as np
from collections import defaultdict

def process_log_csv(log_csv_path):
    preds = []
    l2_dists = []
    l2_by_class = defaultdict(list)

    with open(log_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pred = int(row['pred'])
            l2 = float(row['l2_dist'])
            preds.append(pred)
            l2_dists.append(l2)
            l2_by_class[pred].append(l2)

    num_pred_classes = len(set(preds))
    avg_l2 = np.mean(l2_dists) if l2_dists else float('nan')
    std_l2 = np.std(l2_dists) if l2_dists else float('nan')
    avg_l2_per_class = {k: np.mean(v) for k, v in l2_by_class.items()}
    std_l2_per_class = {k: np.std(v) for k, v in l2_by_class.items()}

    return num_pred_classes, avg_l2, std_l2, avg_l2_per_class, std_l2_per_class

def main(base_dir, output_csv):
    results = []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for img_dir in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_dir)
            log_csv = os.path.join(img_path, 'log.csv')
            if os.path.isdir(img_path) and os.path.isfile(log_csv):
                num_pred_classes, avg_l2, std_l2, avg_l2_per_class, std_l2_per_class = process_log_csv(log_csv)
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

    # Collect all possible class columns
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='Path to base directory (e.g., output_stochastic)', default="output_stochastic")
    parser.add_argument('--output_csv', help='Path to output CSV file', default="output.csv")
    args = parser.parse_args()
    main(args.base_dir, args.output_csv)