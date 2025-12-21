import argparse
import time
import os
import torch
import numpy as np
from PIL import Image
from amheattack.models import MNISTModel, CIFARModel
from amheattack.stochastic_attack import StochasticGrowthAttack
from amheattack.utils import get_mnist_loaders, get_cifar_loaders, ResultLogger, aggregate_log_csv


def load_model(args, device):
    if args.model == 'mnist':
        model = MNISTModel().to(device)
        model.load_state_dict(torch.load('models/mnist.pth', map_location=device))
    elif args.model == 'cifar10':
        model = CIFARModel().to(device)
        model.load_state_dict(torch.load('models/cifar.pth', map_location=device))
    else:
        raise ValueError(f"Model {args.model} not supported")
    model.eval()
    return model

def get_class_name(label, dataset):
    if dataset == 'mnist':
        return str(label)
    elif dataset == 'cifar10':
        cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        return cifar10_classes[label]
    else:
        return str(label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['mnist', 'cifar10'], default="cifar10")
    parser.add_argument('--test_size', type=int, default=2)
    parser.add_argument("--start_from_test_idx", type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='output_stochastic')
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2, help="Number of times to repeat attack per image")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)

    if args.model == 'mnist':
        _, loader = get_mnist_loaders(batch_size=1)
    else:
        _, loader = get_cifar_loaders(batch_size=1)
    
    starting_idx = args.start_from_test_idx
    ending_idx = starting_idx + args.test_size


    subdir = f"{args.model}_stochastic_alpha_{args.alpha}_test_img_{starting_idx}-{ending_idx}_iters_{args.num_iters}"
    full_output_dir = os.path.join(args.output_dir, subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    logger = ResultLogger(full_output_dir, vars(args))

    for idx, (img, label) in enumerate(loader):
        if idx < starting_idx:
            continue
        if idx >= ending_idx:
            break
        img, label = img.to(device), label.item()
        class_name = get_class_name(label, args.model)
        print(f"StochasticGrowthAttack sample {idx}, true label: {label} ({class_name})")
        img_dir = os.path.join(full_output_dir, f"img_{idx}")
        os.makedirs(img_dir, exist_ok=True)
        all_logs = []
        all_hf_logs = []
        for rep in range(args.repeat):
            rep_seed = args.seed + rep
            attacker = StochasticGrowthAttack(
                model=model,
                device=device,
                num_iters=args.num_iters,
                alpha=args.alpha,
                seed=rep_seed,
            )
            start = time.time()
            adv_tensor, success, queries, obj_value, hf_dataset = attacker.attack(img, label)
            elapsed = time.time() - start
            diff = (adv_tensor.cpu() - img.cpu()).view(adv_tensor.size(0), -1)
            l2_dist = (torch.norm(diff, p=2, dim=1) / (diff.size(1) ** 0.5)).item()
            with torch.no_grad():
                pred = torch.argmax(model(adv_tensor.to(device)), dim=1).item()
            pred_class_name = get_class_name(pred, args.model)
            print(f"Result - Success: {success}, Predicted: {pred} ({pred_class_name}), Queries: {queries}, Time: {elapsed:.2f}s, L2: {l2_dist:.4f}, Obj: {obj_value}")
            if success is not None:
                all_logs.append({
                    "idx": idx,
                    "label": label,
                    "pred": pred,
                    "success": success,
                    "queries": queries,
                    "l2_dist": l2_dist,
                    "obj_value": obj_value,
                    "seed": rep_seed
                })
            if success:
                orig_np = img.cpu().numpy().transpose(0,2,3,1)[0]
                orig_img = (orig_np * 255).astype(np.uint8)
                mode = 'L' if args.model == 'mnist' else None
                if mode:
                    orig_pil = Image.fromarray(orig_img.squeeze(), mode)
                else:
                    orig_pil = Image.fromarray(orig_img)
                orig_pil.save(os.path.join(img_dir, f"orig_{args.model}_{idx}_{class_name}_seed_{rep_seed}.png"))
                adv_np = adv_tensor.cpu().numpy().transpose(0,2,3,1)[0]
                adv_img = (adv_np * 255).astype(np.uint8)
                if mode:
                    img_pil = Image.fromarray(adv_img.squeeze(), mode)
                else:
                    img_pil = Image.fromarray(adv_img)
                img_pil.save(os.path.join(img_dir, f"adv_{args.model}_{idx}_{class_name}_to_{pred_class_name}_seed_{rep_seed}.png"))
            # Collect HF logs for all repeats
            if hf_dataset is not None:
                for entry in hf_dataset:
                    entry["seed"] = rep_seed
                    all_hf_logs.append(entry)
        # Save flattened log file for all repeats
        import pandas as pd
        pd.DataFrame(all_logs).to_csv(os.path.join(img_dir, "log.csv"), index=False)
        # Save flattened HF dataset for all repeats
        from datasets import Dataset
        if all_hf_logs:
            Dataset.from_list(all_hf_logs).save_to_disk(os.path.join(img_dir, "hf_dataset"))
        aggregate_log_csv(img_dir)