import argparse
import time
import os
import torch
import numpy as np
from PIL import Image
from amheattack.models import MNISTModel, CIFARModel
from amheattack.attack import AdversarialAttack
from amheattack.utils import get_mnist_loaders, get_cifar_loaders, ResultLogger, aggregate_log_csv
from amheattack.optimizers import GeneticAlgOptimizer, CMAESOptimizer, JADEOptimizer


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

def format_optimizer_subdir(optimizer_name, args, optimizer_kwargs, model_name: str):
    parts = [model_name, optimizer_name]
    if hasattr(args, 'eps'):
        parts.append(f"eps_{args.eps}")
    if hasattr(args, 'alpha'):
        parts.append(f"alpha_{args.alpha}")
    for k, v in optimizer_kwargs.items():
        parts.append(f"{k}_{v}")
    return "_".join(parts)

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
    parser.add_argument('--test_size', type=int, default=50, help="Number of test samples to attack")
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--pop_size', type=int, default=500)
    parser.add_argument('--num_iters', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument("--alpha", type=float, default=10.0, help="coefficient for objective function")
    parser.add_argument("--n_max_resampling", type=int, default=100)
    parser.add_argument("--u_cr", type=float, default=0.2)
    parser.add_argument("--u_cf", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["gen", "cmaes", "jade"], default="jade")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)

    if args.model == 'mnist':
        _, loader = get_mnist_loaders(batch_size=1)
    else:
        _, loader = get_cifar_loaders(batch_size=1)

    if args.optimizer == 'cmaes':
        optimizer_cls = CMAESOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'sigma': args.sigma,
            'n_max_resampling': args.n_max_resampling,
        }
    elif args.optimizer == 'gen':
        optimizer_cls = GeneticAlgOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'sigma': args.sigma,
        }
    elif args.optimizer == 'jade':
        optimizer_cls = JADEOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'uCR': args.u_cr,
            'uF': args.u_cf,
            'c': args.c,
        }

    optimizer_subdir = format_optimizer_subdir(args.optimizer, args, optimizer_kwargs, args.model)
    full_output_dir = os.path.join(args.output_dir, optimizer_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    logger = ResultLogger(full_output_dir, vars(args))

    for idx, (img, label) in enumerate(loader):
        if idx >= args.test_size:
            break
        img, label = img.to(device), label.item()
        class_name = get_class_name(label, args.model)
        print(f"Attacking sample {idx}, true label: {label} ({class_name})")
        attacker = AdversarialAttack(
            model=model,
            device=device,
            num_iters=args.num_iters,
            alpha=args.alpha,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
        start = time.time()
        adv_tensor, success, queries, obj_value = attacker.attack(img, label)
        elapsed = time.time() - start
        l2_dist = torch.norm((adv_tensor.cpu() - img.cpu()).view(adv_tensor.size(0), -1), p=2, dim=1).item()
        with torch.no_grad():
            pred = torch.argmax(model(adv_tensor.to(device)), dim=1).item()
        pred_class_name = get_class_name(pred, args.model)
        print(f"Result - Success: {success}, Predicted: {pred} ({pred_class_name}), Queries: {queries}, Time: {elapsed:.2f}s, L2: {l2_dist:.4f}, Obj: {obj_value}")
        if success is not None:
            logger.add_result(idx, label, pred, success, queries, l2_dist, obj_value)
        if success:
            orig_np = img.cpu().numpy().transpose(0,2,3,1)[0]  # [H,W,C]
            orig_img = (orig_np * 255).astype(np.uint8)
            mode = 'L' if args.model == 'mnist' else None
            if mode:
                orig_pil = Image.fromarray(orig_img.squeeze(), mode)
            else:
                orig_pil = Image.fromarray(orig_img)
            orig_pil.save(os.path.join(full_output_dir, f"orig_{args.model}_{idx}_{class_name}.png"))
            adv_np = adv_tensor.cpu().numpy().transpose(0,2,3,1)[0]  # [H,W,C]
            adv_img = (adv_np * 255).astype(np.uint8)
            if mode:
                img_pil = Image.fromarray(adv_img.squeeze(), mode)
            else:
                img_pil = Image.fromarray(adv_img)
            img_pil.save(os.path.join(full_output_dir, f"adv_{args.model}_{idx}_{class_name}_to_{pred_class_name}.png"))
    aggregate_log_csv(full_output_dir)
