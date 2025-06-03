import argparse
import time
import os
import torch
import numpy as np
from amheattack.models import MNISTModel, CIFARModel
from amheattack.attack import AdversarialAttack
from amheattack.utils import get_mnist_loaders, get_cifar_loaders, ResultLogger
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--test_size', type=int, default=50, help="Number of test samples to attack")
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument("--alpha", type=float, default=10.0, help="coefficient for objective function")
    parser.add_argument("--n_max_resampling", type=int, default=100)
    parser.add_argument("--u_cr", type=float, default=0.1)
    parser.add_argument("--u_cf", type=float, default=0.6)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["gen", "cmaes", "jade"], default="jade")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    logger = ResultLogger(args.output_dir, vars(args))

    # Load appropriate test loader
    if args.model == 'mnist':
        loader, _ = get_mnist_loaders(batch_size=1)
    else:
        loader, _ = get_cifar_loaders(batch_size=1)

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

    for idx, (img, label) in enumerate(loader):
        if idx >= args.test_size:
            break
        img, label = img.to(device), label.item()
        print(f"Attacking sample {idx}, true label: {label}")
        attacker = AdversarialAttack(
            model=model,
            device=device,
            num_iters=args.num_iters,
            alpha=args.alpha,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
        start = time.time()
        adv_tensor, success, queries, iterations = attacker.attack(img, label)
        elapsed = time.time() - start
        with torch.no_grad():
            pred = torch.argmax(model(adv_tensor.to(device)), dim=1).item()
        print(f"Result - Success: {success}, Predicted: {pred}, Queries: {queries}, Iterations: {iterations}, Time: {elapsed:.2f}s")
        if success is not None:
            logger.add_result(idx, label, pred, success, queries, iterations)
        # Save adversarial image
        adv_np = adv_tensor.cpu().numpy().transpose(0,2,3,1)[0]  # [H,W,C]
        adv_img = (adv_np * 255).astype(np.uint8)
        from PIL import Image
        mode = 'L' if args.model == 'mnist' else None
        if mode:
            img_pil = Image.fromarray(adv_img.squeeze(), mode)
        else:
            img_pil = Image.fromarray(adv_img)
        img_pil.save(os.path.join(args.output_dir, f"adv_{args.model}_{idx}_{label}.png"))
