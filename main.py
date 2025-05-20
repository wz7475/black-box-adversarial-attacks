import argparse
import time
import os
import torch
import numpy as np
from models import MNISTModel, CIFARModel
from attack import BasicGeneticAttack
from utils import get_mnist_loaders, get_cifar_loaders, ResultLogger


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
    parser.add_argument('--test_size', type=int, default=10, help="Number of test samples to attack")
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    logger = ResultLogger(args.output_dir, vars(args))

    # Load appropriate test loader
    if args.model == 'mnist':
        loader = get_mnist_loaders(batch_size=1)
    else:
        loader = get_cifar_loaders(batch_size=1)

    for idx, (img, label) in enumerate(loader):
        if idx >= args.test_size:
            break
        img, label = img.to(device), label.item()
        print(f"Attacking sample {idx}, true label: {label}")
        attacker = BasicGeneticAttack(
            model=model,
            pop_size=args.pop_size,
            num_iters=args.num_iters,
            eps=args.eps,
            sigma=args.sigma,
            device=device
        )
        start = time.time()
        adv_tensor, success, queries, iterations = attacker.attack(img, label)
        elapsed = time.time() - start
        with torch.no_grad():
            pred = torch.argmax(model(adv_tensor.to(device)), dim=1).item()
        print(f"Result - Success: {success}, Predicted: {pred}, Queries: {queries}, Iterations: {iterations}, Time: {elapsed:.2f}s")
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
