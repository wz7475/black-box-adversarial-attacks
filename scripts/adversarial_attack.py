import argparse
import time
import os
import torch
import numpy as np
from PIL import Image
from amheattack.models import MNISTModel, CIFARModel
from amheattack.attack import AdversarialAttack
from amheattack.utils import get_mnist_loaders, get_cifar_loaders, ResultLogger, aggregate_log_csv
from amheattack.optimizers import (GeneticAlgOptimizer, JADEOptimizer, DEOptimizer, 
                                    SADEOptimizer, GWOOptimizer,
                                    SHADEOptimizer, LSHADEOptimizer, INFOOptimizer)


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
    
    # ============ Attack Context ============
    parser.add_argument('--model', type=str, choices=['mnist', 'cifar10'], default="cifar10")
    parser.add_argument('--test_size', type=int, default=50, help="Number of test samples to attack")
    parser.add_argument('--output_dir', type=str, default='output')

    # ============ Attack Parameters ============
    parser.add_argument('--eps', type=float, default=0.1, help="Perturbation boundaries - passed to every optimizer")
    parser.add_argument('--num_iters', type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--alpha", type=float, default=10.0, help="Coefficient for objective function (noise regularization)")

    # ============ Shared Optimizer Parameters ============
    parser.add_argument('--pop_size', type=int, default=500, help="Population size for all optimizers")

    # ============ GA Parameters (Genetic Algorithm) ============
    parser.add_argument("--pm", type=float, default=0.1, help="GA: mutation probability [0.01-0.2]")
    parser.add_argument("--mutation", type=str, default="flip", choices=["flip", "swap"], help="GA: mutation strategy")
    parser.add_argument("--mutation_multipoints", type=bool, default=True, help="GA: multipoint mutation")
    parser.add_argument('--sigma', type=float, default=0.1, help="GA: sigma parameter (not used in mealpy GA)")

    # ============ JADE Parameters (Adaptive Differential Evolution) ============
    parser.add_argument("--miu_f", type=float, default=0.5, help="JADE: initial adaptive F [0.4-0.6]")
    parser.add_argument("--miu_cr", type=float, default=0.5, help="JADE: initial adaptive CR [0.4-0.6]")
    parser.add_argument("--pt", type=float, default=0.1, help="JADE: percent of top best agents (p) [0.05-0.2]")
    parser.add_argument("--ap", type=float, default=0.1, help="JADE: adaptation parameter (c) [0.05-0.2]")

    # ============ DE Parameters (Classic Differential Evolution) ============
    parser.add_argument("--wf", type=float, default=0.8, help="DE: weighting factor F [0.0-2.0]")
    parser.add_argument("--cr", type=float, default=0.9, help="DE: crossover rate CR [0.0-1.0]")
    parser.add_argument("--strategy", type=int, default=0, help="DE: strategy (0=rand/1, 1=best/1, 2=rand/2, 3=best/2, 4=current-to-best/1)")

    # ============ SADE Parameters (Self-Adaptive DE) ============
    # SADE auto-adapts F and CR, no additional parameters needed

    # ============ GWO Parameters (Grey Wolf Optimizer) ============
    # GWO uses default parameters, no additional tuning needed

    # ============ SHADE Parameters (Success-History Adaptation DE) ============
    parser.add_argument("--shade_miu_f", type=float, default=0.5, help="SHADE: initial weighting factor F [0.4-0.6]")
    parser.add_argument("--shade_miu_cr", type=float, default=0.5, help="SHADE: initial cross-over probability CR [0.4-0.6]")

    # ============ L-SHADE Parameters (Linear Population Size Reduction SHADE) ============
    parser.add_argument("--lshade_miu_f", type=float, default=0.5, help="L-SHADE: initial weighting factor F [0.4-0.6]")
    parser.add_argument("--lshade_miu_cr", type=float, default=0.5, help="L-SHADE: initial cross-over probability CR [0.4-0.6]")

    # ============ INFO Parameters (weIghted meaN oF vectOrs) ============
    # INFO uses default parameters, no additional tuning needed

    # ============ Optimizer Choice ============
    parser.add_argument("--optimizer", choices=["gen", "jade", "de", "sade", "sapde", "gwo", "shade", "lshade", "info"], default="jade",
                        help="Optimizer: gen=GA, jade=JADE, de=OriginalDE, sade=SADE, sapde=SAP_DE, gwo=GWO, shade=SHADE, lshade=L-SHADE, info=INFO")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)

    if args.model == 'mnist':
        _, loader = get_mnist_loaders(batch_size=1)
    else:
        _, loader = get_cifar_loaders(batch_size=1)

    if args.optimizer == 'gen':
        optimizer_cls = GeneticAlgOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'sigma': args.sigma,
            'pm': args.pm,
            'mutation': args.mutation,
            'mutation_multipoints': args.mutation_multipoints,
        }
    elif args.optimizer == 'jade':
        optimizer_cls = JADEOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'miu_f': args.miu_f,
            'miu_cr': args.miu_cr,
            'pt': args.pt,
            'ap': args.ap,
        }
    elif args.optimizer == 'de':
        optimizer_cls = DEOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'wf': args.wf,
            'cr': args.cr,
            'strategy': args.strategy,
        }
    elif args.optimizer == 'sade':
        optimizer_cls = SADEOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
        }
    elif args.optimizer == 'gwo':
        optimizer_cls = GWOOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
        }
    elif args.optimizer == 'shade':
        optimizer_cls = SHADEOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'miu_f': args.shade_miu_f,
            'miu_cr': args.shade_miu_cr,
        }
    elif args.optimizer == 'lshade':
        optimizer_cls = LSHADEOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'miu_f': args.lshade_miu_f,
            'miu_cr': args.lshade_miu_cr,
        }
    elif args.optimizer == 'info':
        optimizer_cls = INFOOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
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
        # l2_dist = torch.norm((adv_tensor.cpu() - img.cpu()).view(adv_tensor.size(0), -1), p=2, dim=1).item()
        diff = (adv_tensor.cpu() - img.cpu()).view(adv_tensor.size(0), -1)
        l2_dist = (torch.norm(diff, p=2, dim=1) / (diff.size(1) ** 0.5)).item()
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
    """
    temp_output
    └── cifar10_gen_eps_0.1_alpha_10.0_pop_size_500_eps_0.1_sigma_0.1_pm_0.1_mutation_flip_mutation_multipoints_True
        ├── adv_cifar10_0_cat_to_dog.png
        ├── adv_cifar10_2_ship_to_automobile.png
        ├── args.txt
        ├── log.csv     <-------------------- success, perturbation and pred classes for every attack (n+1 rows: header + n-attacks rows)
        ├── aggregation.csv  <--------------- aggregation of logs.csv (2 rows: head + 1 data row)    
        ├── orig_cifar10_0_cat.png
        ├── orig_cifar10_2_ship.png
        └── run.log
    """

