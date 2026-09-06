"""Export MNIST/CIFAR-10 checkpoints and a fixed image bank to the IOHNET01
binary container format used by the IOHexperimenter ``adversarial`` C++ suite.

Also emits:
  - PROVENANCE.txt: torch version, state-dict SHA256, fp32-vs-fp16 accuracy.
  - adversarial_reference.json: a golden file of objective values at
    deterministically-generated perturbations, used to cross-validate the
    C++ port and the ``ioh-bbaa`` Python package against this reference
    implementation.

Usage:
    python scripts/export_ioh_assets.py --out /path/to/IOHexperimenter/static/adversarial
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Tiny models + batch-size-1 forward passes called tens of thousands of times:
# torch's default intra-op thread pool (one per available core) makes every
# call pay thread-spin-up overhead, which dominates runtime for workloads this
# small. Single-threaded is dramatically faster here.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amheattack.models import MNISTModel, CIFARModel

DTYPE_F16 = 0
DTYPE_F32 = 1
DTYPE_U8 = 2

N_IMAGES = 100
# Perturbation grid used to build the golden regression file (mirrored by an
# identical LCG in the C++ generator and in the ioh-bbaa test suite).
EPS_ALPHA_GRID = [(0.1, 0.0), (0.1, 0.05), (0.3, 0.0), (0.3, 0.05), (1.0, 0.0), (1.0, 0.05)]
N_SEEDS_PER_IMAGE = 3

# Must match the preset table in
# IOHexperimenter/include/ioh/problem/adversarial/adversarial_attack.hpp
# (load_default_instances) exactly -- id <-> (dataset, eps, alpha).
PRESETS = [
    (5001, "mnist", 0.1, 0.0), (5002, "mnist", 0.3, 0.0), (5003, "mnist", 1.0, 0.0),
    (5004, "mnist", 0.1, 0.05), (5005, "mnist", 0.3, 0.05), (5006, "mnist", 1.0, 0.05),
    (5011, "cifar10", 0.1, 0.0), (5012, "cifar10", 0.3, 0.0), (5013, "cifar10", 1.0, 0.0),
    (5014, "cifar10", 0.1, 0.05), (5015, "cifar10", 0.3, 0.05), (5016, "cifar10", 1.0, 0.05),
]


def preset_id(dataset: str, eps: float, alpha: float) -> int:
    for pid, ds, e, a in PRESETS:
        if ds == dataset and abs(e - eps) < 1e-12 and abs(a - alpha) < 1e-12:
            return pid
    raise KeyError((dataset, eps, alpha))


def lcg_perturbation(seed: int, dim: int, eps: float) -> np.ndarray:
    """Deterministic perturbation generator, reproduced identically in C++.

    s_{n+1} = (1103515245 * s_n + 12345) mod 2^31 ; x_i = eps * (2*s/2^31 - 1)
    """
    s = seed & 0x7FFFFFFF
    out = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        out[i] = eps * (2.0 * s / 2147483648.0 - 1.0)
    return out


def write_tensor_file(path: Path, tensors: dict):
    """tensors: name -> (np.ndarray, dtype_code)"""
    with open(path, "wb") as f:
        f.write(b"IOHNET01")
        f.write(struct.pack("<I", len(tensors)))
        for name, (arr, dtype_code) in tensors.items():
            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<I", dtype_code))
            if dtype_code == DTYPE_F16:
                f.write(arr.astype(np.float16).tobytes())
            elif dtype_code == DTYPE_F32:
                f.write(arr.astype(np.float32).tobytes())
            elif dtype_code == DTYPE_U8:
                f.write(arr.astype(np.uint8).tobytes())
            else:
                raise ValueError(dtype_code)


def state_dict_sha256(model) -> str:
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        h.update(k.encode())
        h.update(model.state_dict()[k].cpu().numpy().tobytes())
    return h.hexdigest()


def weights_as_tensors(model, layer_names) -> dict:
    """layer_names: ordered list of (state_dict_prefix, exported_name) pairs."""
    sd = model.state_dict()
    tensors = {}
    for prefix, exported in layer_names:
        w = sd[f"{prefix}.weight"].detach().cpu().numpy()
        b = sd[f"{prefix}.bias"].detach().cpu().numpy()
        tensors[f"{exported}.weight"] = (w, DTYPE_F16)
        tensors[f"{exported}.bias"] = (b, DTYPE_F16)
    return tensors


def eval_fp16_roundtrip_accuracy(model, loader, layer_names):
    """Reload the model from its fp16-quantized weights and compare test accuracy
    to the original fp32 model, both computed in this same script."""
    import copy
    orig_correct = 0
    fp16_correct = 0
    total = 0

    quant_model = copy.deepcopy(model)
    sd = quant_model.state_dict()
    for prefix, _ in layer_names:
        for suffix in ("weight", "bias"):
            key = f"{prefix}.{suffix}"
            sd[key] = torch.from_numpy(
                sd[key].detach().cpu().numpy().astype(np.float16).astype(np.float32)
            )
    quant_model.load_state_dict(sd)
    quant_model.eval()

    with torch.no_grad():
        for x, y in loader:
            total += x.size(0)
            orig_correct += (model(x).argmax(1) == y).sum().item()
            fp16_correct += (quant_model(x).argmax(1) == y).sum().item()

    return orig_correct / total, fp16_correct / total


def select_correct_images(model, loader, n):
    """First n images (in dataset order) that the model classifies correctly."""
    images, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            if len(images) >= n:
                break
            pred = model(x).argmax(1).item()
            if pred == y.item():
                images.append(x.squeeze(0).numpy())
                labels.append(int(y.item()))
    if len(images) < n:
        raise RuntimeError(f"only found {len(images)} correctly-classified images, need {n}")
    return np.stack(images), np.array(labels, dtype=np.int64)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def objective_value(model, base_img: np.ndarray, label: int, delta: np.ndarray, clamp: bool) -> float:
    x = base_img.reshape(-1) + delta
    if clamp:
        x = np.clip(x, 0.0, 1.0)
    x_t = torch.from_numpy(x.reshape(base_img.shape).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x_t).squeeze(0).numpy().astype(np.float64)
    probs = softmax_np(logits)
    nll = -np.log(np.clip(probs[label], 1e-12, 1.0))
    return nll  # alpha * ||delta||_2 term applied by caller (kept separate here)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output dir, e.g. IOHexperimenter/static/adversarial")
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])

    provenance_lines = [f"torch version: {torch.__version__}", ""]
    reference = {"grid": EPS_ALPHA_GRID, "n_seeds_per_image": N_SEEDS_PER_IMAGE, "lcg": "s=(1103515245*s+12345) mod 2^31; x_i=eps*(2*s/2^31-1)", "datasets": {}}

    # ---------------- MNIST ----------------
    print("Loading MNIST model + test set...")
    mnist_model = MNISTModel()
    mnist_model.load_state_dict(torch.load(args.models_dir / "mnist.pth", map_location="cpu", weights_only=True))
    mnist_model.eval()
    mnist_test = datasets.MNIST(root=str(args.data_dir / "mnist"), train=False, download=True, transform=transform)
    mnist_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)
    mnist_loader_batched = DataLoader(mnist_test, batch_size=256, shuffle=False)

    mnist_layers = [("features.0", "conv1"), ("features.3", "conv2"), ("classifier.0", "fc1"), ("classifier.3", "fc2")]
    print("Computing fp16 round-trip accuracy (MNIST, full test set)...")
    mnist_acc_fp32, mnist_acc_fp16 = eval_fp16_roundtrip_accuracy(mnist_model, mnist_loader_batched, mnist_layers)
    provenance_lines.append(f"MNIST  state_dict sha256: {state_dict_sha256(mnist_model)}")
    provenance_lines.append(f"MNIST  fp32 test accuracy: {mnist_acc_fp32:.4f}")
    provenance_lines.append(f"MNIST  fp16-roundtrip test accuracy: {mnist_acc_fp16:.4f}  (delta={mnist_acc_fp32 - mnist_acc_fp16:+.4f})")
    provenance_lines.append("")

    print("Selecting 100 correctly-classified MNIST images...")
    mnist_images, mnist_labels = select_correct_images(mnist_model, mnist_loader, N_IMAGES)

    write_tensor_file(args.out / "mnist_cnn.ioh", weights_as_tensors(mnist_model, mnist_layers))
    write_tensor_file(args.out / "mnist_images.ioh", {
        "images": ((mnist_images * 255.0).round().astype(np.uint8), DTYPE_U8),
        "labels": (mnist_labels.astype(np.uint8), DTYPE_U8),
    })
    print(f"Wrote mnist_cnn.ioh, mnist_images.ioh ({mnist_images.shape})")

    # ---------------- CIFAR-10 ----------------
    print("Loading CIFAR-10 model + test set...")
    cifar_model = CIFARModel()
    cifar_model.load_state_dict(torch.load(args.models_dir / "cifar.pth", map_location="cpu", weights_only=True))
    cifar_model.eval()
    cifar_test = datasets.CIFAR10(root=str(args.data_dir / "cifar"), train=False, download=True, transform=transform)
    cifar_loader = DataLoader(cifar_test, batch_size=1, shuffle=False)
    cifar_loader_batched = DataLoader(cifar_test, batch_size=256, shuffle=False)

    cifar_layers = [("features.0", "conv1"), ("features.2", "conv2"), ("features.5", "conv3"),
                     ("features.7", "conv4"), ("classifier.0", "fc1"), ("classifier.3", "fc2")]
    print("Computing fp16 round-trip accuracy (CIFAR-10, full test set)...")
    cifar_acc_fp32, cifar_acc_fp16 = eval_fp16_roundtrip_accuracy(cifar_model, cifar_loader_batched, cifar_layers)
    provenance_lines.append(f"CIFAR10 state_dict sha256: {state_dict_sha256(cifar_model)}")
    provenance_lines.append(f"CIFAR10 fp32 test accuracy: {cifar_acc_fp32:.4f}")
    provenance_lines.append(f"CIFAR10 fp16-roundtrip test accuracy: {cifar_acc_fp16:.4f}  (delta={cifar_acc_fp32 - cifar_acc_fp16:+.4f})")
    provenance_lines.append("")

    print("Selecting 100 correctly-classified CIFAR-10 images...")
    cifar_images, cifar_labels = select_correct_images(cifar_model, cifar_loader, N_IMAGES)

    write_tensor_file(args.out / "cifar_cnn.ioh", weights_as_tensors(cifar_model, cifar_layers))
    write_tensor_file(args.out / "cifar_images.ioh", {
        "images": ((cifar_images * 255.0).round().astype(np.uint8), DTYPE_U8),
        "labels": (cifar_labels.astype(np.uint8), DTYPE_U8),
    })
    print(f"Wrote cifar_cnn.ioh, cifar_images.ioh ({cifar_images.shape})")

    # ---------------- Golden reference file ----------------
    print("Generating golden reference values...")
    in_lines = []  # "fid iid seed y" rows, consumed by the C++ regression test
    for name, model, images, labels in [("mnist", mnist_model, mnist_images, mnist_labels),
                                         ("cifar10", cifar_model, cifar_images, cifar_labels)]:
        dim = images[0].size
        entries = []
        for img_idx in range(5):  # first 5 images is enough to catch regressions
            base = images[img_idx]
            label = int(labels[img_idx])
            iid = img_idx + 1  # AdversarialAttack instance ids are 1-based
            for seed in range(N_SEEDS_PER_IMAGE):
                for eps, alpha in EPS_ALPHA_GRID:
                    delta = lcg_perturbation(seed=seed * 1000 + img_idx, dim=dim, eps=eps)
                    nll = objective_value(model, base, label, delta, clamp=False)
                    l2 = float(np.sqrt(np.sum(delta ** 2)))
                    f_val = nll - alpha * l2  # MAX orientation, matches reference code
                    entries.append({
                        "image_index": img_idx, "true_label": label, "seed": seed,
                        "eps": eps, "alpha": alpha, "clamp": False, "objective": f_val,
                    })
                    fid = preset_id(name, eps, alpha)
                    in_lines.append(f"{fid} {iid} {seed} {float(f_val)!r}")
        reference["datasets"][name] = {"dim": dim, "entries": entries}
        print(f"  {name}: {len(entries)} golden entries")

    with open(args.out / "adversarial_reference.json", "w") as f:
        json.dump(reference, f, indent=2)
    # Lives directly under static/ (sibling of pbofitness*.in, bbobfitness*.in),
    # not under static/adversarial/, to match the repo's other regression files.
    in_path = args.out.parent / "adversarial.in"
    with open(in_path, "w") as f:
        f.write("\n".join(in_lines) + "\n")
    print(f"Wrote {in_path} ({len(in_lines)} rows)")

    provenance_lines.append("Image bank: first 100 correctly-classified test-set images, natural")
    provenance_lines.append("(shuffle=False) order, transforms.ToTensor() only (no normalization).")
    provenance_lines.append("Golden file: adversarial_reference.json, generated by this script.")
    (args.out / "PROVENANCE.txt").write_text("\n".join(provenance_lines) + "\n")

    print("\n".join(provenance_lines))
    print(f"\nDone. Wrote assets to {args.out}")


if __name__ == "__main__":
    main()
