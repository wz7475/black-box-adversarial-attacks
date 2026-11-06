# amheattack — black-box adversarial attacks on image classifiers

`amheattack` is a toolkit for crafting adversarial examples against image classifiers whose **architecture, weights and gradients are unknown**. The attacker only sends images to the model and reads back the output probabilities. It then searches for a small perturbation that changes the predicted class.

Because the search uses no gradients, the attack is a derivative-free optimization problem. Any ask/tell optimizer can be plugged in. The package ships wrappers for several evolutionary algorithms and metaheuristics from [`mealpy`](https://github.com/thieu1995/mealpy), plus a simple stochastic-growth baseline.

## Contents

- [How it works](#how-it-works)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Attacking your own model](#attacking-your-own-model)
- [Attack methods](#attack-methods)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Analysis scripts](#analysis-scripts)
- [Cluster (SLURM) runs](#cluster-slurm-runs)
- [Adding a new optimizer](#adding-a-new-optimizer)
- [Repository layout](#repository-layout)
- [Reference models](#reference-models)
- [License](#license)

## How it works

**Threat model.** The attack treats the classifier as an oracle. It is only ever called as `model(images)` under `torch.no_grad()`. The attack code does not read layers, weights or gradients, so the model can be any architecture. The oracle must:

- accept a float tensor `[B, C, H, W]` with values in `[0, 1]`;
- return class logits `[B, num_classes]`. The attack applies softmax itself and uses only the resulting probabilities.

**Search problem.** For an input image `x` with true label `y`, the attack looks for a perturbation `δ` with the same shape as `x` (`C × H × W`). The population-based attacks bound it per pixel, `δ ∈ [-eps, eps]`. They maximize

```
objective(δ) = -log p(y | x + δ)  -  alpha · ‖δ‖₂
```

- The first term is the negative log-likelihood of the true class. It is large when the model is no longer confident in `y`.
- The second term penalizes large perturbations. `alpha = 0` ignores perturbation size.

An attack **succeeds** when the top-1 prediction of `x + δ` differs from `y`. Images the model already misclassifies are skipped. Adversarial images are clipped to `[0, 1]`.

**Query budget.** Every model evaluation counts as a query. For the population-based attacks, `queries = pop_size × iterations`, and the success query is the one at the end of the generation that produced the first misclassification. The attack records two candidates:

- the **first success**: the first misclassifying perturbation, with the number of queries it took;
- the **best success**: the misclassifying perturbation with the highest objective value seen during the run.

If no candidate succeeds, the best candidate of the final population is returned and the attack is logged as a failure.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```shell
uv sync
uv build
```

Run scripts with the virtual environment activated:

```shell
source .venv/bin/activate
python -m scripts.adversarial_attack --help
```

or prefix any command with `uv run`:

```shell
uv run python -m scripts.adversarial_attack --help
```

Run all `python -m scripts.<name>` commands from the repository root. Datasets are downloaded to `data/` on first use.

## Quick start

### 1. Get a target model

Train the MNIST and CIFAR-10 reference models (see [Reference models](#reference-models)):

```shell
cd scripts && python train_models.py && cd ..
```

`train_models.py` writes checkpoints to `../models/` relative to the current directory, so run it from `scripts/`. This creates `models/mnist.pth` and `models/cifar.pth` in the repository root, where the attack scripts expect them.

The `imagenet` target (`microsoft/resnet-18`) needs no training. It is downloaded from the HuggingFace Hub. The ImageNet validation images come from the gated `ILSVRC/imagenet-1k` dataset, so log in with `huggingface-cli login` or set `HF_TOKEN` first.

### 2. Run an attack

Population-based attack with an optimizer of your choice:

```shell
python -m scripts.adversarial_attack \
    --model cifar10 --optimizer jade \
    --eps 0.1 --alpha 10 --pop_size 500 --num_iters 500 \
    --test_size 50 --output_dir output
```

Stochastic-growth baseline:

```shell
python -m scripts.stochastic_growth_attack \
    --model cifar10 --alpha 0.1 --num_iters 500 \
    --start_from_test_idx 0 --test_size 5 --repeat 10
```

Both scripts print per-image results and write logs and images to the output directory (see [Outputs](#outputs)).

## Attacking your own model

The CLI scripts know three targets: `mnist`, `cifar10` and `imagenet`. To attack another model, wrap it as a `torch.nn.Module` that meets the oracle contract above, then use `AdversarialAttack` directly:

```python
import torch
from amheattack.attack import AdversarialAttack
from amheattack.optimizers import JADEOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyBlackBox().to(device).eval()   # any module: [B,C,H,W] in [0,1] -> logits

attacker = AdversarialAttack(
    model=model,
    device=device,
    num_iters=500,
    optimizer_cls=JADEOptimizer,
    alpha=10.0,
    optimizer_kwargs={"pop_size": 500, "eps": 0.1},  # `shape` is filled in automatically
)

(adv_img, success, queries, best_obj,
 first_adv_img, first_queries, first_obj) = attacker.attack(image, true_label)  # image: [1,C,H,W]
```

`attack()` returns:

| Value | Meaning |
|---|---|
| `adv_img` | Best adversarial image on success, otherwise the best candidate of the last population. |
| `success` | `True` / `False`, or `None` if the model already misclassified the input (nothing to attack). |
| `queries` | Total number of model evaluations. |
| `best_obj` | Objective value of `adv_img`. |
| `first_adv_img`, `first_queries`, `first_obj` | The first successful candidate, its query count and objective value (`None` if the attack failed). |

To use your model from the CLI scripts, add a branch for it in `load_model()` in `scripts/adversarial_attack.py` and `scripts/stochastic_growth_attack.py`, and add its name to `--model` choices.

If your real target is a remote API rather than a local `nn.Module`, put the API call inside the `forward()` of a thin wrapper that returns logits. Note that `AdversarialAttack` queries the model with whole populations (`pop_size` images at a time).

## Attack methods

### Population-based attacks (`scripts/adversarial_attack.py`)

Each optimizer is wrapped behind the `AbstractOptimizer` ask/tell interface in `amheattack/optimizers.py`. Per generation, `AdversarialAttack` calls `ask()` for a population of perturbations, evaluates them in one batch, computes the objective and calls `tell(objective_values)`.

| `--optimizer` | Class | Algorithm |
|---|---|---|
| `gen` | `GeneticAlgOptimizer` | Genetic algorithm (mutation-only) |
| `de` | `DEOptimizer` | Classic differential evolution |
| `jade` | `JADEOptimizer` | JADE (adaptive DE) |
| `sade` | `SADEOptimizer` | Self-adaptive DE |
| `shade` | `SHADEOptimizer` | Success-history adaptive DE |
| `lshade` | `LSHADEOptimizer` | L-SHADE (SHADE with linear population-size reduction) |
| `gwo` | `GWOOptimizer` | Grey Wolf Optimizer |
| `info` | `INFOOptimizer` | INFO (weIghted meaN oF vectOrs) |

Defaults used by the CLI (all other `mealpy` settings are left at their library defaults):

- Shared: population size `--pop_size 500`, iterations `--num_iters 500`.
- SADE, GWO and INFO need no extra hyperparameters. They self-adapt or use fixed rules.

| Optimizer | Parameter | CLI flag | Default |
|---|---|---|---|
| GA | Mutation probability | `--pm` | 0.1 |
|  | Mutation type | `--mutation` | `flip` (multipoint on, `--mutation_multipoints`) |
| DE | Scale factor `F` | `--wf` | 0.8 |
|  | Crossover rate `CR` | `--cr` | 0.9 |
|  | Strategy | `--strategy` | 0 (`rand/1`) |
| JADE | Initial scale factor `μ_F` | `--miu_f` | 0.5 |
|  | Initial crossover rate `μ_CR` | `--miu_cr` | 0.5 |
|  | Top-`p` fraction | `--pt` | 0.1 |
|  | Adaptation rate `c` | `--ap` | 0.1 |
| SHADE | Initial `μ_F` / `μ_CR` | `--shade_miu_f` / `--shade_miu_cr` | 0.5 / 0.5 |
| L-SHADE | Initial `μ_F` / `μ_CR` | `--lshade_miu_f` / `--lshade_miu_cr` | 0.5 / 0.5 |

DE strategies: `0` rand/1, `1` best/1, `2` rand/2, `3` best/2, `4` current-to-best/1.

### Stochastic growth baseline (`scripts/stochastic_growth_attack.py`)

A simple random-search attack in `amheattack/stochastic_attack.py`:

1. Start from `δ = 0`.
2. Each iteration, pick a random pixel coordinate and add `1.0` to it. The perturbed image is clipped to `[0, 1]`, so this saturates the pixel.
3. Keep the change only if the objective improves.
4. Stop at the first misclassification. Then shrink every non-zero coordinate with a binary search down to the smallest value that still misclassifies.

The attack is repeated `--repeat` times per image with different seeds (`--seed`, `--seed + 1`, …), which makes it useful for studying how perturbations differ across runs and target classes.

## CLI reference

Run `--help` on either script for the full list of flags.

**`scripts.adversarial_attack`**

| Flag | Default | Description |
|---|---|---|
| `--model` | `cifar10` | Target model: `mnist`, `cifar10`, `imagenet` |
| `--optimizer` | `jade` | See the optimizer table above |
| `--test_size` | 50 | Number of test images to attack |
| `--eps` | 0.1 | Per-pixel perturbation bound `[-eps, eps]` |
| `--alpha` | 10.0 | Weight of the perturbation-size penalty |
| `--pop_size` | 500 | Population size (queries per iteration) |
| `--num_iters` | 500 | Number of optimizer iterations |
| `--output_dir` | `output` | Where to write results |

**`scripts.stochastic_growth_attack`**

| Flag | Default | Description |
|---|---|---|
| `--model` | `cifar10` | Target model |
| `--test_size` | 2 | Number of images to attack |
| `--start_from_test_idx` | 0 | Index of the first test image |
| `--num_iters` | 500 | Iterations (= queries) per attempt |
| `--alpha` | 0.1 | Weight of the perturbation-size penalty |
| `--repeat` | 2 | Attempts per image |
| `--seed` | 1 | Base random seed |
| `--output_dir` | `output_stochastic` | Where to write results |

## Outputs

`scripts.adversarial_attack` creates one sub-directory per configuration, named after the model, optimizer and parameters:

```
output/
└── cifar10_gen_eps_0.1_alpha_10.0_pop_size_500_eps_0.1_sigma_0.1_pm_0.1_mutation_flip_mutation_multipoints_True/
    ├── args.txt               # CLI arguments of the run
    ├── run.log                # per-image log
    ├── log.csv                # one row per attacked image
    ├── aggregation.csv        # summary statistics of log.csv
    ├── orig_cifar10_0_cat.png
    └── adv_cifar10_0_cat_to_dog.png   # saved for successful attacks
```

`log.csv` columns: `idx, true_label, pred_label, success, queries, l2_dist, obj_value, first_success_queries, first_success_l2, first_success_obj`.

`l2_dist` is the L2 norm of the perturbation divided by `sqrt(C·H·W)` (the root-mean-square change per value). `first_success_*` describe the first misclassifying candidate. They are empty when the attack failed.

`aggregation.csv` contains the success ratio and the mean and standard deviation of queries, L2 distance and objective value over the images.

`scripts.stochastic_growth_attack` writes `output_stochastic/<model>_stochastic_alpha_<alpha>_test_img_<start>-<end>_iters_<n>/img_<idx>/`, which holds `log.csv` (one row per repeat, including `seed`), `aggregation.csv`, the original and adversarial PNGs, and a `hf_dataset/` (HuggingFace `datasets` format) with the raw perturbations.

## Analysis scripts

| Script | Purpose |
|---|---|
| `scripts/adversarial_attacks_merge_results_to_table.py` | Merge all `aggregation.csv` files under `--output_dir` into one table (`--output_file`, default `merged_results.csv`). |
| `scripts/analyze_stochastic_growth.py` | Per-image statistics (distinct predicted classes, mean/std L2, per-class L2) for stochastic-growth runs. |
| `scripts/stochastic_growth_attl_distance_from_mean_image.py` | Distance of each run's perturbation from the mean perturbation of its image (CSV + JSON). |
| `scripts/avg_pertubration_stochastic_growth.py` | Render the mean and standard deviation of perturbations per image as heat maps. |
| `scripts/collect_perturbation_images.py` | Gather the rendered mean-perturbation images into one directory. |
| `scripts/figures/boxplot.py` | Box plots of L2 distance and objective value for CIFAR-10 vs. ImageNet runs. |
| `scripts/export_ioh_assets.py` | Export MNIST/CIFAR-10 checkpoints and a fixed image bank to the binary container format used by the IOHexperimenter `adversarial` suite (`--out` required). |

Notebooks in `notebooks/` reproduce the plots built from these outputs. Precomputed result tables are in `results/` and training logs in `logs/`.

## Cluster (SLURM) runs

`shell_scripts/` has job-array launchers for grid searches over model, `eps`, `alpha` and optimizer (`sbatch_optimizers*.sh`), and for the stochastic-growth attack (`sbatch_stochastic_attack*.sh`, `run_tmux_stochastic_attack.sh`). The scripts take the log directory, CPUs, memory, account, partition and time limit as positional arguments. The defaults are specific to one cluster (account and partition names), so adjust them before use.

## Adding a new optimizer

1. Implement `AbstractOptimizer` in `amheattack/optimizers.py`, for example `OracleOptimizer`. The constructor receives `pop_size`, `eps` and `shape` (the perturbation shape `(C, H, W)`) plus your own keyword arguments:

    ```python
    class AbstractOptimizer(ABC):
        def ask(self) -> np.ndarray:
            """returns candidates (as many as population size)"""

        def tell(self, candidates_scores: np.ndarray) -> None:
            """updates inner params to generate better candidates"""
    ```

    `ask()` must return a float32 array shaped `(pop_size, *shape)`, and `tell()` receives one objective value per candidate (higher is better). The existing `mealpy` wrappers in the same file are working examples.

2. Register it in `scripts/adversarial_attack.py`: import the class, add its name to the `--optimizer` choices, and add a branch that sets `optimizer_cls` and `optimizer_kwargs`:

    ```python
    if args.optimizer == 'oracle':
        optimizer_cls = OracleOptimizer
        optimizer_kwargs = {
            'pop_size': args.pop_size,
            'eps': args.eps,
            'use_revealed_solution': True,
        }
    ```

3. Benchmark it:

    ```shell
    python -m scripts.adversarial_attack --optimizer oracle --other_args
    ```

## Repository layout

```
amheattack/
├── attack.py             # AdversarialAttack: query loop, objective, success tracking
├── optimizers.py         # AbstractOptimizer + mealpy-based optimizer wrappers
├── stochastic_attack.py  # StochasticGrowthAttack baseline
├── models.py             # MNISTModel, CIFARModel, HFResNetModel (reference targets)
└── utils.py              # dataset loaders, ResultLogger, aggregate_log_csv
scripts/                  # attack runners, model training, analysis and figures
shell_scripts/            # SLURM / tmux launchers
notebooks/                # analysis notebooks
results/, logs/           # precomputed result tables and training logs
```

## Reference models

These models are only the example targets bundled with the repo. The attack itself is independent of them.

| Name | Architecture | Training |
|---|---|---|
| `mnist` | Conv(1→32, 5×5) → ReLU → MaxPool → Conv(32→64, 5×5) → ReLU → MaxPool → FC(3136→1024) → ReLU → Dropout(0.5) → FC(1024→10) | 5 epochs |
| `cifar10` | Conv(3→64) → ReLU → Conv(64→64) → ReLU → MaxPool → Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool → FC(8192→256) → ReLU → Dropout(0.5) → FC(256→10) (all convs 3×3, padding 1) | 40 epochs |
| `imagenet` | Pretrained `microsoft/resnet-18` from HuggingFace, wrapped so it accepts `[0, 1]` images (ImageNet normalization is applied inside) | Not trained here |

`scripts/train_models.py` trains `mnist` and `cifar10` with cross-entropy loss, Adam (learning rate `1e-3`) and batch size 4096. After every epoch it evaluates on the test split and saves a checkpoint only when that accuracy improves. Reference accuracies from `logs/models_training.txt`: MNIST 97.9%, CIFAR-10 71.7%.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
