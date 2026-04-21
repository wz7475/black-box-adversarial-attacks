# Large-scale Testing Global Optimization Methods with Black-box Adversarial Attacks
Existing global optimization benchmark suites are of a moderate size and are based on a small number of analytical functions that date back even to 1970's. This causes a risk of biasing the development of global optimization methods. We argue that the tasks related to the black-box adversarial attack (BBAA) can serve as valuable global optimization benchmark in many-dimensional space. We demonstrate the efficiency of several types of evolutionary algorithms and other metaheuristics in solving example BBAA problems.
Thus we take a step towards convergence of global optimization methods to the challenges and needs that arise in the modern machine learning field.

## Implementation Details of Classifiers

The target classification models were trained to minimize the Cross-Entropy loss using the Adam optimizer with a learning rate of $\eta = 10^{-3}$. To ensure computational efficiency during the training phase, a large batch size of $4096$ was utilized. The CIFAR-10 model was trained for 40 epochs. Given the relative simplicity of the digit recognition task, this was sufficient to reach a high baseline accuracy.

The CIFAR-10 model uses a deeper feature extractor: Conv2d($3 \to 64$, kernel $3\times3$, padding 1) $\to$ ReLU $\to$ Conv2d($64 \to 64$, kernel $3\times3$, padding 1) $\to$ ReLU $\to$ MaxPool($2\times2$), then Conv2d($64 \to 128$, kernel $3\times3$, padding 1) $\to$ ReLU $\to$ Conv2d($128 \to 128$, kernel $3\times3$, padding 1) $\to$ ReLU $\to$ MaxPool($2\times2$). The classifier head maps $8\times8\times128 = 8192 \to 256$ with ReLU and Dropout($p=0.5$), then $256 \to 10$.

For training, each dataset was split into a training set and a validation set: the models were optimized on the training set, while the validation set was used to monitor generalization performance. During training, a "best-model" saving strategy was implemented, where the model state was preserved only when a new peak in validation accuracy was achieved. These pre-trained models were then used as the static targets for all subsequent black-box optimization experiments.

---

## Implementation Details of Optimizers

All optimizers were run with their default hyperparameter values as provided by the `mealpy` library.

The two settings shared across every optimizer are the population size $N_\text{pop} = 500$ and the maximum number of iterations $T = 500$. Optimizer-specific defaults are listed in the table below. SADE, GWO, and INFO require no additional hyperparameters beyond the shared ones, as they perform internal self-adaptation or use fixed algorithmic rules.

**Table: Default hyperparameters used for each optimizer**

| **Optim.** | **Parameter** | **Default** |
|---|---|---|
| GA | Mutation probability $p_m$ | 0.1 |
|  | Mutation type | flip, multipoint |
| JADE | Initial scale factor $\mu_F$ | 0.5 |
|  | Initial crossover rate $\mu_{CR}$ | 0.5 |
|  | Top-$p$ fraction $p_t$ | 0.1 |
|  | Adaptation rate $c$ | 0.1 |
| DE | Scale factor $F$ | 0.8 |
|  | Crossover rate $CR$ | 0.9 |
|  | Strategy | `rand/1` |
| SADE | Self-adaptive; $F$ and $CR$ adapted automatically. | |
| GWO | No additional parameters; default algorithmic rules used. | |
| SHADE | Initial scale factor $\mu_F$ | 0.5 |
|  | Initial crossover rate $\mu_{CR}$ | 0.5 |
| INFO | No additional parameters; default algorithmic rules used. | |



## setup
make sure to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) and run
```shell
uv sync
uv build
```

to run scripts
```shell
source .venv/bin/activate
python <script.py> <args>
```
or add prefix `uv run`
```shell
uv run python <script.py> <args>
```

## benchmarking already added optimizers
1. train models
```shell
python train_models.py
```
2. run attack (check help for detaul)
```shell
python -m scripts.adversarial_attack --args
python -m scripts.stochastic_growth_attack --args
```

## benchmarking new optimizer
1. Add new implmentation of `AbstractOptimizer` in `amheattack/optimizers.py` eg. `OracleOptimizer`
```python
class AbstractOptimizer(ABC):
    def ask(self) -> np.ndarray:
        """returns candidates (as many as population size)"""

    def tell(self, candidates_scores: np.ndarray) -> None:
        """updates inner params to generate better candidates"""
```

2. In `scripts/adversarial_attack.py` import new optimizer and setup it's args
```python
if args.optimizer == 'oracle':
    optimizer_cls = OracleOptimizer
    optimizer_kwargs = {
        'use_revealed_solution': True,
    }
```

3. Then benchmark your optimizer
```shell
python -m scripts.adversarial_attack --optimizer oracle --other_args
```