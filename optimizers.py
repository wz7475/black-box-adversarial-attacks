from abc import ABC

import numpy as np
from cmaes import CMA


class AbstractOptimizer(ABC):
    def ask(self) -> np.ndarray:
        """returns candidates (as many as population size)"""

    def tell(self, candidates_scores: np.ndarray) -> None:
        """updates inner params to generate better candidates"""


class GeneticAlgOptimizer(AbstractOptimizer):
    def __init__(self, pop_size, shape, eps, sigma):
        self.pop_size = pop_size
        self.shape = shape
        self.eps = eps
        self.sigma = sigma
        # Initialize population in [0,1]
        self.population = np.random.rand(self.pop_size, *self.shape).astype(np.float32)
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)

    def ask(self) -> np.ndarray:
        return (self.population * 2 - 1) * self.eps  # scale to [-eps, eps]

    def tell(self, candidates_scores: np.ndarray) -> None:
        self.fitness = candidates_scores
        best_idx = np.argmax(self.fitness)
        best_noise = self.population[best_idx]
        # Mutate to create new population
        new_pop = []
        for _ in range(self.pop_size):
            noise = best_noise + np.random.normal(loc=0.0, scale=self.sigma, size=best_noise.shape).astype(np.float32)
            noise = np.clip(noise, 0.0, 1.0)
            new_pop.append(noise)
        self.population = np.stack(new_pop)


class CMAESOptimizer(AbstractOptimizer):
    def __init__(self, pop_size, shape, eps, sigma, maximize=True):
        self.pop_size = pop_size
        self.shape = shape
        self.eps = eps
        self.sigma = sigma
        self.n_dim = np.prod(shape)
        self.maximize = maximize
        # Set bounds to [-eps, eps] for each dimension
        bounds = np.array([[-eps, eps]] * self.n_dim)
        self.cma = CMA(
            mean=np.zeros(self.n_dim, dtype=np.float32),
            sigma=sigma,
            bounds=bounds,
            population_size=pop_size
        )
        self._last_candidates = None

    def ask(self) -> np.ndarray:
        # Ask for pop_size candidates, reshape to (pop_size, *shape)
        candidates = []
        for i in range(self.pop_size):
            x = self.cma.ask()
            candidates.append(x)
            print(f"cma ask iteration {i}")
        self._last_candidates = np.stack(candidates)
        return self._last_candidates.reshape(self.pop_size, *self.shape)

    def tell(self, candidates_scores: np.ndarray) -> None:
        # Flatten candidates to (pop_size, n_dim)
        if not self.maximize:
            candidates_scores = -candidates_scores
        solutions = [
            (self._last_candidates[i], candidates_scores[i])
            for i in range(self.pop_size)
        ]
        self.cma.tell(solutions)
