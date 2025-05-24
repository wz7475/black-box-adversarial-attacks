from abc import ABC

import numpy as np


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
