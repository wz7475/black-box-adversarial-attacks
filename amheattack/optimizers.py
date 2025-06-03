from abc import ABC

import numpy as np
from .cmaes import CMA


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
    def __init__(self, pop_size, shape, eps, sigma, n_max_resampling, maximize=True):
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
            population_size=pop_size,
            n_max_resampling=n_max_resampling,
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


class JADEOptimizer(AbstractOptimizer):
    """relies on https://github.com/ekorudiawan/JADE/blob/master/sources/jade.py"""
    def __init__(self, pop_size, shape, eps, maximize=True, uCR=0.1, uF=0.6, c=0.1):
        self.pop_size = pop_size
        self.shape = shape
        self.eps = eps
        self.n_dim = np.prod(shape)
        self.maximize = maximize
        self.uCR = uCR
        self.uF = uF
        self.c = c
        # Population in [-eps, eps]
        self.population = np.random.uniform(-eps, eps, size=(pop_size, self.n_dim)).astype(np.float32)
        self.fitness = np.zeros(pop_size, dtype=np.float32)
        self._last_candidates = None

    def ask(self) -> np.ndarray:
        # Reshape population to (pop_size, *shape)
        self._last_candidates = self.population.reshape(self.pop_size, *self.shape)
        return self._last_candidates

    def tell(self, candidates_scores: np.ndarray) -> None:
        # For maximization, use scores as is; for minimization, negate
        if not self.maximize:
            candidates_scores = -candidates_scores
        self.fitness = candidates_scores
        # JADE update
        NP = self.pop_size
        n_params = self.n_dim
        target_vectors = self.population.copy()
        target_fitness = self.fitness.copy()
        Fi = np.zeros(NP)
        CRi = np.zeros(NP)
        onethirdNP = NP // 3
        sCR = []
        sF = []
        random_onethird_idx = np.random.choice(np.arange(0, NP), size=onethirdNP, replace=False).tolist()
        for pop in range(NP):
            CRi[pop] = np.clip(np.random.normal(self.uCR, 0.1), 0, 1)
            if pop in random_onethird_idx:
                Fi[pop] = np.interp(np.random.rand(), (0, 1), (0, 1.2))
            else:
                Fi[pop] = np.clip(np.random.normal(self.uF, 0.1), 0, 1.2)
        for pop in range(NP):
            current_best_idx = np.argmax(target_fitness)
            index_choice = [i for i in range(NP) if i != pop]
            a, b = np.random.choice(index_choice, 2)
            donor_vector = target_vectors[pop] + Fi[pop] * (target_vectors[current_best_idx] - target_vectors[pop]) + \
                           Fi[pop] * (target_vectors[a] - target_vectors[b])
            cross_points = np.random.rand(n_params) <= CRi[pop]
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            # Clip to bounds
            trial_vector = np.clip(trial_vector, -self.eps, self.eps)
            trial_fitness = np.mean(self.fitness)
            if trial_fitness > target_fitness[pop]:
                target_vectors[pop] = trial_vector
                sCR.append(CRi[pop])
                sF.append(Fi[pop])
                target_fitness[pop] = trial_fitness
        # Update uCR and uF
        if sCR:
            self.uCR = (1 - self.c) * self.uCR + self.c * np.mean(sCR)
        if sF:
            self.uF = (1 - self.c) * self.uF + self.c * (np.sum(np.power(sF, 2)) / np.sum(sF))
        self.population = target_vectors


