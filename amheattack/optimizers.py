from abc import ABC

import numpy as np
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.DE import JADE
from mealpy import FloatVar


class AbstractOptimizer(ABC):
    def ask(self) -> np.ndarray:
        """returns candidates (as many as population size)"""

    def tell(self, candidates_scores: np.ndarray) -> None:
        """updates inner params to generate better candidates"""



class GeneticAlgOptimizer(AbstractOptimizer):
    """Wrapper around mealpy BaseGA with mutation-only (pc=0)"""
    def __init__(self, pop_size, shape, eps, sigma, pm=0.1, mutation="flip", mutation_multipoints=True):
        self.pop_size = pop_size
        self.shape = shape
        self.eps = eps
        self.n_dim = int(np.prod(shape))
        
        # Define mealpy problem with bounds [-eps, eps]
        self.problem_dict = {
            "bounds": FloatVar(lb=(-eps,) * self.n_dim, ub=(eps,) * self.n_dim, name="perturbation"),
            "minmax": "max",
            "obj_func": lambda x: 0.0,  # Dummy; we'll override with tell()
        }
        
        # Initialize mealpy BaseGA with very low pc (effectively no crossover, mutation only)
        self.optimizer = BaseGA(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
            pc=0.001,  # Minimal crossover (validator requires > 0)
            pm=pm,
            mutation=mutation,
            mutation_multipoints=mutation_multipoints,
        )
        
        # Initialize population
        self.optimizer.solve(self.problem_dict, mode='single', seed=None)
        self._iteration = 0

    def ask(self) -> np.ndarray:
        """Returns current population as perturbations shaped (pop_size, *shape)"""
        perturbations = np.array([agent.solution for agent in self.optimizer.pop])
        return perturbations.reshape(self.pop_size, *self.shape).astype(np.float32)

    def tell(self, candidates_scores: np.ndarray) -> None:
        """Update population fitness and evolve to next generation"""
        # Update fitness for current population (mealpy maximizes by default in our setup)
        for idx, agent in enumerate(self.optimizer.pop):
            agent.target.set_objectives(np.array([candidates_scores[idx]]))
            agent.target.calculate_fitness(agent.target.weights)
        
        # Update global best
        pop_temp, self.optimizer.g_best = self.optimizer.update_global_best_agent(self.optimizer.pop)
        
        # Evolve to next generation
        self._iteration += 1
        self.optimizer.evolve(self._iteration)




class JADEOptimizer(AbstractOptimizer):
    """Wrapper around mealpy JADE (Differential Evolution)"""
    def __init__(self, pop_size, shape, eps, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1):
        self.pop_size = pop_size
        self.shape = shape
        self.eps = eps
        self.n_dim = int(np.prod(shape))
        
        # Define mealpy problem with bounds [-eps, eps]
        self.problem_dict = {
            "bounds": FloatVar(lb=(-eps,) * self.n_dim, ub=(eps,) * self.n_dim, name="perturbation"),
            "minmax": "max",
            "obj_func": lambda x: 0.0,  # Dummy; we'll override with tell()
        }
        
        # Initialize mealpy JADE
        self.optimizer = JADE(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
            miu_f=miu_f,
            miu_cr=miu_cr,
            pt=pt,
            ap=ap,
        )
        
        # Initialize population
        self.optimizer.solve(self.problem_dict, mode='single', seed=None)
        self._iteration = 0

    def ask(self) -> np.ndarray:
        """Returns current population as perturbations shaped (pop_size, *shape)"""
        perturbations = np.array([agent.solution for agent in self.optimizer.pop])
        return perturbations.reshape(self.pop_size, *self.shape).astype(np.float32)

    def tell(self, candidates_scores: np.ndarray) -> None:
        """Update population fitness and evolve to next generation"""
        # Update fitness for current population (mealpy maximizes by default in our setup)
        for idx, agent in enumerate(self.optimizer.pop):
            agent.target.set_objectives(np.array([candidates_scores[idx]]))
            agent.target.calculate_fitness(agent.target.weights)
        
        # Update global best
        pop_temp, self.optimizer.g_best = self.optimizer.update_global_best_agent(self.optimizer.pop)
        
        # Evolve to next generation
        self._iteration += 1
        self.optimizer.evolve(self._iteration)


