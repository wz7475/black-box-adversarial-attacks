from abc import ABC
import warnings

import numpy as np
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.DE import JADE, OriginalDE, SADE, SAP_DE
from mealpy.evolutionary_based.SHADE import L_SHADE, OriginalSHADE
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.math_based.INFO import OriginalINFO
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




class DEOptimizer(AbstractOptimizer):
    """Wrapper around mealpy OriginalDE (Classic Differential Evolution)"""
    def __init__(self, pop_size, shape, eps, wf=0.8, cr=0.9, strategy=0):
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
        
        # Initialize mealpy OriginalDE
        # strategy: 0=DE/rand/1/bin, 1=DE/best/1/bin, 2=DE/rand/2/bin, 3=DE/best/2/bin, 4=DE/current-to-best/1/bin
        self.optimizer = OriginalDE(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
            wf=wf,  # Weighting factor (F)
            cr=cr,  # Crossover rate
            strategy=strategy,
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




class SADEOptimizer(AbstractOptimizer):
    """Wrapper around mealpy SADE (Self-Adaptive Differential Evolution)"""
    def __init__(self, pop_size, shape, eps):
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
        
        # Initialize mealpy SADE (self-adapts F and CR parameters)
        self.optimizer = SADE(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
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



class GWOOptimizer(AbstractOptimizer):
    """Wrapper around mealpy GWO (Grey Wolf Optimizer)"""
    def __init__(self, pop_size, shape, eps):
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
        
        # Initialize mealpy OriginalGWO
        self.optimizer = OriginalGWO(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
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




class SHADEOptimizer(AbstractOptimizer):
    """Wrapper around mealpy OriginalSHADE (Success-History Adaptation DE)"""
    def __init__(self, pop_size, shape, eps, miu_f=0.5, miu_cr=0.5):
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
        
        # Initialize mealpy OriginalSHADE
        self.optimizer = OriginalSHADE(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
            miu_f=miu_f,
            miu_cr=miu_cr,
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




class LSHADEOptimizer(AbstractOptimizer):
    """Wrapper around mealpy L_SHADE (Linear Population Size Reduction SHADE)"""
    def __init__(self, pop_size, shape, eps, miu_f=0.5, miu_cr=0.5):
        self.initial_pop_size = pop_size
        self.shape = shape
        self.eps = eps
        self.n_dim = int(np.prod(shape))
        
        # Define mealpy problem with bounds [-eps, eps]
        self.problem_dict = {
            "bounds": FloatVar(lb=(-eps,) * self.n_dim, ub=(eps,) * self.n_dim, name="perturbation"),
            "minmax": "max",
            "obj_func": lambda x: 0.0,  # Dummy; we'll override with tell()
        }
        
        # Initialize mealpy L_SHADE
        self.optimizer = L_SHADE(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
            miu_f=miu_f,
            miu_cr=miu_cr,
        )
        
        # Initialize population
        self.optimizer.solve(self.problem_dict, mode='single', seed=None)
        self._iteration = 0

    def ask(self) -> np.ndarray:
        """Returns current population as perturbations shaped (current_pop_size, *shape)"""
        # L-SHADE reduces population size over time, so we get actual current size
        current_pop_size = len(self.optimizer.pop)
        perturbations = np.array([agent.solution for agent in self.optimizer.pop])
        return perturbations.reshape(current_pop_size, *self.shape).astype(np.float32)

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




class INFOOptimizer(AbstractOptimizer):
    """Wrapper around mealpy OriginalINFO (weIghted meaN oF vectOrs)"""
    def __init__(self, pop_size, shape, eps):
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
        
        # Initialize mealpy OriginalINFO
        self.optimizer = OriginalINFO(
            epoch=1,  # We manually control iterations via ask/tell
            pop_size=pop_size,
        )
        
        # Initialize population (suppress expected RuntimeWarnings from INFO algorithm)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
            self.optimizer.solve(self.problem_dict, mode='single', seed=None)
        self._iteration = 0

    def ask(self) -> np.ndarray:
        """Returns current population as perturbations shaped (pop_size, *shape)"""
        perturbations = np.array([agent.solution for agent in self.optimizer.pop])
        # Check for NaN/Inf values and clip to bounds
        perturbations = np.nan_to_num(perturbations, nan=0.0, posinf=self.eps, neginf=-self.eps)
        perturbations = np.clip(perturbations, -self.eps, self.eps)
        return perturbations.reshape(self.pop_size, *self.shape).astype(np.float32)

    def tell(self, candidates_scores: np.ndarray) -> None:
        """Update population fitness and evolve to next generation"""
        # Replace NaN scores with very negative values (for maximization)
        candidates_scores = np.nan_to_num(candidates_scores, nan=-1e10, posinf=-1e10, neginf=-1e10)
        
        # Update fitness for current population (mealpy maximizes by default in our setup)
        for idx, agent in enumerate(self.optimizer.pop):
            agent.target.set_objectives(np.array([candidates_scores[idx]]))
            agent.target.calculate_fitness(agent.target.weights)
        
        # Update global best
        pop_temp, self.optimizer.g_best = self.optimizer.update_global_best_agent(self.optimizer.pop)
        
        # Evolve to next generation (suppress expected RuntimeWarnings from INFO algorithm)
        self._iteration += 1
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
            self.optimizer.evolve(self._iteration)


