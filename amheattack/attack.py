from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from .optimizers import AbstractOptimizer


class AdversarialAttack:
    def __init__(self, model: torch.nn.Module, device: torch.device, num_iters: int, optimizer_cls: Type[AbstractOptimizer],
                 alpha: float, optimizer_kwargs: dict):
        self.model = model
        self.device = device
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.num_iters = num_iters
        self.alpha = alpha

    def _query_model(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()
        return probs_np

    def query_model_with_perturbation(self, base_img: torch.Tensor, perturbations: np.ndarray, ) -> np.ndarray:
        populated_images = base_img.repeat(self.optimizer_kwargs['pop_size'], 1, 1, 1)
        perturbed_images = populated_images + torch.from_numpy(perturbations).to(torch.float32).to(self.device)
        return self._query_model(perturbed_images)

    def objective_function(self, probs: np.ndarray, true_label: int, perturbations: np.ndarray) -> float:
        probs_for_true_label = probs[:, true_label]
        nll = -np.log(np.clip(probs_for_true_label, 1e-12, 1.0))
        # high values of nll indicate that model probably misclassified sample
        noise_regularization = np.sqrt(np.sum(perturbations ** 2, axis=(1, 2, 3)))
        # high values of noise_reg indicate much noise added, visual change of image will be significant
        return nll - self.alpha * noise_regularization

    def attack(self, input_tensor, true_label):
        """
        input_tensor: PyTorch tensor [1, C, H, W] with values in [0,1]
        true_label: int
        Returns: adv_tensor (best tensor), success (bool), total_queries (int), best_obj_value (float),
                 first_success_tensor (tensor or None), first_success_queries (int or None), first_success_obj (float or None)
        """
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            orig_output = self.model(input_tensor)
        orig_pred = torch.argmax(orig_output, dim=1).item()
        if orig_pred != true_label:
            # attack does not make sense, model misclassified original images
            return input_tensor, None, 0, None, None, None, None

        _, C, H, W = input_tensor.shape
        optimizer = self.optimizer_cls(**{**self.optimizer_kwargs, 'shape': (C, H, W)})
        total_queries = 0

        # Track first success
        first_success_tensor = None
        first_success_queries = None
        first_success_obj = None

        # Track best successful perturbation (highest objective among all successes)
        best_success_tensor = None
        best_success_obj = -np.inf

        fitness = None
        perturbations = None

        for _ in range(self.num_iters):
            perturbations = optimizer.ask()
            probs = self.query_model_with_perturbation(input_tensor, perturbations)
            fitness = self.objective_function(probs, true_label, perturbations)
            preds = np.argmax(probs, axis=1)
            total_queries += self.optimizer_kwargs['pop_size']

            success_idxs = np.where(preds != true_label)[0]
            for s_idx in success_idxs:
                adv_t = (torch.from_numpy(perturbations[s_idx]).unsqueeze(0) + input_tensor.cpu()).clip(0, 1)
                obj = float(fitness[s_idx])
                if first_success_tensor is None:
                    first_success_tensor = adv_t
                    first_success_queries = total_queries
                    first_success_obj = obj
                    print(f"First success at query {total_queries}: obj={obj:.4f}")
                if obj > best_success_obj:
                    best_success_obj = obj
                    best_success_tensor = adv_t

            optimizer.tell(fitness)

        success = first_success_tensor is not None
        if success:
            print(f"Best success obj={best_success_obj:.4f}")
            return best_success_tensor, True, total_queries, best_success_obj, first_success_tensor, first_success_queries, first_success_obj
        else:
            # Attack failed: return best candidate from final population
            assert fitness is not None and perturbations is not None, "num_iters must be > 0"
            best_idx = int(np.argmax(fitness))
            adv_tensor = (torch.from_numpy(perturbations[best_idx]).unsqueeze(0) + input_tensor.cpu()).clip(0, 1)
            best_obj_value = float(fitness[best_idx])
            print(f"Fail: best candidate objective function value = {best_obj_value:.4f}")
            return adv_tensor, False, total_queries, best_obj_value, None, None, None
