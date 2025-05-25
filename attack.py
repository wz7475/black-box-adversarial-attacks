from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from optimizers import AbstractOptimizer


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
        Returns: adv_tensor (tensor), success (bool), queries (int), iterations (int)
        """
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            orig_output = self.model(input_tensor)
        orig_pred = torch.argmax(orig_output, dim=1).item()
        if orig_pred != true_label:
            # attack does not make sense, model misclassified original images
            return input_tensor, None, 0, 0

        _, C, H, W = input_tensor.shape
        optimizer = self.optimizer_cls(**{**self.optimizer_kwargs, 'shape': (C, H, W)})
        queries = 0

        for iteration in range(self.num_iters):
            perturbations = optimizer.ask()
            probs = self.query_model_with_perturbation(input_tensor, perturbations)
            fitness = self.objective_function(probs, true_label, perturbations)
            preds = np.argmax(probs, axis=1)
            queries += self.optimizer_kwargs['pop_size']
            success_idxs = np.where(preds != true_label)[0]
            if success_idxs.size > 0:
                idx = success_idxs[0]
                adv_tensor = torch.from_numpy(perturbations[idx]).unsqueeze(0) + input_tensor.cpu()
                return adv_tensor, True, queries, iteration
            optimizer.tell(fitness)
            # if iteration % 100 == 0:
            #     print(f"iteration {iteration}, fitness {fitness}")
        # Attack failed: return best candidate
        perturbations = optimizer.ask()
        best_idx = np.argmax(fitness)
        adv_tensor = torch.from_numpy(perturbations[best_idx]).unsqueeze(0) + input_tensor.cpu()
        return adv_tensor, False, queries, self.optimizer_kwargs.get('num_iters', 1000)
