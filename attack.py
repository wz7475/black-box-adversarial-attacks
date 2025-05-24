from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F




class BasicGeneticAttack:
    def __init__(self, model, pop_size=20, num_iters=1000, eps=0.1, sigma=0.1, device='cpu'):
        self.model = model
        self.pop_size = pop_size
        self.num_iters = num_iters
        self.eps = eps # scales to noise/perturbation applied to image
        self.sigma = sigma # controls std of gaussian mutation
        self.device = device


    def _query_model(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()
        return probs_np

    def query_model_with_perturbation(self, base_img: torch.Tensor, perturbations: np.ndarray,) -> np.ndarray:
        populated_images = base_img.repeat(self.pop_size, 1, 1, 1)
        perturbed_images = populated_images + torch.from_numpy(perturbations).to(self.device)
        return self._query_model(perturbed_images)

    @staticmethod
    def objective_function(probs: np.ndarray, true_label: int):
        probs_for_true_label = probs[:, true_label]
        nll = -np.log(np.clip(probs_for_true_label, 1e-12, 1.0))
        return nll

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
        # Initialize population noise in [0,1]
        population = np.random.rand(self.pop_size, C, H, W).astype(np.float32)
        queries = 0

        for iteration in range(1, self.num_iters + 1):
            # Generate candidate tensors by converting noise to perturbations
            perturbations = (population * 2 - 1) * self.eps  # scale to [-eps, eps]
            probs = self.query_model_with_perturbation(input_tensor, perturbations)
            fitness = self.objective_function(probs, true_label)
            preds = np.argmax(probs, axis=1)
            queries += self.pop_size
            success_idxs = np.where(preds != true_label)[0]
            if success_idxs.size > 0:
                idx = success_idxs[0]
                adv_tensor = torch.from_numpy(perturbations[idx]).unsqueeze(0) + input_tensor.cpu()
                return adv_tensor, True, queries, iteration
            # Select best noise
            best_idx = np.argmax(fitness)
            best_noise = population[best_idx]
            # Mutate to create new population
            new_pop = []
            for _ in range(self.pop_size):
                noise = best_noise + np.random.normal(loc=0.0, scale=self.sigma, size=best_noise.shape).astype(np.float32)
                noise = np.clip(noise, 0.0, 1.0)
                new_pop.append(noise)
            population = np.stack(new_pop)
        # Attack failed: return best candidate
        best_idx = np.argmax(fitness)
        adv_tensor = torch.from_numpy(perturbations[best_idx]).unsqueeze(0) + input_tensor.cpu()
        return adv_tensor, False, queries, self.num_iters