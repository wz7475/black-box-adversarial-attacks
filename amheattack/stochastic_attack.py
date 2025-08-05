import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

class StochasticGrowthAttack:
    def __init__(self, model: torch.nn.Module, device: torch.device, num_iters: int, alpha: float, seed: int = 1):
        self.model = model
        self.device = device
        self.num_iters = num_iters
        self.alpha = alpha
        self.seed = seed
        np.random.seed(self.seed)
        self.logs = []

    def _query_model(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def objective_function(self, probs: np.ndarray, true_label: int, perturbation: np.ndarray) -> float:
        probs_for_true_label = probs[:, true_label]
        nll = -np.log(np.clip(probs_for_true_label, 1e-12, 1.0))
        noise_regularization = np.sqrt(np.sum(perturbation ** 2))
        return nll - self.alpha * noise_regularization

    def reduce_perturbation_binary_search(self, input_tensor, perturbation, true_label):
        """
        For each index where perturbation > 0, perform binary search to reduce it to the smallest value
        that still causes misclassification.
        """
        reduced_perturbation = perturbation.copy()
        indices = np.argwhere(perturbation > 0)
        for idx in indices:
            idx = tuple(idx)
            low, high = 0.0, perturbation[idx]
            best_val = high
            # Binary search
            for _ in range(100):  # 10 steps for precision
                mid = (low + high) / 2
                candidate = reduced_perturbation.copy()
                candidate[idx] = mid
                perturbed_img = input_tensor + torch.from_numpy(candidate).unsqueeze(0).to(self.device)
                perturbed_img = perturbed_img.clip(0, 1)
                probs = self._query_model(perturbed_img)
                pred = np.argmax(probs, axis=1)[0]
                if pred != true_label:
                    best_val = mid
                    high = mid
                else:
                    low = mid
            reduced_perturbation[idx] = best_val
        return reduced_perturbation

    def attack(self, input_tensor, true_label):
        """
        input_tensor: PyTorch tensor [1, C, H, W] with values in [0,1]
        true_label: int
        Returns: adv_tensor (tensor), success (bool), queries (int), best_obj_value (float), hf_dataset (Dataset)
        """
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            orig_output = self.model(input_tensor)
        orig_pred = torch.argmax(orig_output, dim=1).item()
        if orig_pred != true_label:
            return input_tensor, None, 0, None, None

        _, C, H, W = input_tensor.shape
        perturbation = np.zeros((C, H, W), dtype=np.float32)
        queries = 0

        # Initial objective
        probs = self._query_model(input_tensor)
        current_obj = self.objective_function(probs, true_label, perturbation).item()
        queries += 1

        for iteration in range(self.num_iters):
            # Draw random dimension index
            idx = np.unravel_index(np.random.randint(C * H * W), (C, H, W))
            candidate = perturbation.copy()
            candidate[idx] += 1.0  # Add one to selected dimension

            # Query model with candidate perturbation
            perturbed_img = input_tensor + torch.from_numpy(candidate).unsqueeze(0).to(self.device)
            perturbed_img = perturbed_img.clip(0, 1)
            probs = self._query_model(perturbed_img)
            obj_value = self.objective_function(probs, true_label, candidate).item()
            queries += 1

            # If improves, update perturbation
            if obj_value > current_obj:
                perturbation = candidate
                current_obj = obj_value

            # Check if classifier changed class
            pred = np.argmax(probs, axis=1)[0]
            if pred != true_label:
                # Reduce perturbation before saving logs
                reduced_perturbation = self.reduce_perturbation_binary_search(input_tensor, perturbation, true_label)
                adv_tensor = torch.from_numpy(reduced_perturbation).unsqueeze(0) + input_tensor.cpu()
                adv_tensor = adv_tensor.clip(0, 1)
                print(f"StochasticGrowthAttack Success: obj={current_obj}, perturbation={reduced_perturbation}")
                self.logs.append({
                    "success": True,
                    "loss": float(current_obj),
                    "perturbation": reduced_perturbation.tolist()
                })
                hf_dataset = Dataset.from_list(self.logs)
                return adv_tensor, True, queries, current_obj, hf_dataset

        # Attack failed
        adv_tensor = torch.from_numpy(perturbation).unsqueeze(0) + input_tensor.cpu()
        adv_tensor = adv_tensor.clip(0, 1)
        print(f"StochasticGrowthAttack Fail: obj={current_obj}, perturbation={perturbation}")
        self.logs.append({
            "success": False,
            "loss": float(current_obj),
            "perturbation": perturbation.tolist()
        })
        hf_dataset = Dataset.from_list(self.logs)
        return adv_tensor, False, queries, current_obj, hf_dataset
