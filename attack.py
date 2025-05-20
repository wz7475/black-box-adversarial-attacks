import torch
import torch.nn.functional as F
import numpy as np

class BasicGeneticAttack:
    def __init__(self, model, pop_size=20, num_iters=1000, eps=0.1, sigma=0.1, device='cpu'):
        self.model = model
        self.pop_size = pop_size
        self.num_iters = num_iters
        self.eps = eps
        self.sigma = sigma
        self.device = device

    def attack(self, input_tensor, true_label):
        """
        input_tensor: PyTorch tensor [1, C, H, W] with values in [0,1]
        true_label: int
        Returns: adv_tensor (tensor), success (bool), queries (int), iterations (int)
        """
        # Check original prediction
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            orig_output = self.model(input_tensor)
        orig_pred = torch.argmax(orig_output, dim=1).item()
        if orig_pred != true_label:
            return input_tensor, True, 0, 0

        _, C, H, W = input_tensor.shape
        # Initialize population noise in [0,1]
        population = np.random.rand(self.pop_size, C, H, W).astype(np.float32)
        queries = 0

        for iteration in range(1, self.num_iters + 1):
            # Generate candidate tensors by converting noise to perturbations
            perturbations = (population * 2 - 1) * self.eps  # scale to [-eps, eps]
            # Expand base image
            base_img = input_tensor.cpu().numpy().transpose(0,2,3,1)[0]  # [H,W,C]
            candidates_np = []
            for i in range(self.pop_size):
                pert = perturbations[i].transpose(1,2,0)  # [H,W,C]
                cand = np.clip(base_img + pert, 0.0, 1.0)
                candidates_np.append(cand)
            candidates_np = np.stack(candidates_np)  # [pop_size, H, W, C]
            # Convert to tensor batch
            batch = torch.from_numpy(candidates_np.transpose(0,3,1,2)).to(self.device).float()
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
            queries += self.pop_size
            probs_np = probs.cpu().numpy()
            true_probs = probs_np[:, true_label]
            fitness = -np.log(np.clip(true_probs, 1e-12, 1.0))
            preds = np.argmax(probs_np, axis=1)
            success_idxs = np.where(preds != true_label)[0]
            if success_idxs.size > 0:
                idx = success_idxs[0]
                adv_tensor = batch[idx].unsqueeze(0)
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
        adv_tensor = batch[best_idx].unsqueeze(0)
        return adv_tensor, False, queries, self.num_iters