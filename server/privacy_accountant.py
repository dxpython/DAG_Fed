import numpy as np
import torch
from collections import defaultdict, deque
from scipy.special import comb
from scipy.optimize import minimize_scalar


class PrivacyAccountant:
    def __init__(self, config):
        self.config = config

        self.target_epsilon = config.target_epsilon
        self.target_delta = config.target_delta
        self.total_steps = config.total_steps

        self.client_privacy_costs = defaultdict(list)
        self.global_privacy_cost = 0.0

        self.alpha_max = config.alpha_max if hasattr(config, 'alpha_max') else 32
        self.alpha_orders = np.arange(2, self.alpha_max + 1)

        self.privacy_loss_history = deque(maxlen=config.history_size)

        self.composition_method = config.composition_method if hasattr(config, 'composition_method') else 'rdp'

        self.multi_level_perturbation = config.multi_level_perturbation if hasattr(config,
                                                                                   'multi_level_perturbation') else False
        self.level1_epsilon = config.level1_epsilon if hasattr(config, 'level1_epsilon') else 0.5
        self.level2_epsilon = config.level2_epsilon if hasattr(config, 'level2_epsilon') else 0.3

    def add_privacy_cost(self, client_id, privacy_cost):
        if isinstance(privacy_cost, dict):
            self.client_privacy_costs[client_id].append(privacy_cost)
        else:
            self.client_privacy_costs[client_id].append({'total': privacy_cost})

        self._update_global_privacy_cost()

    def compute_rdp_cost(self, noise_multiplier, batch_size, dataset_size, steps=1):
        if noise_multiplier <= 0:
            return np.inf * np.ones_like(self.alpha_orders)

        q = batch_size / dataset_size

        rdp_costs = np.zeros_like(self.alpha_orders, dtype=float)

        for i, alpha in enumerate(self.alpha_orders):
            if alpha == 1:
                continue

            rdp_costs[i] = q * alpha / (2 * noise_multiplier ** 2)

        return rdp_costs * steps

    def compute_layered_rdp_cost(self, layer_noise_multipliers, batch_size, dataset_size, steps=1):
        total_rdp = np.zeros_like(self.alpha_orders, dtype=float)
        layer_rdp = {}

        for layer_name, noise_multiplier in layer_noise_multipliers.items():
            layer_rdp[layer_name] = self.compute_rdp_cost(
                noise_multiplier, batch_size, dataset_size, steps
            )
            total_rdp += layer_rdp[layer_name]

        return {
            'total_rdp': total_rdp,
            'layer_rdp': layer_rdp,
            'alpha_orders': self.alpha_orders
        }

    def compute_multi_level_perturbation_cost(self, base_noise_multiplier, batch_size, dataset_size, steps=1):
        if not self.multi_level_perturbation:
            return self.compute_rdp_cost(base_noise_multiplier, batch_size, dataset_size, steps)

        level1_noise = base_noise_multiplier * np.sqrt(
            self.level1_epsilon / (self.level1_epsilon + self.level2_epsilon))
        level2_noise = base_noise_multiplier * np.sqrt(
            self.level2_epsilon / (self.level1_epsilon + self.level2_epsilon))

        level1_rdp = self.compute_rdp_cost(level1_noise, batch_size, dataset_size, steps)
        level2_rdp = self.compute_rdp_cost(level2_noise, batch_size, dataset_size, steps)

        combined_rdp = np.minimum(level1_rdp + level2_rdp,
                                  self.compute_rdp_cost(base_noise_multiplier, batch_size, dataset_size, steps))

        return combined_rdp

    def apply_differential_perturbation(self, gradients, noise_multiplier, sensitivity=1.0):
        if not self.multi_level_perturbation:
            noise = torch.normal(0, noise_multiplier * sensitivity, size=gradients.shape)
            return gradients + noise

        level1_multiplier = noise_multiplier * np.sqrt(
            self.level1_epsilon / (self.level1_epsilon + self.level2_epsilon))
        level2_multiplier = noise_multiplier * np.sqrt(
            self.level2_epsilon / (self.level1_epsilon + self.level2_epsilon))

        level1_noise = torch.normal(0, level1_multiplier * sensitivity, size=gradients.shape)
        level2_noise = torch.normal(0, level2_multiplier * sensitivity, size=gradients.shape)

        perturbed_gradients = gradients + level1_noise + level2_noise

        return perturbed_gradients

    def rdp_to_dp(self, rdp_costs, delta=None):
        if delta is None:
            delta = self.target_delta

        if np.any(np.isinf(rdp_costs)):
            return np.inf

        eps_values = []
        for i, alpha in enumerate(self.alpha_orders):
            if alpha > 1:
                eps = rdp_costs[i] + np.log(delta) / (alpha - 1)
                eps_values.append(eps)

        if not eps_values:
            return np.inf

        return min(eps_values)

    def compose_privacy_costs(self, privacy_costs_list):
        if self.composition_method == 'rdp':
            return self._rdp_composition(privacy_costs_list)
        elif self.composition_method == 'advanced':
            return self._advanced_composition(privacy_costs_list)
        else:
            return self._basic_composition(privacy_costs_list)

    def _rdp_composition(self, privacy_costs_list):
        total_rdp = np.zeros_like(self.alpha_orders, dtype=float)

        for cost in privacy_costs_list:
            if isinstance(cost, dict) and 'total_rdp' in cost:
                total_rdp += cost['total_rdp']
            elif isinstance(cost, np.ndarray):
                total_rdp += cost

        return {
            'total_rdp': total_rdp,
            'epsilon': self.rdp_to_dp(total_rdp),
            'delta': self.target_delta
        }

    def _advanced_composition(self, privacy_costs_list):
        total_epsilon = sum(cost.get('epsilon', 0) for cost in privacy_costs_list)
        k = len(privacy_costs_list)

        if k == 0:
            return {'epsilon': 0, 'delta': 0}

        delta_prime = self.target_delta / (2 * k)

        epsilon_adv = total_epsilon + np.sqrt(2 * k * np.log(1 / delta_prime)) * total_epsilon / k

        return {
            'epsilon': epsilon_adv,
            'delta': self.target_delta
        }

    def _basic_composition(self, privacy_costs_list):
        total_epsilon = sum(cost.get('epsilon', 0) for cost in privacy_costs_list)
        total_delta = sum(cost.get('delta', 0) for cost in privacy_costs_list)

        return {
            'epsilon': total_epsilon,
            'delta': min(total_delta, 1.0)
        }

    def _update_global_privacy_cost(self):
        all_costs = []
        for client_costs in self.client_privacy_costs.values():
            all_costs.extend(client_costs)

        if all_costs:
            composed_cost = self.compose_privacy_costs(all_costs)
            self.global_privacy_cost = composed_cost.get('epsilon', 0)

            self.privacy_loss_history.append({
                'epsilon': self.global_privacy_cost,
                'delta': composed_cost.get('delta', self.target_delta),
                'step': len(self.privacy_loss_history)
            })

    def get_privacy_budget_status(self):
        remaining_budget = max(0, self.target_epsilon - self.global_privacy_cost)

        status = {
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta,
            'consumed': self.global_privacy_cost,
            'remaining': remaining_budget,
            'utilization': self.global_privacy_cost / self.target_epsilon if self.target_epsilon > 0 else 0,
            'exhausted': self.global_privacy_cost >= self.target_epsilon
        }

        return status

    def check_budget_exhausted(self):
        return self.global_privacy_cost >= self.target_epsilon

    def get_client_privacy_status(self, client_id):
        if client_id not in self.client_privacy_costs:
            return {'costs': [], 'total_epsilon': 0}

        client_costs = self.client_privacy_costs[client_id]
        composed_cost = self.compose_privacy_costs(client_costs)

        return {
            'costs': client_costs,
            'total_epsilon': composed_cost.get('epsilon', 0),
            'total_delta': composed_cost.get('delta', self.target_delta)
        }

    def estimate_remaining_steps(self, current_step, avg_cost_per_step):
        if avg_cost_per_step <= 0:
            return float('inf')

        remaining_budget = self.target_epsilon - self.global_privacy_cost
        return max(0, remaining_budget / avg_cost_per_step)

    def optimize_noise_allocation(self, layer_importances, total_noise_budget):
        def objective(allocations):
            cost = 0
            for i, (layer_name, importance) in enumerate(layer_importances.items()):
                cost += importance * allocations[i] ** 2
            return cost

        def constraint(allocations):
            return sum(allocations) - total_noise_budget

        num_layers = len(layer_importances)
        bounds = [(0.01, total_noise_budget) for _ in range(num_layers)]

        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0=np.ones(num_layers) * total_noise_budget / num_layers,
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint}
        )

        if result.success:
            return dict(zip(layer_importances.keys(), result.x))
        else:
            uniform_allocation = total_noise_budget / num_layers
            return {name: uniform_allocation for name in layer_importances.keys()}

    def optimize_multi_level_allocation(self, total_epsilon_budget):
        if not self.multi_level_perturbation:
            return {'level1': total_epsilon_budget, 'level2': 0}

        def objective(allocation):
            level1_eps, level2_eps = allocation
            if level1_eps + level2_eps > total_epsilon_budget:
                return np.inf

            effectiveness = level1_eps * 0.7 + level2_eps * 0.5
            return -effectiveness

        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0=[total_epsilon_budget * 0.6, total_epsilon_budget * 0.4],
            bounds=[(0, total_epsilon_budget), (0, total_epsilon_budget)],
            constraints={'type': 'eq', 'fun': lambda x: x[0] + x[1] - total_epsilon_budget}
        )

        if result.success:
            return {'level1': result.x[0], 'level2': result.x[1]}
        else:
            return {'level1': self.level1_epsilon, 'level2': self.level2_epsilon}

    def get_privacy_loss_trajectory(self):
        return list(self.privacy_loss_history)

    def compute_privacy_amplification(self, sampling_rate, mechanism_privacy):
        if sampling_rate >= 1.0:
            return mechanism_privacy

        amplified_epsilon = mechanism_privacy['epsilon'] * sampling_rate

        return {
            'epsilon': amplified_epsilon,
            'delta': mechanism_privacy.get('delta', self.target_delta)
        }

    def validate_privacy_parameters(self, noise_multiplier, batch_size, dataset_size):
        if noise_multiplier <= 0:
            return False, "噪声乘子必须为正数"

        if batch_size <= 0 or dataset_size <= 0:
            return False, "批大小和数据集大小必须为正数"

        if batch_size > dataset_size:
            return False, "批大小不能大于数据集大小"

        if self.multi_level_perturbation:
            test_rdp = self.compute_multi_level_perturbation_cost(noise_multiplier, batch_size, dataset_size, 1)
        else:
            test_rdp = self.compute_rdp_cost(noise_multiplier, batch_size, dataset_size, 1)

        test_epsilon = self.rdp_to_dp(test_rdp)

        if test_epsilon + self.global_privacy_cost > self.target_epsilon:
            return False, f"会导致隐私预算超支，预计消耗: {test_epsilon:.4f}"

        return True, "参数有效"

    def reset(self):
        self.client_privacy_costs.clear()
        self.global_privacy_cost = 0.0
        self.privacy_loss_history.clear()

    def export_privacy_report(self):
        report = {
            'configuration': {
                'target_epsilon': self.target_epsilon,
                'target_delta': self.target_delta,
                'composition_method': self.composition_method,
                'alpha_orders': self.alpha_orders.tolist(),
                'multi_level_perturbation': self.multi_level_perturbation,
                'level1_epsilon': self.level1_epsilon if self.multi_level_perturbation else None,
                'level2_epsilon': self.level2_epsilon if self.multi_level_perturbation else None
            },
            'current_status': self.get_privacy_budget_status(),
            'trajectory': self.get_privacy_loss_trajectory(),
            'client_breakdown': {
                client_id: self.get_client_privacy_status(client_id)
                for client_id in self.client_privacy_costs.keys()
            }
        }

        return report