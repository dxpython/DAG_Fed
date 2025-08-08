import torch
import numpy as np
from collections import defaultdict, deque
from .dp_utils import layer_importance_score, privacy_budget_allocation


class NoiseScheduler:
    def __init__(self, config):
        self.config = config
        self.current_round = 0

        # 层次化噪声管理
        self.layer_noise_levels = {}
        self.layer_importance_history = defaultdict(deque)
        self.base_noise_levels = {}

        # 动态调度相关
        self.performance_history = deque(maxlen=config.perf_window_size)
        self.noise_adjustment_factor = 1.0

        # 隐私预算追踪
        self.privacy_budget_consumed = defaultdict(float)
        self.total_privacy_budget = config.total_epsilon

    def initialize_layer_noise(self, model_layers):
        """初始化各层噪声水平"""
        num_layers = len(model_layers)
        base_sigma = self.config.base_noise_multiplier

        for i, layer_name in enumerate(model_layers):
            depth_factor = 1.0 - (i / num_layers) * 0.3
            self.base_noise_levels[layer_name] = base_sigma * depth_factor
            self.layer_noise_levels[layer_name] = self.base_noise_levels[layer_name]

    def update_layer_importance(self, gradients):
        """更新层重要性评估"""
        importance_scores = layer_importance_score(gradients, method=self.config.importance_method)

        for layer_name, score in importance_scores.items():
            self.layer_importance_history[layer_name].append(score.item())

            if len(self.layer_importance_history[layer_name]) > self.config.importance_window:
                self.layer_importance_history[layer_name].popleft()

    def get_layer_noise(self, layer_name, global_round):
        """获取指定层的噪声水平"""
        if layer_name not in self.layer_noise_levels:
            return self.config.default_noise_multiplier

        base_noise = self.layer_noise_levels[layer_name]

        # 动态调整因子
        adjustment = self._compute_dynamic_adjustment(layer_name, global_round)

        # 轮次衰减
        round_decay = self._compute_round_decay(global_round)

        final_noise = base_noise * adjustment * round_decay
        return max(final_noise, self.config.min_noise_multiplier)

    def _compute_dynamic_adjustment(self, layer_name, global_round):
        """计算动态调整因子"""
        if layer_name not in self.layer_importance_history:
            return 1.0

        importance_history = list(self.layer_importance_history[layer_name])
        if len(importance_history) < 2:
            return 1.0

        recent_importance = np.mean(importance_history[-3:])
        historical_importance = np.mean(importance_history)

        if recent_importance > historical_importance * 1.1:
            return 0.9
        elif recent_importance < historical_importance * 0.9:
            return 1.1
        else:
            return 1.0

    def _compute_round_decay(self, global_round):
        """计算轮次衰减因子"""
        if self.config.noise_decay_method == 'exponential':
            return np.exp(-self.config.decay_rate * global_round)
        elif self.config.noise_decay_method == 'linear':
            return max(0.1, 1.0 - self.config.decay_rate * global_round)
        elif self.config.noise_decay_method == 'cosine':
            return 0.5 * (1 + np.cos(np.pi * global_round / self.config.total_rounds))
        else:
            return 1.0

    def update_noise_schedule(self, accuracy, loss, global_round):
        """基于性能反馈更新噪声调度"""
        self.current_round = global_round

        # 记录性能历史
        self.performance_history.append({
            'accuracy': accuracy,
            'loss': loss,
            'round': global_round
        })

        if len(self.performance_history) >= 2:
            # 计算性能变化
            prev_perf = self.performance_history[-2]
            curr_perf = self.performance_history[-1]

            acc_change = curr_perf['accuracy'] - prev_perf['accuracy']
            loss_change = curr_perf['loss'] - prev_perf['loss']

            self._adaptive_noise_adjustment(acc_change, loss_change)

    def _adaptive_noise_adjustment(self, acc_change, loss_change):
        """自适应噪声调整"""
        if acc_change < -0.05 or loss_change > 0.1:
            self.noise_adjustment_factor *= 0.95
        elif acc_change > 0.02 and loss_change < -0.05:
            self.noise_adjustment_factor *= 1.02

        self.noise_adjustment_factor = np.clip(self.noise_adjustment_factor, 0.5, 2.0)

    def allocate_privacy_budget(self, gradients):
        """分配隐私预算"""
        # 计算层重要性
        importance_scores = layer_importance_score(gradients)

        remaining_budget = self.total_privacy_budget - sum(self.privacy_budget_consumed.values())

        if remaining_budget <= 0:
            return {name: 0 for name in importance_scores.keys()}

        budget_allocation = privacy_budget_allocation(
            importance_scores,
            remaining_budget * self.config.budget_allocation_rate,
            method=self.config.budget_allocation_method
        )

        for layer_name, budget in budget_allocation.items():
            self.privacy_budget_consumed[layer_name] += budget

        return budget_allocation

    def get_privacy_budget_status(self):
        """获取隐私预算状态"""
        total_consumed = sum(self.privacy_budget_consumed.values())
        remaining = self.total_privacy_budget - total_consumed

        return {
            'total_budget': self.total_privacy_budget,
            'consumed': total_consumed,
            'remaining': remaining,
            'utilization': total_consumed / self.total_privacy_budget,
            'layer_consumption': dict(self.privacy_budget_consumed)
        }

    def reset_scheduler(self):
        """重置调度器状态"""
        self.current_round = 0
        self.layer_importance_history.clear()
        self.performance_history.clear()
        self.privacy_budget_consumed.clear()
        self.noise_adjustment_factor = 1.0

        # 重置噪声水平到基础值
        for layer_name in self.layer_noise_levels:
            self.layer_noise_levels[layer_name] = self.base_noise_levels[layer_name]