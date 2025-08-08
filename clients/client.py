import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from .dp_utils import clip_gradients, add_noise
from .noise_scheduler import NoiseScheduler

class FedClient:
    def __init__(self, client_id, model, train_loader, config):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.config = config

        self.grad_norms_history = deque(maxlen=config.window_size)
        self.clip_bounds = {}

        self.noise_scheduler = NoiseScheduler(config)
        self.layer_sensitivity = {}

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, global_round):
        """本地训练一轮"""
        self.model.train()
        total_loss = 0

        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                gradients = self._extract_gradients()
                self._update_clip_bounds(gradients)
                clipped_grads = self._apply_dynamic_clipping(gradients)

                # 层次差分隐私噪声
                noisy_grads = self._add_hierarchical_noise(clipped_grads, global_round)

                self._apply_gradients(noisy_grads)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _extract_gradients(self):
        """提取各层梯度"""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    def _update_clip_bounds(self, gradients):
        """动态更新剪切阈值"""
        # 计算当前梯度范数
        grad_norms = {}
        for name, grad in gradients.items():
            grad_norms[name] = torch.norm(grad).item()

        self.grad_norms_history.append(grad_norms)

        # 基于历史统计更新剪切阈值
        if len(self.grad_norms_history) >= self.config.min_history:
            for name in grad_norms.keys():
                norms = [h[name] for h in self.grad_norms_history if name in h]
                # 使用分位点作为动态剪切阈值
                self.clip_bounds[name] = np.quantile(norms, self.config.quantile_q)

    def _apply_dynamic_clipping(self, gradients):
        """应用动态剪切"""
        clipped_grads = {}
        for name, grad in gradients.items():
            if name in self.clip_bounds:
                clipped_grads[name] = clip_gradients(grad, self.clip_bounds[name])
            else:
                clipped_grads[name] = clip_gradients(grad, self.config.default_clip)
        return clipped_grads

    def _add_hierarchical_noise(self, gradients, global_round):
        """添加层次化噪声"""
        noisy_grads = {}
        for name, grad in gradients.items():
            # 获取当前层的噪声强度
            sigma = self.noise_scheduler.get_layer_noise(name, global_round)
            noisy_grads[name] = add_noise(grad, sigma)
        return noisy_grads

    def _apply_gradients(self, gradients):
        """将梯度应用到模型"""
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name]

    def get_model_weights(self):
        """获取模型权重"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_weights(self, weights):
        """设置模型权重"""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].clone()

    def evaluate(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total