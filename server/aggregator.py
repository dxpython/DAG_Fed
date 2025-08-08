import torch
import numpy as np
from collections import defaultdict, OrderedDict
from .privacy_accountant import PrivacyAccountant


class FedAggregator:
    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config
        self.current_round = 0

        # 隐私会计
        self.privacy_accountant = PrivacyAccountant(config)

        # 聚合历史
        self.aggregation_history = []
        self.client_weights_history = defaultdict(list)

        # 性能跟踪
        self.global_performance = {
            'accuracy': [],
            'loss': [],
            'privacy_budget': []
        }

    def aggregate(self, client_updates, client_weights=None):
        """聚合客户端更新"""
        if client_weights is None:
            client_weights = self._compute_default_weights(client_updates)

        total_weight = sum(client_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            client_weights = {k: v / total_weight for k, v in client_weights.items()}

        # 执行聚合
        if self.config.aggregation_method == 'fedavg':
            aggregated_params = self._fedavg_aggregate(client_updates, client_weights)
        elif self.config.aggregation_method == 'scaffold':
            aggregated_params = self._scaffold_aggregate(client_updates, client_weights)
        elif self.config.aggregation_method == 'adaptive':
            aggregated_params = self._adaptive_aggregate(client_updates, client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")

        # 更新全局模型
        self._update_global_model(aggregated_params)

        self._record_aggregation(client_updates, client_weights)

        self.current_round += 1

        return self.get_global_model()

    def _compute_default_weights(self, client_updates):
        """计算默认客户端权重"""
        if self.config.weighting_method == 'uniform':
            num_clients = len(client_updates)
            return {cid: 1.0 / num_clients for cid in client_updates.keys()}

        elif self.config.weighting_method == 'data_size':
            total_samples = sum(update.get('data_size', 1) for update in client_updates.values())
            return {cid: update.get('data_size', 1) / total_samples
                    for cid, update in client_updates.items()}

        elif self.config.weighting_method == 'loss_based':
            losses = {cid: update.get('loss', 1.0) for cid, update in client_updates.items()}
            inv_losses = {cid: 1.0 / loss for cid, loss in losses.items()}
            total_inv_loss = sum(inv_losses.values())
            return {cid: inv_loss / total_inv_loss for cid, inv_loss in inv_losses.items()}

        else:
            num_clients = len(client_updates)
            return {cid: 1.0 / num_clients for cid in client_updates.keys()}

    def _fedavg_aggregate(self, client_updates, client_weights):
        """FedAvg聚合"""
        aggregated_params = OrderedDict()

        # 初始化聚合参数
        first_client = next(iter(client_updates.values()))
        for name in first_client['params'].keys():
            aggregated_params[name] = torch.zeros_like(first_client['params'][name])

        # 加权聚合
        for client_id, update in client_updates.items():
            weight = client_weights[client_id]
            for name, param in update['params'].items():
                aggregated_params[name] += weight * param

        return aggregated_params

    def _scaffold_aggregate(self, client_updates, client_weights):
        """Scaffold聚合"""
        aggregated_params = OrderedDict()
        aggregated_controls = OrderedDict()

        # 获取当前全局模型参数
        global_params = {name: param.clone() for name, param in self.global_model.named_parameters()}

        # 初始化
        first_client = next(iter(client_updates.values()))
        for name in first_client['params'].keys():
            aggregated_params[name] = torch.zeros_like(first_client['params'][name])
            aggregated_controls[name] = torch.zeros_like(first_client['params'][name])

        # 聚合参数和控制变量
        # 真难呀，这里
        for client_id, update in client_updates.items():
            weight = client_weights[client_id]

            # 聚合模型参数
            for name, param in update['params'].items():
                aggregated_params[name] += weight * param

            # 聚合控制变量
            if 'controls' in update:
                for name, control in update['controls'].items():
                    aggregated_controls[name] += weight * control

        if hasattr(self, 'global_controls'):
            for name in aggregated_controls.keys():
                self.global_controls[name] = aggregated_controls[name]
        else:
            self.global_controls = aggregated_controls

        return aggregated_params

    def _adaptive_aggregate(self, client_updates, client_weights):
        """自适应聚合"""
        # 计算客户端更新的相似性
        similarities = self._compute_client_similarities(client_updates)

        # 调整权重
        adjusted_weights = self._adjust_weights_by_similarity(client_weights, similarities)

        # 执行FedAvg聚合
        return self._fedavg_aggregate(client_updates, adjusted_weights)

    def _compute_client_similarities(self, client_updates):
        """计算客户端更新相似性"""
        similarities = {}
        client_ids = list(client_updates.keys())

        for i, client_i in enumerate(client_ids):
            similarities[client_i] = {}
            for j, client_j in enumerate(client_ids):
                if i != j:
                    sim = self._cosine_similarity(
                        client_updates[client_i]['params'],
                        client_updates[client_j]['params']
                    )
                    similarities[client_i][client_j] = sim
                else:
                    similarities[client_i][client_j] = 1.0

        return similarities

    def _cosine_similarity(self, params1, params2):
        """计算参数向量的余弦相似度"""
        dot_product = 0
        norm1 = 0
        norm2 = 0

        for name in params1.keys():
            if name in params2:
                p1_flat = params1[name].flatten()
                p2_flat = params2[name].flatten()

                dot_product += torch.dot(p1_flat, p2_flat)
                norm1 += torch.norm(p1_flat) ** 2
                norm2 += torch.norm(p2_flat) ** 2

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (torch.sqrt(norm1) * torch.sqrt(norm2))

    def _adjust_weights_by_similarity(self, client_weights, similarities):
        """根据相似性调整权重"""
        adjusted_weights = {}

        for client_id, weight in client_weights.items():
            avg_similarity = np.mean(list(similarities[client_id].values()))

            diversity_factor = 1.0 - avg_similarity
            adjusted_weights[client_id] = weight * (1 + diversity_factor)

        # 重新归一化
        total_weight = sum(adjusted_weights.values())
        return {k: v / total_weight for k, v in adjusted_weights.items()}

    def _update_global_model(self, aggregated_params):
        """更新全局模型参数"""
        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                param.data = aggregated_params[name]

    def _record_aggregation(self, client_updates, client_weights):
        """记录聚合历史"""
        round_info = {
            'round': self.current_round,
            'num_clients': len(client_updates),
            'client_weights': client_weights.copy(),
            'timestamp': torch.tensor(self.current_round) 
        }

        self.aggregation_history.append(round_info)

        for client_id, weight in client_weights.items():
            self.client_weights_history[client_id].append(weight)

    def evaluate_global_model(self, test_loader):
        """评估全局模型性能"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        # 记录性能
        self.global_performance['accuracy'].append(accuracy)
        self.global_performance['loss'].append(avg_loss)

        # 记录隐私预算状态
        privacy_status = self.privacy_accountant.get_privacy_budget_status()
        self.global_performance['privacy_budget'].append(privacy_status['consumed'])

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'privacy_budget': privacy_status
        }

    def get_global_model(self):
        """获取全局模型权重"""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}

    def update_privacy_budget(self, client_privacy_costs):
        """更新隐私预算"""
        for client_id, cost in client_privacy_costs.items():
            self.privacy_accountant.add_privacy_cost(client_id, cost)

    def get_aggregation_stats(self):
        """获取聚合统计信息"""
        if not self.aggregation_history:
            return {}

        stats = {
            'total_rounds': len(self.aggregation_history),
            'avg_clients_per_round': np.mean([h['num_clients'] for h in self.aggregation_history]),
            'client_participation': dict(self.client_weights_history),
            'performance_trajectory': self.global_performance
        }

        return stats

    def check_privacy_budget(self):
        """检查隐私预算状态"""
        return self.privacy_accountant.check_budget_exhausted()

    def reset_aggregator(self):
        """重置聚合器状态"""
        self.current_round = 0
        self.aggregation_history.clear()
        self.client_weights_history.clear()
        self.global_performance = {
            'accuracy': [],
            'loss': [],
            'privacy_budget': []
        }
        self.privacy_accountant.reset()

        if hasattr(self, 'global_controls'):
            delattr(self, 'global_controls')