import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import math


class ModelMetrics:
    """模型性能指标计算"""

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """计算Top-K准确率"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

    @staticmethod
    def cross_entropy_loss(output, target):
        """计算交叉熵损失"""
        criterion = nn.CrossEntropyLoss()
        return criterion(output, target)

    @staticmethod
    def f1_score(output, target, num_classes):
        """计算F1分数"""
        _, pred = torch.max(output, 1)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        cm = confusion_matrix(target, pred, labels=range(num_classes))

        # 计算每个类别的精确率和召回率
        precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-8)
        recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-8)

        # F1分数
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'macro_f1': np.mean(f1),
            'per_class_f1': f1,
            'precision': precision,
            'recall': recall
        }

    @staticmethod
    def confusion_matrix_metrics(output, target, num_classes):
        """计算混淆矩阵相关指标"""
        _, pred = torch.max(output, 1)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        cm = confusion_matrix(target, pred, labels=range(num_classes))

        return {
            'confusion_matrix': cm,
            'accuracy': np.trace(cm) / np.sum(cm),
            'per_class_accuracy': np.diag(cm) / np.sum(cm, axis=1)
        }


class PrivacyMetrics:
    """隐私相关指标计算"""

    @staticmethod
    def gradient_norm(gradients):
        """计算梯度范数"""
        total_norm = 0
        layer_norms = {}

        for name, grad in gradients.items():
            if grad is not None:
                layer_norm = torch.norm(grad).item()
                layer_norms[name] = layer_norm
                total_norm += layer_norm ** 2

        total_norm = math.sqrt(total_norm)

        return {
            'total_norm': total_norm,
            'layer_norms': layer_norms,
            'max_layer_norm': max(layer_norms.values()) if layer_norms else 0
        }

    @staticmethod
    def noise_to_signal_ratio(original_grad, noisy_grad):
        """计算噪声信号比"""
        noise = noisy_grad - original_grad
        signal_power = torch.norm(original_grad) ** 2
        noise_power = torch.norm(noise) ** 2

        if signal_power == 0:
            return float('inf')

        return (noise_power / signal_power).item()

    @staticmethod
    def privacy_budget_efficiency(accuracy, privacy_cost):
        """计算隐私预算效率"""
        if privacy_cost == 0:
            return float('inf')
        return accuracy / privacy_cost

    @staticmethod
    def gradient_similarity(grad1, grad2):
        """计算梯度相似度"""
        # 展平所有梯度
        flat_grad1 = torch.cat([g.flatten() for g in grad1.values()])
        flat_grad2 = torch.cat([g.flatten() for g in grad2.values()])

        # 余弦相似度
        cos_sim = F.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0))

        # 欧氏距离
        l2_dist = torch.norm(flat_grad1 - flat_grad2)

        return {
            'cosine_similarity': cos_sim.item(),
            'l2_distance': l2_dist.item()
        }


class AttackMetrics:
    """攻击相关指标计算"""

    @staticmethod
    def reconstruction_mse(original_data, reconstructed_data):
        """计算重建MSE"""
        mse = F.mse_loss(reconstructed_data, original_data, reduction='mean')
        return mse.item()

    @staticmethod
    def reconstruction_psnr(original_data, reconstructed_data):
        """计算重建PSNR"""
        mse = F.mse_loss(reconstructed_data, original_data, reduction='mean')
        if mse == 0:
            return float('inf')

        max_val = torch.max(original_data)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()

    @staticmethod
    def reconstruction_ssim(original_data, reconstructed_data):
        """计算重建SSIM"""
        mu1 = torch.mean(original_data)
        mu2 = torch.mean(reconstructed_data)

        sigma1_sq = torch.var(original_data)
        sigma2_sq = torch.var(reconstructed_data)
        sigma12 = torch.mean((original_data - mu1) * (reconstructed_data - mu2))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim.item()

    @staticmethod
    def label_accuracy(true_labels, predicted_labels):
        """计算标签预测准确率"""
        if len(true_labels) != len(predicted_labels):
            return 0.0

        correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
        return correct / len(true_labels)

    @staticmethod
    def attack_success_rate(attack_results, threshold=0.1):
        """计算攻击成功率"""
        if not attack_results:
            return 0.0

        successful_attacks = sum(1 for result in attack_results if result['mse'] < threshold)
        return successful_attacks / len(attack_results)

    @staticmethod
    def privacy_leakage_score(original_data, reconstructed_data, method='mse'):
        """计算隐私泄露分数"""
        if method == 'mse':
            return 1.0 - AttackMetrics.reconstruction_mse(original_data, reconstructed_data)
        elif method == 'psnr':
            psnr = AttackMetrics.reconstruction_psnr(original_data, reconstructed_data)
            return min(1.0, psnr / 50.0)  # 归一化到[0,1]
        elif method == 'ssim':
            return AttackMetrics.reconstruction_ssim(original_data, reconstructed_data)
        else:
            raise ValueError(f"Unknown method: {method}")


class FederatedMetrics:
    """联邦学习特有指标"""

    @staticmethod
    def client_drift(client_models, global_model):
        """计算客户端漂移"""
        drifts = {}

        global_params = {name: param.data for name, param in global_model.named_parameters()}

        for client_id, client_model in client_models.items():
            total_drift = 0
            layer_drifts = {}

            for name, param in client_model.named_parameters():
                if name in global_params:
                    layer_drift = torch.norm(param.data - global_params[name]).item()
                    layer_drifts[name] = layer_drift
                    total_drift += layer_drift ** 2

            drifts[client_id] = {
                'total_drift': math.sqrt(total_drift),
                'layer_drifts': layer_drifts
            }

        return drifts

    @staticmethod
    def aggregation_quality(client_updates, aggregated_update):
        """计算聚合质量"""
        # 计算聚合后的更新与各客户端更新的相似性
        similarities = []

        for client_update in client_updates.values():
            sim = PrivacyMetrics.gradient_similarity(client_update, aggregated_update)
            similarities.append(sim['cosine_similarity'])

        return {
            'avg_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_similarity': np.std(similarities)
        }

    @staticmethod
    def convergence_speed(accuracy_history, threshold=0.95):
        """计算收敛速度"""
        max_acc = max(accuracy_history)
        target_acc = max_acc * threshold

        for i, acc in enumerate(accuracy_history):
            if acc >= target_acc:
                return i + 1

        return len(accuracy_history)

    @staticmethod
    def stability_metrics(metric_history, window=10):
        """计算稳定性指标"""
        if len(metric_history) < window:
            return {'variance': 0, 'trend': 0}

        recent_values = metric_history[-window:]
        variance = np.var(recent_values)
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        return {
            'variance': variance,
            'trend': trend,
            'is_stable': variance < 0.01 and abs(trend) < 0.001
        }


class ComprehensiveEvaluator:
    """综合评估器"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model_metrics = ModelMetrics()
        self.privacy_metrics = PrivacyMetrics()
        self.attack_metrics = AttackMetrics()
        self.federated_metrics = FederatedMetrics()

    def evaluate_model(self, model, data_loader, device):
        """评估模型性能"""
        model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = self.model_metrics.cross_entropy_loss(output, target)
                total_loss += loss.item()

                all_outputs.append(output)
                all_targets.append(target)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算各项指标
        acc = self.model_metrics.accuracy(all_outputs, all_targets)[0]
        f1_metrics = self.model_metrics.f1_score(all_outputs, all_targets, self.num_classes)
        cm_metrics = self.model_metrics.confusion_matrix_metrics(all_outputs, all_targets, self.num_classes)

        return {
            'accuracy': acc.item(),
            'loss': total_loss / len(data_loader),
            'f1_score': f1_metrics['macro_f1'],
            'per_class_f1': f1_metrics['per_class_f1'],
            'confusion_matrix': cm_metrics['confusion_matrix'],
            'per_class_accuracy': cm_metrics['per_class_accuracy']
        }

    def evaluate_privacy(self, original_grads, noisy_grads, privacy_cost):
        """评估隐私保护效果"""
        # 梯度范数分析
        orig_norm = self.privacy_metrics.gradient_norm(original_grads)
        noisy_norm = self.privacy_metrics.gradient_norm(noisy_grads)

        # 噪声信号比
        nsr = self.privacy_metrics.noise_to_signal_ratio(
            torch.cat([g.flatten() for g in original_grads.values()]),
            torch.cat([g.flatten() for g in noisy_grads.values()])
        )

        # 梯度相似度
        similarity = self.privacy_metrics.gradient_similarity(original_grads, noisy_grads)

        return {
            'original_norm': orig_norm,
            'noisy_norm': noisy_norm,
            'noise_to_signal_ratio': nsr,
            'gradient_similarity': similarity,
            'privacy_cost': privacy_cost
        }

    def evaluate_attack_resistance(self, attack_results):
        """评估攻击抵抗能力"""
        if not attack_results:
            return {'success_rate': 0, 'avg_mse': 0, 'avg_psnr': float('inf')}

        mse_values = [r['mse'] for r in attack_results]
        psnr_values = [r.get('psnr', 0) for r in attack_results]

        success_rate = self.attack_metrics.attack_success_rate(attack_results)

        return {
            'success_rate': success_rate,
            'avg_mse': np.mean(mse_values),
            'std_mse': np.std(mse_values),
            'avg_psnr': np.mean(psnr_values),
            'min_mse': np.min(mse_values),
            'max_mse': np.max(mse_values)
        }

    def evaluate_federated_learning(self, client_models, global_model, accuracy_history):
        """评估联邦学习过程"""
        # 客户端漂移
        drift_metrics = self.federated_metrics.client_drift(client_models, global_model)

        # 收敛速度
        convergence_speed = self.federated_metrics.convergence_speed(accuracy_history)

        # 稳定性指标
        stability = self.federated_metrics.stability_metrics(accuracy_history)

        return {
            'client_drift': drift_metrics,
            'convergence_speed': convergence_speed,
            'stability': stability,
            'final_accuracy': accuracy_history[-1] if accuracy_history else 0,
            'max_accuracy': max(accuracy_history) if accuracy_history else 0
        }

    def comprehensive_evaluation(self, model, test_loader, device,
                                 privacy_data=None, attack_data=None,
                                 federated_data=None):
        """综合评估"""
        results = {}

        # 模型性能评估
        results['model_performance'] = self.evaluate_model(model, test_loader, device)

        # 隐私保护评估
        if privacy_data:
            results['privacy_protection'] = self.evaluate_privacy(**privacy_data)

        # 攻击抵抗评估
        if attack_data:
            results['attack_resistance'] = self.evaluate_attack_resistance(attack_data)

        # 联邦学习评估
        if federated_data:
            results['federated_learning'] = self.evaluate_federated_learning(**federated_data)

        return results


def compute_dataset_stats(data_loader):
    """计算数据集统计信息"""
    all_data = []
    all_labels = []

    for data, labels in data_loader:
        all_data.append(data)
        all_labels.append(labels)

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        'mean': torch.mean(all_data, dim=[0, 2, 3]),
        'std': torch.std(all_data, dim=[0, 2, 3]),
        'min': torch.min(all_data),
        'max': torch.max(all_data),
        'shape': all_data.shape,
        'num_samples': len(all_data),
        'num_classes': len(torch.unique(all_labels)),
        'class_distribution': torch.bincount(all_labels)
    }


def compare_experiments(results_list, metric_names=None):
    """比较多个实验结果"""
    if metric_names is None:
        metric_names = ['accuracy', 'f1_score', 'privacy_cost', 'attack_success_rate']

    comparison = {}

    for metric in metric_names:
        values = []
        for results in results_list:
            value = results.get(metric, 0)
            values.append(value)

        comparison[metric] = {
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    return comparison