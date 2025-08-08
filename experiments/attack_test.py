import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
import copy
from .metrics import AttackMetrics


class DLGAttack:
    """Deep Leakage from Gradients 攻击"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # 攻击参数
        self.max_iterations = config.dlg_iterations if hasattr(config, 'dlg_iterations') else 1000
        self.lr = config.dlg_lr if hasattr(config, 'dlg_lr') else 1.0
        self.tv_weight = config.dlg_tv_weight if hasattr(config, 'dlg_tv_weight') else 1e-4

        self.attack_metrics = AttackMetrics()

    def attack(self, target_gradients, original_data=None, original_labels=None, batch_size=1):
        """执行DLG攻击"""
        dummy_data = torch.randn(batch_size, *self.model.conv1.weight.shape[1:]).to(self.device)
        dummy_data.requires_grad = True

        dummy_labels = torch.randint(0, self.config.num_classes, (batch_size,)).to(self.device)
        dummy_labels.requires_grad = False

        # 优化器
        optimizer = torch.optim.LBFGS([dummy_data], lr=self.lr)

        # 攻击结果记录
        best_loss = float('inf')
        best_data = None
        best_labels = None

        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()

            # 计算虚拟梯度
            dummy_output = self.model(dummy_data)
            dummy_loss = criterion(dummy_output, dummy_labels)

            dummy_gradients = torch.autograd.grad(
                dummy_loss, self.model.parameters(), create_graph=True
            )

            # 计算梯度匹配损失
            grad_loss = 0
            for target_grad, dummy_grad in zip(target_gradients, dummy_gradients):
                grad_loss += F.mse_loss(dummy_grad, target_grad)

            # 总变分正则化
            tv_loss = self._total_variation_loss(dummy_data)

            total_loss = grad_loss + self.tv_weight * tv_loss
            total_loss.backward()

            return total_loss

        # 迭代优化
        for iteration in range(self.max_iterations):
            loss = optimizer.step(closure)

            if loss < best_loss:
                best_loss = loss.item()
                best_data = dummy_data.detach().clone()
                best_labels = dummy_labels.clone()

            # 早停机制
            if iteration % 100 == 0 and loss < 1e-4:
                break

        # 计算攻击结果
        attack_result = self._evaluate_attack_result(
            best_data, best_labels, original_data, original_labels
        )

        return attack_result

    def _total_variation_loss(self, data):
        """计算总变分损失"""
        tv_loss = 0
        for i in range(data.shape[0]):
            img = data[i]
            tv_loss += torch.sum(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + \
                       torch.sum(torch.abs(img[:, :-1, :] - img[:, 1:, :]))
        return tv_loss

    def _evaluate_attack_result(self, reconstructed_data, reconstructed_labels,
                                original_data, original_labels):
        """评估攻击结果"""
        result = {
            'reconstructed_data': reconstructed_data,
            'reconstructed_labels': reconstructed_labels,
            'success': False
        }

        if original_data is not None:
            # 计算重建误差
            mse = self.attack_metrics.reconstruction_mse(original_data, reconstructed_data)
            psnr = self.attack_metrics.reconstruction_psnr(original_data, reconstructed_data)
            ssim = self.attack_metrics.reconstruction_ssim(original_data, reconstructed_data)

            result.update({
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim,
                'success': mse < 0.1  # 阈值可调
            })

        if original_labels is not None:
            # 计算标签准确率
            label_acc = self.attack_metrics.label_accuracy(
                original_labels.cpu().numpy(),
                reconstructed_labels.cpu().numpy()
            )
            result['label_accuracy'] = label_acc

        return result

    def batch_attack(self, gradient_batches, original_data_batches=None,
                     original_label_batches=None):
        """批量攻击"""
        attack_results = []

        for i, gradients in enumerate(gradient_batches):
            orig_data = original_data_batches[i] if original_data_batches else None
            orig_labels = original_label_batches[i] if original_label_batches else None

            result = self.attack(gradients, orig_data, orig_labels)
            attack_results.append(result)

        return attack_results


class PropertyInferenceAttack:
    """属性推断攻击"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # 攻击者模型
        self.shadow_model = copy.deepcopy(model)
        self.attack_model = self._create_attack_model()

        self.attack_metrics = AttackMetrics()

    def _create_attack_model(self):
        """创建攻击模型"""
        # 简单的二分类器用于推断属性
        feature_dim = self.config.num_classes  # 使用输出向量作为特征

        attack_model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 二分类：有/无特定属性
        ).to(self.device)

        return attack_model

    def train_attack_model(self, shadow_data, shadow_labels, property_labels):
        """训练攻击模型"""
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练影子模型
        self._train_shadow_model(shadow_data, shadow_labels)

        # 生成攻击训练数据
        attack_features = []
        attack_targets = []

        self.shadow_model.eval()
        with torch.no_grad():
            for data, prop_label in zip(shadow_data, property_labels):
                output = self.shadow_model(data.unsqueeze(0))
                attack_features.append(output.squeeze())
                attack_targets.append(prop_label)

        attack_features = torch.stack(attack_features)
        attack_targets = torch.tensor(attack_targets, dtype=torch.long).to(self.device)

        # 训练攻击模型
        self.attack_model.train()
        for epoch in range(100):
            optimizer.zero_grad()

            predictions = self.attack_model(attack_features)
            loss = criterion(predictions, attack_targets)

            loss.backward()
            optimizer.step()

    def _train_shadow_model(self, shadow_data, shadow_labels):
        """训练影子模型"""
        optimizer = torch.optim.SGD(self.shadow_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        self.shadow_model.train()
        for epoch in range(50):
            for i in range(0, len(shadow_data), 32):
                batch_data = shadow_data[i:i + 32]
                batch_labels = shadow_labels[i:i + 32]

                optimizer.zero_grad()
                outputs = self.shadow_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

    def infer_property(self, target_data):
        """推断目标数据的属性"""
        self.model.eval()
        self.attack_model.eval()

        with torch.no_grad():
            # 获取目标模型输出
            target_output = self.model(target_data)

            # 使用攻击模型预测属性
            property_prediction = self.attack_model(target_output)
            property_prob = F.softmax(property_prediction, dim=1)

        return {
            'property_prediction': property_prediction,
            'property_probability': property_prob,
            'confidence': torch.max(property_prob, dim=1)[0]
        }

    def evaluate_attack(self, test_data, true_properties):
        """评估属性推断攻击"""
        correct = 0
        total = len(test_data)
        confidences = []

        for i, data in enumerate(test_data):
            result = self.infer_property(data.unsqueeze(0))
            predicted = torch.argmax(result['property_prediction'], dim=1).item()

            if predicted == true_properties[i]:
                correct += 1

            confidences.append(result['confidence'].item())

        return {
            'accuracy': correct / total,
            'avg_confidence': np.mean(confidences),
            'attack_success_rate': correct / total
        }


class MembershipInferenceAttack:
    """成员推断攻击"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # 攻击模型
        self.attack_model = self._create_attack_model()

    def _create_attack_model(self):
        """创建成员推断攻击模型"""
        return nn.Sequential(
            nn.Linear(self.config.num_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # 二分类：成员/非成员
        ).to(self.device)

    def train_attack_model(self, member_data, member_labels, non_member_data, non_member_labels):
        """训练成员推断攻击模型"""
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 生成训练数据
        member_outputs = []
        non_member_outputs = []

        self.model.eval()
        with torch.no_grad():
            # 成员数据的模型输出
            for data in member_data:
                output = self.model(data.unsqueeze(0))
                member_outputs.append(F.softmax(output, dim=1).squeeze())

            # 非成员数据的模型输出
            for data in non_member_data:
                output = self.model(data.unsqueeze(0))
                non_member_outputs.append(F.softmax(output, dim=1).squeeze())

        # 构造训练数据
        train_features = torch.stack(member_outputs + non_member_outputs)
        train_labels = torch.tensor([1] * len(member_outputs) + [0] * len(non_member_outputs)).to(self.device)

        # 训练攻击模型
        self.attack_model.train()
        for epoch in range(100):
            optimizer.zero_grad()

            predictions = self.attack_model(train_features)
            loss = criterion(predictions, train_labels)

            loss.backward()
            optimizer.step()

    def infer_membership(self, test_data):
        """推断成员身份"""
        self.model.eval()
        self.attack_model.eval()

        with torch.no_grad():
            # 获取模型输出
            model_output = self.model(test_data)
            model_prob = F.softmax(model_output, dim=1)

            # 攻击模型预测
            membership_pred = self.attack_model(model_prob)
            membership_prob = F.softmax(membership_pred, dim=1)

        return {
            'membership_prediction': membership_pred,
            'membership_probability': membership_prob,
            'is_member': torch.argmax(membership_prob, dim=1)
        }


class AttackTestSuite:
    """攻击测试套件"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # 初始化各种攻击
        self.dlg_attack = DLGAttack(model, device, config)
        self.property_attack = PropertyInferenceAttack(model, device, config)
        self.membership_attack = MembershipInferenceAttack(model, device, config)

        self.attack_metrics = AttackMetrics()

    def run_dlg_attack(self, gradients, original_data=None, original_labels=None):
        """运行DLG攻击测试"""
        print("Running DLG Attack...")

        # 单次攻击
        result = self.dlg_attack.attack(gradients, original_data, original_labels)

        # 批量攻击测试
        if isinstance(gradients, list):
            batch_results = self.dlg_attack.batch_attack(
                gradients, original_data, original_labels
            )

            # 计算攻击统计
            success_count = sum(1 for r in batch_results if r['success'])
            avg_mse = np.mean([r['mse'] for r in batch_results if 'mse' in r])

            result.update({
                'batch_results': batch_results,
                'batch_success_rate': success_count / len(batch_results),
                'batch_avg_mse': avg_mse
            })

        print(f"DLG Attack completed. Success: {result.get('success', False)}")
        return result

    def run_property_inference_attack(self, target_data, property_labels,
                                      shadow_data=None, shadow_labels=None):
        """运行属性推断攻击测试"""
        print("Running Property Inference Attack...")

        if shadow_data is not None and shadow_labels is not None:
            # 训练攻击模型
            self.property_attack.train_attack_model(
                shadow_data, shadow_labels, property_labels[:len(shadow_data)]
            )

        # 执行攻击
        results = []
        for data in target_data:
            result = self.property_attack.infer_property(data.unsqueeze(0))
            results.append(result)

        # 评估攻击效果
        if len(property_labels) >= len(target_data):
            evaluation = self.property_attack.evaluate_attack(
                target_data, property_labels[:len(target_data)]
            )

            print(f"Property Inference Attack - Accuracy: {evaluation['accuracy']:.3f}")
            return {
                'individual_results': results,
                'evaluation': evaluation
            }

        return {'individual_results': results}

    def run_membership_inference_attack(self, test_data, member_data, non_member_data):
        """运行成员推断攻击测试"""
        print("Running Membership Inference Attack...")

        # 训练攻击模型
        self.membership_attack.train_attack_model(
            member_data, None, non_member_data, None
        )

        # 执行攻击
        results = []
        for data in test_data:
            result = self.membership_attack.infer_membership(data.unsqueeze(0))
            results.append(result)

        print("Membership Inference Attack completed.")
        return results

    def comprehensive_attack_test(self, test_data, **kwargs):
        """综合攻击测试"""
        print("Starting Comprehensive Attack Test...")

        attack_results = {}

        # DLG攻击
        if 'gradients' in kwargs:
            attack_results['dlg'] = self.run_dlg_attack(
                kwargs['gradients'],
                kwargs.get('original_data'),
                kwargs.get('original_labels')
            )

        # 属性推断攻击
        if 'property_labels' in kwargs:
            attack_results['property_inference'] = self.run_property_inference_attack(
                test_data,
                kwargs['property_labels'],
                kwargs.get('shadow_data'),
                kwargs.get('shadow_labels')
            )

        # 成员推断攻击
        if 'member_data' in kwargs and 'non_member_data' in kwargs:
            attack_results['membership_inference'] = self.run_membership_inference_attack(
                test_data,
                kwargs['member_data'],
                kwargs['non_member_data']
            )

        print("Comprehensive Attack Test completed.")
        return attack_results

    def evaluate_defense_effectiveness(self, clean_results, defended_results):
        """评估防御效果"""
        defense_metrics = {}

        # 比较DLG攻击效果
        if 'dlg' in clean_results and 'dlg' in defended_results:
            clean_mse = clean_results['dlg'].get('mse', 0)
            defended_mse = defended_results['dlg'].get('mse', 0)

            defense_metrics['dlg_protection'] = {
                'mse_increase': defended_mse - clean_mse,
                'success_rate_decrease':
                    clean_results['dlg'].get('success', 0) - defended_results['dlg'].get('success', 0)
            }

        # 比较属性推断攻击效果
        if 'property_inference' in clean_results and 'property_inference' in defended_results:
            clean_acc = clean_results['property_inference']['evaluation']['accuracy']
            defended_acc = defended_results['property_inference']['evaluation']['accuracy']

            defense_metrics['property_protection'] = {
                'accuracy_decrease': clean_acc - defended_acc
            }

        return defense_metrics


def run_attack_experiments(model, device, config, test_data, **attack_data):
    """运行攻击实验的便捷函数"""
    attack_suite = AttackTestSuite(model, device, config)

    # 运行综合攻击测试
    results = attack_suite.comprehensive_attack_test(test_data, **attack_data)

    # 生成攻击报告
    report = {
        'attack_results': results,
        'summary': {
            'total_attacks': len(results),
            'successful_attacks': sum(1 for r in results.values() if r.get('success', False)),
            'avg_attack_success_rate': np.mean([
                r.get('success', 0) for r in results.values() if 'success' in r
            ])
        }
    }

    return report