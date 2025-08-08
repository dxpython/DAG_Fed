import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from config import Config, QuickConfigs
from clients.client import FedClient
from server.aggregator import FedAggregator
from server.privacy_accountant import PrivacyAccountant
from models.model_cnn import create_model
from data.data_loader import create_federated_dataloaders
from experiments.logger import ExperimentLogger
from experiments.metrics import ComprehensiveEvaluator
from experiments.attack_test import AttackTestSuite
from utils.helper import *
from utils.plots import ExperimentPlotter


class DAGCHDPSystem:
    """DAGC-HDP联邦学习系统"""

    def __init__(self, config):
        self.config = config
        self.device = get_device()

        # 设置随机种子
        set_seed(config.seed)

        # 初始化组件
        self.global_model = None
        self.clients = {}
        self.aggregator = None
        self.privacy_accountant = None

        # 数据加载器
        self.client_loaders = {}
        self.test_loader = None

        # 实验工具
        self.logger = None
        self.evaluator = None
        self.attack_suite = None
        self.plotter = None

        # 训练状态
        self.current_round = 0
        self.training_metrics = defaultdict(list)
        self.client_metrics = defaultdict(dict)

        print(f"DAGC-HDP System initialized on {self.device}")
        config.print_config()

    def initialize_system(self):
        """初始化系统组件"""
        print("Initializing DAGC-HDP system...")

        # 创建实验目录
        self.experiment_dir = create_experiment_dir(
            self.config.output_dir,
            self.config.get_experiment_id()
        )

        # 初始化日志器
        self.logger = ExperimentLogger(
            self.config.get_experiment_id(),
            self.config,
            log_dir=self.experiment_dir
        )

        # 初始化全局模型
        self.global_model = create_model(
            self.config.dataset_name,
            self.config.model_type,
            self.config.num_classes
        ).to(self.device)

        model_info = get_model_size(self.global_model)
        self.logger.log(f"Model created: {model_info}")

        # 加载数据
        self.load_data()

        # 初始化客户端
        self.initialize_clients()

        # 初始化服务器
        self.initialize_server()

        # 初始化评估工具
        self.evaluator = ComprehensiveEvaluator(self.config.num_classes)

        # 初始化攻击测试
        if self.config.enable_attack_test:
            self.attack_suite = AttackTestSuite(self.global_model, self.device, self.config)

        # 初始化可视化
        self.plotter = ExperimentPlotter(
            save_dir=os.path.join(self.experiment_dir, 'plots'),
            figsize=self.config.plot_figsize,
            dpi=self.config.plot_dpi
        )

        self.logger.log("System initialization completed")

    def load_data(self):
        """加载数据"""
        self.logger.log("Loading federated datasets...")

        self.client_loaders, self.test_loader, dataset_stats = create_federated_dataloaders(self.config)

        self.logger.log(f"Loaded {len(self.client_loaders)} client datasets")
        self.logger.log(f"Dataset stats: {dataset_stats}")

        # 保存数据集统计
        save_results(dataset_stats, os.path.join(self.experiment_dir, 'dataset_stats.json'))

    def initialize_clients(self):
        """初始化客户端"""
        self.logger.log("Initializing federated clients...")

        for client_id, client_loader in self.client_loaders.items():
            # 创建客户端模型
            client_model = create_model(
                self.config.dataset_name,
                self.config.model_type,
                self.config.num_classes
            ).to(self.device)

            # 初始化客户端
            client = FedClient(client_id, client_model, client_loader, self.config)
            self.clients[client_id] = client

        self.logger.log(f"Initialized {len(self.clients)} clients")

    def initialize_server(self):
        """初始化服务器"""
        self.logger.log("Initializing federated server...")

        # 聚合器
        self.aggregator = FedAggregator(self.global_model, self.config)

        # 隐私会计
        self.privacy_accountant = PrivacyAccountant(self.config)

        self.logger.log("Server initialized")

    def select_clients(self, round_num):
        """选择参与训练的客户端"""
        if self.config.client_selection_method == "random":
            selected_clients = random.sample(
                list(self.clients.keys()),
                self.config.num_selected_clients
            )
        elif self.config.client_selection_method == "loss_based":
            # 基于损失选择（简化实现）
            client_losses = {cid: self.client_metrics.get(cid, {}).get('loss', 1.0)
                             for cid in self.clients.keys()}
            sorted_clients = sorted(client_losses.items(), key=lambda x: x[1], reverse=True)
            selected_clients = [cid for cid, _ in sorted_clients[:self.config.num_selected_clients]]
        else:
            # 默认随机选择
            selected_clients = random.sample(
                list(self.clients.keys()),
                self.config.num_selected_clients
            )

        return selected_clients

    def train_round(self, round_num):
        """执行一轮训练"""
        self.logger.log_round_start(round_num, "Starting training round")

        # 选择客户端
        selected_clients = self.select_clients(round_num)
        self.logger.log_round_start(round_num, selected_clients)

        # 分发全局模型
        global_weights = self.aggregator.get_global_model()

        # 客户端训练
        client_updates = {}
        client_privacy_costs = {}

        for client_id in selected_clients:
            client = self.clients[client_id]

            # 设置全局模型权重
            client.set_model_weights(global_weights)

            # 本地训练
            train_loss = client.local_train(round_num)

            # 获取更新
            updated_weights = client.get_model_weights()

            # 计算隐私成本
            if self.config.use_dp:
                privacy_cost = self._compute_privacy_cost(client_id, round_num)
                client_privacy_costs[client_id] = privacy_cost
            else:
                client_privacy_costs[client_id] = 0

            client_updates[client_id] = {
                'params': updated_weights,
                'loss': train_loss,
                'data_size': len(client.train_loader.dataset)
            }

            # 记录客户端指标
            self.client_metrics[client_id] = {
                'loss': train_loss,
                'privacy_cost': client_privacy_costs[client_id]
            }

            self.logger.log_client_metrics(client_id, round_num, self.client_metrics[client_id])

        # 服务器聚合
        aggregated_model = self.aggregator.aggregate(client_updates)

        # 更新隐私预算
        if self.config.use_dp:
            self.aggregator.update_privacy_budget(client_privacy_costs)

        # 评估全局模型
        eval_results = self.aggregator.evaluate_global_model(self.test_loader)

        # 更新噪声调度
        if self.config.use_adaptive_scheduling:
            self._update_noise_schedule(round_num, eval_results)

        # 记录轮次指标
        self.training_metrics['accuracy'].append(eval_results['accuracy'])
        self.training_metrics['loss'].append(eval_results['loss'])
        self.training_metrics['privacy_budget'].append(
            eval_results['privacy_budget']['consumed']
        )

        self.logger.log_round_end(round_num, eval_results)

        return eval_results

    def _compute_privacy_cost(self, client_id, round_num):
        """计算隐私成本"""
        client = self.clients[client_id]

        # 获取噪声参数
        noise_multipliers = {}
        for name, param in client.model.named_parameters():
            noise_multipliers[name] = client.noise_scheduler.get_layer_noise(name, round_num)

        # 计算RDP成本
        batch_size = self.config.batch_size
        dataset_size = len(client.train_loader.dataset)
        steps = self.config.local_epochs * len(client.train_loader)

        rdp_cost = self.privacy_accountant.compute_layered_rdp_cost(
            noise_multipliers, batch_size, dataset_size, steps
        )

        return rdp_cost

    def _update_noise_schedule(self, round_num, eval_results):
        """更新噪声调度"""
        accuracy = eval_results['accuracy']
        loss = eval_results['loss']

        # 更新所有客户端的噪声调度
        for client in self.clients.values():
            client.noise_scheduler.update_noise_schedule(accuracy, loss, round_num)

    def run_attack_test(self, round_num):
        """运行攻击测试"""
        if not self.config.enable_attack_test:
            return

        if round_num % self.config.attack_test_interval != 0:
            return

        self.logger.log(f"Running attack test at round {round_num}")

        # 准备攻击数据
        test_data = []
        test_labels = []

        for batch_idx, (data, labels) in enumerate(self.test_loader):
            test_data.append(data)
            test_labels.append(labels)
            if batch_idx >= 10:  # 限制测试数据量
                break

        test_data = torch.cat(test_data, dim=0)[:100]  # 前100个样本
        test_labels = torch.cat(test_labels, dim=0)[:100]

        # 运行攻击测试
        attack_results = {}

        if self.config.enable_attack_test:
            # 获取梯度用于DLG攻击
            gradients = self._extract_gradients_for_attack()

            # 运行综合攻击测试
            attack_results = self.attack_suite.comprehensive_attack_test(
                test_data,
                gradients=gradients,
                original_data=test_data[:10],
                original_labels=test_labels[:10]
            )

        # 记录攻击结果
        for attack_type, results in attack_results.items():
            self.logger.log_attack_result(attack_type, round_num, results)

        return attack_results

    def _extract_gradients_for_attack(self):
        """提取梯度用于攻击测试"""
        # 简化实现：使用第一个客户端的梯度
        if not self.clients:
            return []

        first_client = next(iter(self.clients.values()))
        gradients = []

        # 获取一个批次的梯度
        for batch_idx, (data, target) in enumerate(first_client.train_loader):
            if batch_idx >= 1:  # 只用一个批次
                break

            data, target = data.to(self.device), target.to(self.device)

            # 计算梯度
            first_client.model.zero_grad()
            output = first_client.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            # 提取梯度
            batch_gradients = []
            for param in first_client.model.parameters():
                if param.grad is not None:
                    batch_gradients.append(param.grad.clone())

            gradients.append(batch_gradients)

        return gradients

    def save_checkpoint(self, round_num):
        """保存检查点"""
        checkpoint_data = {
            'global_model': self.global_model.state_dict(),
            'round': round_num,
            'training_metrics': dict(self.training_metrics),
            'client_metrics': dict(self.client_metrics)
        }

        # 保存隐私会计状态
        if self.privacy_accountant:
            checkpoint_data['privacy_accountant'] = self.privacy_accountant.export_privacy_report()

        self.logger.save_checkpoint(round_num, checkpoint_data)

    def generate_plots(self, round_num):
        """生成可视化图表"""
        if round_num % self.config.plot_interval != 0:
            return

        # 训练曲线
        if self.training_metrics:
            self.plotter.plot_training_curves(
                dict(self.training_metrics),
                save_name=f"training_curves_round_{round_num}"
            )

        # 隐私预算分解
        if self.privacy_accountant:
            privacy_report = self.privacy_accountant.export_privacy_report()
            self.plotter.plot_privacy_budget_breakdown(
                privacy_report['current_status'],
                save_name=f"privacy_budget_round_{round_num}"
            )

    def run_experiment(self):
        """运行完整实验"""
        self.logger.log("Starting DAGC-HDP experiment")

        start_time = time.time()

        try:
            # 主训练循环
            for round_num in range(1, self.config.total_rounds + 1):
                self.current_round = round_num

                # 检查隐私预算
                if self.config.use_dp and self.privacy_accountant.check_budget_exhausted():
                    self.logger.log("Privacy budget exhausted, stopping training")
                    break

                # 训练一轮
                eval_results = self.train_round(round_num)

                # 攻击测试
                if self.config.enable_attack_test:
                    attack_results = self.run_attack_test(round_num)

                # 保存检查点
                if round_num % self.config.save_checkpoint_interval == 0:
                    self.save_checkpoint(round_num)

                # 生成图表
                self.generate_plots(round_num)

                # 估计剩余时间
                elapsed = time.time() - start_time
                remaining = estimate_time_remaining(round_num, self.config.total_rounds, start_time)
                self.logger.log(f"Round {round_num}/{self.config.total_rounds} completed. "
                                f"Elapsed: {format_time(elapsed)}, Remaining: {format_time(remaining)}")

        except KeyboardInterrupt:
            self.logger.log("Training interrupted by user")

        except Exception as e:
            self.logger.log(f"Training error: {str(e)}", level="ERROR")
            raise

        finally:
            self.finalize_experiment()

    def finalize_experiment(self):
        """完成实验"""
        self.logger.log("Finalizing experiment...")

        # 最终评估
        final_results = self.evaluator.evaluate_model(
            self.global_model, self.test_loader, self.device
        )

        self.logger.log(f"Final results: {final_results}")

        # 保存最终模型
        save_model(
            self.global_model,
            os.path.join(self.experiment_dir, 'final_model.pth'),
            {
                'final_accuracy': final_results['accuracy'],
                'total_rounds': self.current_round,
                'config': self.config.to_dict()
            }
        )

        # 保存所有结果
        all_results = {
            'training_metrics': dict(self.training_metrics),
            'client_metrics': dict(self.client_metrics),
            'final_results': final_results,
            'config': self.config.to_dict()
        }

        if self.privacy_accountant:
            all_results['privacy_report'] = self.privacy_accountant.export_privacy_report()

        save_results(all_results, os.path.join(self.experiment_dir, 'experiment_results.json'))

        # 生成最终图表
        self.plotter.plot_training_curves(dict(self.training_metrics), save_name="final_training_curves")

        # 生成综合报告
        self.plotter.create_summary_report(
            {
                'training_metrics': dict(self.training_metrics),
                'privacy_data': self.privacy_accountant.export_privacy_report()[
                    'current_status'] if self.privacy_accountant else {},
                'final_results': final_results
            },
            report_name="comprehensive_report"
        )

        # 关闭日志器
        final_results = self.logger.close()

        self.logger.log("Experiment completed successfully")

        return final_results


def run_single_experiment(config_name=None):
    """运行单个实验"""
    if config_name:
        if hasattr(QuickConfigs, config_name):
            config = getattr(QuickConfigs, config_name)()
        else:
            raise ValueError(f"Unknown config: {config_name}")
    else:
        config = Config()

    config.validate_config()
    system = DAGCHDPSystem(config)
    system.initialize_system()
    results = system.run_experiment()

    return results


def run_ablation_study():
    """运行消融实验"""
    print("Starting ablation study...")

    base_config = Config()
    scenarios = base_config.get_scenario_configs()

    ablation_results = {}

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")

        # 创建配置
        config = Config()
        config.update_from_scenario(scenario)

        # 运行实验
        system = DAGCHDPSystem(config)
        system.initialize_system()
        results = system.run_experiment()

        ablation_results[scenario['name']] = results

    # 保存消融实验结果
    save_results(ablation_results, "./output/ablation_study_results.json")

    # 生成对比图表
    plotter = ExperimentPlotter(save_dir="./output/ablation_plots")
    plotter.plot_ablation_study(ablation_results, save_name="ablation_comparison")

    print("Ablation study completed!")
    return ablation_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="DAGC-HDP Federated Learning System")
    parser.add_argument("--config", type=str, help="Quick config name")
    parser.add_argument("--scenario", type=str, help="Experiment scenario")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--rounds", type=int, default=100, help="Training rounds")
    parser.add_argument("--clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Privacy budget")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    try:
        if args.ablation:
            # 运行消融实验
            run_ablation_study()
        else:
            # 单个实验
            if args.debug:
                config = QuickConfigs.debug_config()
            elif args.config:
                config = run_single_experiment(args.config)
                return
            else:
                config = Config()

            # 命令行参数覆盖
            if args.scenario:
                config.set_scenario(args.scenario)
            if args.dataset:
                config.dataset_name = args.dataset
            if args.rounds:
                config.total_rounds = args.rounds
            if args.clients:
                config.num_clients = args.clients
            if args.epsilon:
                config.total_epsilon = args.epsilon

            # 运行实验
            run_single_experiment()

    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()