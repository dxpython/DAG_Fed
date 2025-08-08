import os
import json
import time
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import torch

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentLogger:
    def __init__(self, experiment_name, config, log_dir='./logs'):
        self.experiment_name = experiment_name
        self.config = config
        self.log_dir = log_dir
        self.start_time = time.time()

        # 创建实验目录
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 初始化记录器
        self.console_log = True
        self.file_log = True
        self.tensorboard_log = TENSORBOARD_AVAILABLE and getattr(config, 'use_tensorboard', True)

        # 文件日志
        if self.file_log:
            self.log_file = open(os.path.join(self.experiment_dir, 'experiment.log'), 'w')

        # TensorBoard
        if self.tensorboard_log:
            self.tb_writer = SummaryWriter(os.path.join(self.experiment_dir, 'tensorboard'))

        # 指标存储
        self.metrics = defaultdict(list)
        self.round_metrics = defaultdict(dict)
        self.client_metrics = defaultdict(lambda: defaultdict(list))

        # 隐私相关记录
        self.privacy_metrics = defaultdict(list)
        self.attack_results = defaultdict(dict)

        # 保存配置
        self.save_config()

        self.log("=== Experiment Started ===")
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log(self, message, level='INFO'):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"

        if self.console_log:
            print(log_message)

        if self.file_log:
            self.log_file.write(log_message + '\n')
            self.log_file.flush()

    def log_round_start(self, round_num, selected_clients):
        """记录轮次开始"""
        self.log(f"Round {round_num} started with {len(selected_clients)} clients: {selected_clients}")

    def log_round_end(self, round_num, metrics):
        """记录轮次结束"""
        self.round_metrics[round_num] = metrics

        # 记录到TensorBoard
        if self.tensorboard_log:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Round/{key}', value, round_num)

        # 记录关键指标
        acc = metrics.get('accuracy', 0)
        loss = metrics.get('loss', 0)
        privacy_budget = metrics.get('privacy_budget', {}).get('consumed', 0)

        self.log(f"Round {round_num} - Accuracy: {acc:.4f}, Loss: {loss:.4f}, Privacy Budget: {privacy_budget:.4f}")

        # 存储历史指标
        self.metrics['accuracy'].append(acc)
        self.metrics['loss'].append(loss)
        self.metrics['privacy_budget'].append(privacy_budget)

    def log_client_metrics(self, client_id, round_num, metrics):
        """记录客户端指标"""
        self.client_metrics[client_id][round_num] = metrics

        if self.tensorboard_log:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Client_{client_id}/{key}', value, round_num)

    def log_privacy_metrics(self, round_num, privacy_data):
        """记录隐私相关指标"""
        self.privacy_metrics[round_num] = privacy_data

        if self.tensorboard_log:
            for key, value in privacy_data.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Privacy/{key}', value, round_num)

        # 记录关键隐私指标
        consumed = privacy_data.get('consumed', 0)
        remaining = privacy_data.get('remaining', 0)
        utilization = privacy_data.get('utilization', 0)

        self.log(f"Privacy - Consumed: {consumed:.4f}, Remaining: {remaining:.4f}, Utilization: {utilization:.2%}")

    def log_attack_result(self, attack_type, round_num, results):
        """记录攻击实验结果"""
        if attack_type not in self.attack_results:
            self.attack_results[attack_type] = {}

        self.attack_results[attack_type][round_num] = results

        if self.tensorboard_log:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Attack_{attack_type}/{key}', value, round_num)

        # 记录攻击结果
        success_rate = results.get('success_rate', 0)
        avg_mse = results.get('avg_mse', 0)

        self.log(f"Attack {attack_type} - Success Rate: {success_rate:.2%}, Avg MSE: {avg_mse:.6f}")

    def log_gradient_stats(self, round_num, grad_stats):
        """记录梯度统计信息"""
        if self.tensorboard_log:
            for layer_name, stats in grad_stats.items():
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(f'Gradient/{layer_name}_{stat_name}', value, round_num)

    def log_noise_schedule(self, round_num, noise_data):
        """记录噪声调度信息"""
        if self.tensorboard_log:
            for layer_name, noise_level in noise_data.items():
                if isinstance(noise_level, (int, float)):
                    self.tb_writer.add_scalar(f'Noise/{layer_name}', noise_level, round_num)

    def save_checkpoint(self, round_num, model_state=None, additional_data=None):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_data = {
            'round': round_num,
            'metrics': dict(self.round_metrics),
            'client_metrics': dict(self.client_metrics),
            'privacy_metrics': dict(self.privacy_metrics),
            'attack_results': dict(self.attack_results),
            'timestamp': datetime.now().isoformat()
        }

        if model_state is not None:
            checkpoint_data['model_state'] = model_state

        if additional_data is not None:
            checkpoint_data.update(additional_data)

        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_round_{round_num}.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        self.log(f"Checkpoint saved for round {round_num}")

    def save_config(self):
        """保存实验配置"""
        config_dict = {}
        for key, value in vars(self.config).items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)

        config_file = os.path.join(self.experiment_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def save_final_results(self):
        """保存最终实验结果"""
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': time.time() - self.start_time,
            'total_rounds': len(self.round_metrics),
            'final_metrics': {
                'accuracy': self.metrics['accuracy'][-1] if self.metrics['accuracy'] else 0,
                'loss': self.metrics['loss'][-1] if self.metrics['loss'] else 0,
                'privacy_budget': self.metrics['privacy_budget'][-1] if self.metrics['privacy_budget'] else 0
            },
            'round_metrics': dict(self.round_metrics),
            'client_metrics': dict(self.client_metrics),
            'privacy_metrics': dict(self.privacy_metrics),
            'attack_results': dict(self.attack_results),
            'metric_trajectories': dict(self.metrics)
        }

        # 保存JSON格式
        results_file = os.path.join(self.experiment_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 保存pickle格式
        results_pkl = os.path.join(self.experiment_dir, 'results.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(results, f)

        self.log("Final results saved")
        return results

    def get_summary_stats(self):
        """获取实验摘要统计"""
        if not self.metrics['accuracy']:
            return {}

        stats = {
            'max_accuracy': max(self.metrics['accuracy']),
            'final_accuracy': self.metrics['accuracy'][-1],
            'avg_accuracy': np.mean(self.metrics['accuracy']),
            'min_loss': min(self.metrics['loss']) if self.metrics['loss'] else 0,
            'final_loss': self.metrics['loss'][-1] if self.metrics['loss'] else 0,
            'total_privacy_budget': self.metrics['privacy_budget'][-1] if self.metrics['privacy_budget'] else 0,
            'convergence_round': self._find_convergence_round(),
            'total_rounds': len(self.round_metrics)
        }

        return stats

    def _find_convergence_round(self, window=5, threshold=0.01):
        """寻找收敛轮次"""
        if len(self.metrics['accuracy']) < window:
            return -1

        for i in range(window, len(self.metrics['accuracy'])):
            recent_acc = self.metrics['accuracy'][i - window:i]
            if max(recent_acc) - min(recent_acc) < threshold:
                return i - window

        return -1

    def plot_metrics(self, save_path=None):
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # 准确率
            if self.metrics['accuracy']:
                axes[0, 0].plot(self.metrics['accuracy'])
                axes[0, 0].set_title('Accuracy')
                axes[0, 0].set_xlabel('Round')
                axes[0, 0].set_ylabel('Accuracy')

            # 损失
            if self.metrics['loss']:
                axes[0, 1].plot(self.metrics['loss'])
                axes[0, 1].set_title('Loss')
                axes[0, 1].set_xlabel('Round')
                axes[0, 1].set_ylabel('Loss')

            # 隐私预算
            if self.metrics['privacy_budget']:
                axes[1, 0].plot(self.metrics['privacy_budget'])
                axes[1, 0].set_title('Privacy Budget Consumption')
                axes[1, 0].set_xlabel('Round')
                axes[1, 0].set_ylabel('Consumed Budget')

            # 攻击成功率
            if 'dlg' in self.attack_results:
                rounds = sorted(self.attack_results['dlg'].keys())
                success_rates = [self.attack_results['dlg'][r].get('success_rate', 0) for r in rounds]
                axes[1, 1].plot(rounds, success_rates)
                axes[1, 1].set_title('Attack Success Rate')
                axes[1, 1].set_xlabel('Round')
                axes[1, 1].set_ylabel('Success Rate')

            plt.tight_layout()

            if save_path is None:
                save_path = os.path.join(self.experiment_dir, 'metrics_plot.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.log(f"Metrics plot saved to {save_path}")

        except ImportError:
            self.log("Matplotlib not available, skipping plot generation")

    def close(self):
        """关闭日志器"""
        self.log("=== Experiment Completed ===")
        self.log(f"Total duration: {time.time() - self.start_time:.2f} seconds")

        if self.file_log:
            self.log_file.close()

        if self.tensorboard_log:
            self.tb_writer.close()

        # 保存最终结果
        final_results = self.save_final_results()

        # 生成摘要
        summary = self.get_summary_stats()
        self.log(f"Experiment Summary: {summary}")

        return final_results


class MetricsAggregator:
    """指标聚合器"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def add_metric(self, name, value, round_num=None):
        """添加指标"""
        self.metrics[name].append({'value': value, 'round': round_num, 'timestamp': time.time()})

    def get_metric_history(self, name):
        """获取指标历史"""
        return self.metrics.get(name, [])

    def get_latest_metric(self, name):
        """获取最新指标值"""
        history = self.metrics.get(name, [])
        return history[-1]['value'] if history else None

    def compute_average(self, name, last_n=None):
        """计算平均值"""
        history = self.metrics.get(name, [])
        if not history:
            return 0

        values = [item['value'] for item in history]
        if last_n:
            values = values[-last_n:]

        return np.mean(values)

    def compute_trend(self, name, window=5):
        """计算趋势"""
        history = self.metrics.get(name, [])
        if len(history) < window:
            return 0

        recent_values = [item['value'] for item in history[-window:]]
        return np.mean(np.diff(recent_values))