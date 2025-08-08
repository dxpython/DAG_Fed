import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExperimentPlotter:
    """实验结果可视化器"""

    def __init__(self, save_dir=None, figsize=(10, 6), dpi=300):
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(self, metrics_dict, title="Training Curves", save_name=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 准确率曲线
        if 'accuracy' in metrics_dict:
            axes[0, 0].plot(metrics_dict['accuracy'], marker='o', linewidth=2)
            axes[0, 0].set_title('Accuracy')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)

        # 损失曲线
        if 'loss' in metrics_dict:
            axes[0, 1].plot(metrics_dict['loss'], marker='s', linewidth=2, color='red')
            axes[0, 1].set_title('Loss')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # 隐私预算消耗
        if 'privacy_budget' in metrics_dict:
            axes[1, 0].plot(metrics_dict['privacy_budget'], marker='^', linewidth=2, color='green')
            axes[1, 0].set_title('Privacy Budget Consumption')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Consumed Budget')
            axes[1, 0].grid(True, alpha=0.3)

        # 收敛速度
        if 'accuracy' in metrics_dict:
            convergence_data = np.diff(metrics_dict['accuracy'])
            axes[1, 1].plot(convergence_data, marker='d', linewidth=2, color='purple')
            axes[1, 1].set_title('Convergence Speed')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Accuracy Change')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_privacy_budget_breakdown(self, budget_data, title="Privacy Budget Breakdown", save_name=None):
        """绘制隐私预算分解"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 饼图 - 各层预算分配
        if 'layer_consumption' in budget_data:
            layer_data = budget_data['layer_consumption']
            labels = list(layer_data.keys())
            sizes = list(layer_data.values())

            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Privacy Budget by Layer')

        # 累积消耗曲线
        if 'consumption_history' in budget_data:
            history = budget_data['consumption_history']
            rounds = range(len(history))

            ax2.plot(rounds, history, marker='o', linewidth=2, color='red')
            ax2.fill_between(rounds, history, alpha=0.3, color='red')
            ax2.set_title('Cumulative Privacy Budget Consumption')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Consumed Budget')
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_attack_results(self, attack_data, title="Attack Results", save_name=None):
        """绘制攻击结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # DLG攻击成功率
        if 'dlg' in attack_data:
            dlg_data = attack_data['dlg']
            if 'success_rate_history' in dlg_data:
                rounds = range(len(dlg_data['success_rate_history']))
                success_rates = dlg_data['success_rate_history']

                axes[0, 0].plot(rounds, success_rates, marker='o', linewidth=2, color='red')
                axes[0, 0].set_title('DLG Attack Success Rate')
                axes[0, 0].set_xlabel('Round')
                axes[0, 0].set_ylabel('Success Rate')
                axes[0, 0].grid(True, alpha=0.3)

        # 重建误差分布
        if 'reconstruction_errors' in attack_data:
            errors = attack_data['reconstruction_errors']
            axes[0, 1].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].set_title('Reconstruction Error Distribution')
            axes[0, 1].set_xlabel('MSE')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

        # 属性推断准确率
        if 'property_inference' in attack_data:
            prop_data = attack_data['property_inference']
            if 'accuracy_history' in prop_data:
                rounds = range(len(prop_data['accuracy_history']))
                accuracies = prop_data['accuracy_history']

                axes[1, 0].plot(rounds, accuracies, marker='s', linewidth=2, color='green')
                axes[1, 0].set_title('Property Inference Accuracy')
                axes[1, 0].set_xlabel('Round')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].grid(True, alpha=0.3)

        # 攻击效果对比
        if 'comparison' in attack_data:
            comp_data = attack_data['comparison']
            attack_types = list(comp_data.keys())
            success_rates = list(comp_data.values())

            bars = axes[1, 1].bar(attack_types, success_rates, color=['red', 'green', 'blue'])
            axes[1, 1].set_title('Attack Success Rate Comparison')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)

            # 添加数值标签
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{rate:.3f}', ha='center', va='bottom')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_client_performance(self, client_data, title="Client Performance", save_name=None):
        """绘制客户端性能对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 客户端准确率对比
        if 'accuracy' in client_data:
            client_acc = client_data['accuracy']
            clients = list(client_acc.keys())
            accuracies = list(client_acc.values())

            axes[0, 0].bar(clients, accuracies, alpha=0.7)
            axes[0, 0].set_title('Client Accuracy Comparison')
            axes[0, 0].set_xlabel('Client ID')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 客户端数据分布
        if 'data_distribution' in client_data:
            dist_data = client_data['data_distribution']

            # 创建热力图数据
            clients = list(dist_data.keys())
            classes = list(range(len(list(dist_data.values())[0])))

            heatmap_data = np.array([dist_data[client] for client in clients])

            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                        xticklabels=classes, yticklabels=clients, ax=axes[0, 1])
            axes[0, 1].set_title('Client Data Distribution')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Client')

        # 客户端训练轮次参与度
        if 'participation' in client_data:
            part_data = client_data['participation']
            clients = list(part_data.keys())
            participations = list(part_data.values())

            axes[1, 0].bar(clients, participations, alpha=0.7, color='green')
            axes[1, 0].set_title('Client Participation Rate')
            axes[1, 0].set_xlabel('Client ID')
            axes[1, 0].set_ylabel('Participation Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 客户端模型差异
        if 'model_divergence' in client_data:
            div_data = client_data['model_divergence']

            # 绘制箱线图
            divergences = list(div_data.values())
            axes[1, 1].boxplot(divergences, labels=list(div_data.keys()))
            axes[1, 1].set_title('Client Model Divergence')
            axes[1, 1].set_xlabel('Client ID')
            axes[1, 1].set_ylabel('Divergence')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_noise_schedule(self, noise_data, title="Noise Schedule", save_name=None):
        """绘制噪声调度"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 各层噪声水平变化
        if 'layer_noise_history' in noise_data:
            layer_history = noise_data['layer_noise_history']
            rounds = range(len(list(layer_history.values())[0]))

            for layer_name, noise_levels in layer_history.items():
                axes[0, 0].plot(rounds, noise_levels, marker='o', label=layer_name)

            axes[0, 0].set_title('Layer Noise Levels')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Noise Level')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 噪声调整因子
        if 'adjustment_factor' in noise_data:
            factors = noise_data['adjustment_factor']
            rounds = range(len(factors))

            axes[0, 1].plot(rounds, factors, marker='s', linewidth=2, color='red')
            axes[0, 1].set_title('Noise Adjustment Factor')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Adjustment Factor')
            axes[0, 1].grid(True, alpha=0.3)

        # 梯度范数统计
        if 'gradient_norms' in noise_data:
            grad_norms = noise_data['gradient_norms']

            axes[1, 0].hist(grad_norms, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_title('Gradient Norm Distribution')
            axes[1, 0].set_xlabel('Gradient Norm')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 剪切阈值动态调整
        if 'clip_bounds' in noise_data:
            clip_history = noise_data['clip_bounds']
            rounds = range(len(clip_history))

            axes[1, 1].plot(rounds, clip_history, marker='^', linewidth=2, color='green')
            axes[1, 1].set_title('Dynamic Clipping Bounds')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Clipping Bound')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_comparison(self, comparison_data, title="Method Comparison", save_name=None):
        """绘制方法对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        methods = list(comparison_data.keys())

        # 准确率对比
        if all('accuracy' in data for data in comparison_data.values()):
            accuracies = [data['accuracy'][-1] for data in comparison_data.values()]

            bars = axes[0, 0].bar(methods, accuracies, alpha=0.7)
            axes[0, 0].set_title('Final Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom')

        # 收敛速度对比
        if all('accuracy' in data for data in comparison_data.values()):
            for method, data in comparison_data.items():
                rounds = range(len(data['accuracy']))
                axes[0, 1].plot(rounds, data['accuracy'], marker='o', label=method)

            axes[0, 1].set_title('Convergence Comparison')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 隐私预算消耗对比
        if all('privacy_budget' in data for data in comparison_data.values()):
            budget_consumptions = [data['privacy_budget'][-1] for data in comparison_data.values()]

            bars = axes[1, 0].bar(methods, budget_consumptions, alpha=0.7, color='green')
            axes[1, 0].set_title('Privacy Budget Consumption')
            axes[1, 0].set_ylabel('Consumed Budget')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, budget in zip(bars, budget_consumptions):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{budget:.3f}', ha='center', va='bottom')

        # 攻击抵抗能力对比
        if all('attack_resistance' in data for data in comparison_data.values()):
            attack_resistance = [data['attack_resistance'] for data in comparison_data.values()]

            bars = axes[1, 1].bar(methods, attack_resistance, alpha=0.7, color='red')
            axes[1, 1].set_title('Attack Resistance')
            axes[1, 1].set_ylabel('Resistance Score')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_data_distribution(self, data_stats, title="Data Distribution", save_name=None):
        """绘制数据分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 客户端数据量分布
        if 'client_data_sizes' in data_stats:
            sizes = data_stats['client_data_sizes']
            clients = list(sizes.keys())
            data_sizes = list(sizes.values())

            axes[0, 0].bar(clients, data_sizes, alpha=0.7)
            axes[0, 0].set_title('Client Data Size Distribution')
            axes[0, 0].set_xlabel('Client ID')
            axes[0, 0].set_ylabel('Data Size')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 类别分布
        if 'class_distribution' in data_stats:
            class_dist = data_stats['class_distribution']
            classes = list(class_dist.keys())
            counts = list(class_dist.values())

            axes[0, 1].pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Global Class Distribution')

        # 非IID程度
        if 'non_iid_metrics' in data_stats:
            metrics = data_stats['non_iid_metrics']

            if 'kl_divergences' in metrics:
                kl_divs = metrics['kl_divergences']
                axes[1, 0].hist(kl_divs, bins=20, alpha=0.7, color='blue', edgecolor='black')
                axes[1, 0].set_title('KL Divergence Distribution')
                axes[1, 0].set_xlabel('KL Divergence')
                axes[1, 0].set_ylabel('Frequency')

        # 客户端相似性矩阵
        if 'client_similarity' in data_stats:
            similarity_matrix = data_stats['client_similarity']

            sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlBu',
                        ax=axes[1, 1])
            axes[1, 1].set_title('Client Similarity Matrix')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_ablation_study(self, ablation_data, title="Ablation Study", save_name=None):
        """绘制消融实验"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        components = list(ablation_data.keys())

        # 各组件对准确率的影响
        if all('accuracy' in data for data in ablation_data.values()):
            accuracies = [data['accuracy'][-1] for data in ablation_data.values()]

            bars = axes[0, 0].bar(components, accuracies, alpha=0.7)
            axes[0, 0].set_title('Component Impact on Accuracy')
            axes[0, 0].set_ylabel('Final Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom')

        # 隐私保护效果
        if all('privacy_protection' in data for data in ablation_data.values()):
            privacy_scores = [data['privacy_protection'] for data in ablation_data.values()]

            bars = axes[0, 1].bar(components, privacy_scores, alpha=0.7, color='green')
            axes[0, 1].set_title('Privacy Protection Effect')
            axes[0, 1].set_ylabel('Protection Score')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 收敛速度对比
        if all('convergence_speed' in data for data in ablation_data.values()):
            conv_speeds = [data['convergence_speed'] for data in ablation_data.values()]

            bars = axes[1, 0].bar(components, conv_speeds, alpha=0.7, color='red')
            axes[1, 0].set_title('Convergence Speed')
            axes[1, 0].set_ylabel('Rounds to Converge')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 性能-隐私权衡
        if all('accuracy' in data and 'privacy_cost' in data for data in ablation_data.values()):
            accuracies = [data['accuracy'][-1] for data in ablation_data.values()]
            privacy_costs = [data['privacy_cost'] for data in ablation_data.values()]

            scatter = axes[1, 1].scatter(privacy_costs, accuracies, s=100, alpha=0.7)

            # 添加标签
            for i, comp in enumerate(components):
                axes[1, 1].annotate(comp, (privacy_costs[i], accuracies[i]))

            axes[1, 1].set_title('Accuracy vs Privacy Cost')
            axes[1, 1].set_xlabel('Privacy Cost')
            axes[1, 1].set_ylabel('Accuracy')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def _save_plot(self, fig, filename):
        """保存图片"""
        if self.save_dir:
            filepath = os.path.join(self.save_dir, f"{filename}.png")
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {filepath}")

    def create_summary_report(self, all_data, report_name="experiment_report"):
        """创建综合报告"""
        from matplotlib.backends.backend_pdf import PdfPages

        if self.save_dir:
            pdf_path = os.path.join(self.save_dir, f"{report_name}.pdf")

            with PdfPages(pdf_path) as pdf:
                # 训练曲线
                if 'training_metrics' in all_data:
                    fig1 = self.plot_training_curves(all_data['training_metrics'],
                                                     title="Training Performance")
                    pdf.savefig(fig1, bbox_inches='tight')
                    plt.close(fig1)

                # 隐私预算分析
                if 'privacy_data' in all_data:
                    fig2 = self.plot_privacy_budget_breakdown(all_data['privacy_data'],
                                                              title="Privacy Budget Analysis")
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)

                # 攻击结果
                if 'attack_results' in all_data:
                    fig3 = self.plot_attack_results(all_data['attack_results'],
                                                    title="Attack Resistance Analysis")
                    pdf.savefig(fig3, bbox_inches='tight')
                    plt.close(fig3)

                # 方法对比
                if 'comparison' in all_data:
                    fig4 = self.plot_comparison(all_data['comparison'],
                                                title="Method Comparison")
                    pdf.savefig(fig4, bbox_inches='tight')
                    plt.close(fig4)

            print(f"Summary report saved to {pdf_path}")


# 便捷函数
def quick_plot_training(metrics, save_path=None):
    """快速绘制训练曲线"""
    plotter = ExperimentPlotter()
    return plotter.plot_training_curves(metrics, save_name=save_path)


def quick_plot_comparison(comparison_data, save_path=None):
    """快速绘制方法对比"""
    plotter = ExperimentPlotter()
    return plotter.plot_comparison(comparison_data, save_name=save_path)


def quick_plot_attack_results(attack_data, save_path=None):
    """快速绘制攻击结果"""
    plotter = ExperimentPlotter()
    return plotter.plot_attack_results(attack_data, save_name=save_path)