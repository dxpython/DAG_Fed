import os
import torch

class Config:
    """DAGC-HDP配置类"""
    # ========== 实验基础配置 ==========
    experiment_name = "DAGC-HDP"
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # 输出目录
    output_dir = "./output"
    log_dir = "./logs"
    model_dir = "./models"
    plot_dir = "./plots"

    # ========== 数据集配置 ==========
    dataset_name = "cifar10"  # cifar10, cifar100, mnist, fashionmnist
    data_dir = "./data"
    batch_size = 32
    test_batch_size = 100
    num_workers = 2

    # 非IID设置
    non_iid_method = "dirichlet"  # dirichlet, class_based, quantity_based, mixed, iid
    non_iid_alpha = 0.5  # Dirichlet分布参数
    classes_per_client = 2  # 每个客户端类别数

    # ========== 模型配置 ==========
    model_type = "auto"  # auto, cnn, cifar_cnn, resnet18, resnet34, lenet, vgg
    num_classes = 10

    # ========== 联邦学习配置 ==========
    num_clients = 10
    num_selected_clients = 5
    total_rounds = 100
    local_epochs = 5

    # 客户端选择策略
    client_selection_method = "random"  # random, loss_based, data_size

    # ========== 优化器配置 ==========
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    lr_scheduler = "cosine"  # cosine, step, none

    # ========== DAGC (动态自适应梯度剪切) 配置 ==========
    # 梯度剪切
    default_clip = 1.0
    quantile_q = 0.8  # 动态剪切分位点
    window_size = 10  # 滑动窗口大小
    min_history = 5  # 最小历史记录数

    # 剪切方法
    clipping_method = "adaptive"  # adaptive, fixed, layer_wise

    # ========== HDP (层次差分隐私) 配置 ==========
    # 隐私预算
    total_epsilon = 8.0
    target_delta = 1e-5

    # 噪声配置
    base_noise_multiplier = 0.1
    min_noise_multiplier = 0.01
    default_noise_multiplier = 0.1

    # 噪声调度
    noise_decay_method = "cosine"  # cosine, exponential, linear, none
    decay_rate = 0.01

    # 层次噪声分配
    importance_method = "norm"  # norm, variance, magnitude
    importance_window = 20
    budget_allocation_method = "proportional"  # proportional, inverse, uniform
    budget_allocation_rate = 0.1

    # 隐私会计
    composition_method = "rdp"  # rdp, advanced, basic
    alpha_max = 32
    history_size = 1000

    # ========== 动态调度配置 ==========
    # 性能反馈
    perf_window_size = 10
    performance_threshold = 0.01

    # 噪声调整
    noise_adjustment_patience = 5
    noise_adjustment_factor = 0.1

    # ========== 聚合器配置 ==========
    aggregation_method = "fedavg"  # fedavg, scaffold, adaptive
    weighting_method = "uniform"  # uniform, data_size, loss_based

    # ========== 攻击测试配置 ==========
    enable_attack_test = True
    attack_test_interval = 10  # 每隔多少轮测试一次

    # DLG攻击
    dlg_iterations = 1000
    dlg_lr = 1.0
    dlg_tv_weight = 1e-4

    # 属性推断攻击
    enable_property_inference = True
    property_inference_epochs = 100

    # 成员推断攻击
    enable_membership_inference = True
    membership_inference_epochs = 100

    # ========== 日志配置 ==========
    log_level = "INFO"
    console_log = True
    file_log = True
    use_tensorboard = True

    # 保存配置
    save_model_interval = 20
    save_checkpoint_interval = 10

    # ========== 可视化配置 ==========
    plot_interval = 10
    plot_dpi = 300
    plot_figsize = (10, 6)

    # ========== 实验场景配置 ==========
    # 场景1: 标准对比实验
    scenario_baseline = {
        "name": "baseline",
        "use_dp": False,
        "use_dynamic_clipping": False,
        "use_hierarchical_noise": False,
        "use_adaptive_scheduling": False
    }

    # 场景2: 仅差分隐私
    scenario_dp_only = {
        "name": "dp_only",
        "use_dp": True,
        "use_dynamic_clipping": False,
        "use_hierarchical_noise": False,
        "use_adaptive_scheduling": False,
        "base_noise_multiplier": 0.1
    }

    # 场景3: 仅动态剪切
    scenario_dagc_only = {
        "name": "dagc_only",
        "use_dp": True,
        "use_dynamic_clipping": True,
        "use_hierarchical_noise": False,
        "use_adaptive_scheduling": False
    }

    # 场景4: 仅层次噪声
    scenario_hdp_only = {
        "name": "hdp_only",
        "use_dp": True,
        "use_dynamic_clipping": False,
        "use_hierarchical_noise": True,
        "use_adaptive_scheduling": False
    }

    # 场景5: 完整DAGC-HDP
    scenario_full = {
        "name": "dagc_hdp_full",
        "use_dp": True,
        "use_dynamic_clipping": True,
        "use_hierarchical_noise": True,
        "use_adaptive_scheduling": True
    }

    # 当前实验场景
    current_scenario = scenario_full

    # ========== 数据集特定配置 ==========
    dataset_configs = {
        'cifar10': {
            'num_classes': 10,
            'input_channels': 3,
            'input_size': 32,
            'model_type': 'cifar_cnn'
        },
        'cifar100': {
            'num_classes': 100,
            'input_channels': 3,
            'input_size': 32,
            'model_type': 'cifar_cnn'
        },
        'mnist': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': 28,
            'model_type': 'cnn'
        },
        'fashionmnist': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': 28,
            'model_type': 'cnn'
        }
    }

    def __init__(self):
        # 根据数据集自动配置
        if self.dataset_name in self.dataset_configs:
            dataset_config = self.dataset_configs[self.dataset_name]
            self.num_classes = dataset_config['num_classes']
            if self.model_type == 'auto':
                self.model_type = dataset_config['model_type']

        # 根据当前场景更新配置
        self.update_from_scenario(self.current_scenario)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def update_from_scenario(self, scenario):
        """根据实验场景更新配置"""
        for key, value in scenario.items():
            if key != 'name':
                setattr(self, key, value)

    def get_scenario_configs(self):
        """获取所有实验场景配置"""
        return [
            self.scenario_baseline,
            self.scenario_dp_only,
            self.scenario_dagc_only,
            self.scenario_hdp_only,
            self.scenario_full
        ]

    def set_scenario(self, scenario_name):
        """设置实验场景"""
        scenarios = {
            'baseline': self.scenario_baseline,
            'dp_only': self.scenario_dp_only,
            'dagc_only': self.scenario_dagc_only,
            'hdp_only': self.scenario_hdp_only,
            'full': self.scenario_full
        }

        if scenario_name in scenarios:
            self.current_scenario = scenarios[scenario_name]
            self.update_from_scenario(self.current_scenario)
            print(f"Scenario set to: {scenario_name}")
        else:
            print(f"Unknown scenario: {scenario_name}")

    def get_experiment_id(self):
        """生成实验ID"""
        scenario_name = self.current_scenario['name']
        return f"{self.experiment_name}_{scenario_name}_{self.dataset_name}_clients{self.num_clients}_rounds{self.total_rounds}"

    def validate_config(self):
        """验证配置有效性"""
        errors = []

        # 基本参数检查
        if self.num_selected_clients > self.num_clients:
            errors.append("num_selected_clients cannot be greater than num_clients")

        if self.total_epsilon <= 0:
            errors.append("total_epsilon must be positive")

        if self.target_delta <= 0 or self.target_delta >= 1:
            errors.append("target_delta must be in (0, 1)")

        if self.non_iid_alpha <= 0:
            errors.append("non_iid_alpha must be positive")

        # 数据集检查
        if self.dataset_name not in self.dataset_configs:
            errors.append(f"Unsupported dataset: {self.dataset_name}")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))

        return True

    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("DAGC-HDP Configuration")
        print("=" * 60)
        print(f"Experiment: {self.get_experiment_id()}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.model_type}")
        print(f"Clients: {self.num_clients} (select {self.num_selected_clients})")
        print(f"Rounds: {self.total_rounds}")
        print(f"Local epochs: {self.local_epochs}")
        print(f"Privacy budget: ε={self.total_epsilon}, δ={self.target_delta}")
        print(f"Non-IID: {self.non_iid_method} (α={self.non_iid_alpha})")
        print(f"Scenario: {self.current_scenario['name']}")
        print("=" * 60)

    def to_dict(self):
        """转换为字典"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_dict[key] = value
        return config_dict

    def save_config(self, filepath):
        """保存配置到文件"""
        import json

        config_dict = self.to_dict()

        for key, value in config_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                config_dict[key] = str(value)

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath):
        """从文件加载配置"""
        import json

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)

        return config


# 预设配置
class QuickConfigs:
    """快速配置预设"""

    @staticmethod
    def cifar10_iid():
        """CIFAR-10 IID配置"""
        config = Config()
        config.dataset_name = "cifar10"
        config.non_iid_method = "iid"
        config.num_clients = 10
        config.total_rounds = 100
        return config

    @staticmethod
    def cifar10_non_iid():
        """CIFAR-10 非IID配置"""
        config = Config()
        config.dataset_name = "cifar10"
        config.non_iid_method = "dirichlet"
        config.non_iid_alpha = 0.5
        config.num_clients = 10
        config.total_rounds = 100
        return config

    @staticmethod
    def mnist_baseline():
        """MNIST基线配置"""
        config = Config()
        config.dataset_name = "mnist"
        config.set_scenario("baseline")
        config.num_clients = 20
        config.total_rounds = 50
        return config

    @staticmethod
    def high_privacy():
        """高隐私保护配置"""
        config = Config()
        config.total_epsilon = 1.0
        config.base_noise_multiplier = 1.0
        config.num_clients = 20
        config.total_rounds = 200
        return config

    @staticmethod
    def low_privacy():
        """低隐私保护配置"""
        config = Config()
        config.total_epsilon = 10.0
        config.base_noise_multiplier = 0.05
        config.num_clients = 10
        config.total_rounds = 50
        return config

    @staticmethod
    def debug_config():
        """调试配置"""
        config = Config()
        config.num_clients = 3
        config.num_selected_clients = 2
        config.total_rounds = 5
        config.local_epochs = 1
        config.batch_size = 16
        config.enable_attack_test = False
        return config


# 全局配置实例
config = Config()

# 配置验证
if __name__ == "__main__":
    # 验证配置
    config.validate_config()
    config.print_config()

    # 测试场景切换
    print("\nTesting scenario switching:")
    for scenario_name in ['baseline', 'dp_only', 'full']:
        config.set_scenario(scenario_name)
        print(f"Current scenario: {config.current_scenario['name']}")

    # 测试快速配置
    print("\nTesting quick configs:")
    debug_config = QuickConfigs.debug_config()
    print(f"Debug config - clients: {debug_config.num_clients}, rounds: {debug_config.total_rounds}")