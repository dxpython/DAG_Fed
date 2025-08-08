import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from collections import defaultdict, Counter
import random
import os
import pickle


class FedDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset_name
        self.num_clients = config.num_clients
        self.data_dir = config.data_dir

        # 数据集配置
        self.dataset_configs = {
            'cifar10': {
                'num_classes': 10,
                'input_size': (3, 32, 32),
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010]
            },
            'cifar100': {
                'num_classes': 100,
                'input_size': (3, 32, 32),
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761]
            },
            'mnist': {
                'num_classes': 10,
                'input_size': (1, 28, 28),
                'mean': [0.1307],
                'std': [0.3081]
            },
            'fashionmnist': {
                'num_classes': 10,
                'input_size': (1, 28, 28),
                'mean': [0.2860],
                'std': [0.3530]
            }
        }

        # 非IID配置
        self.non_iid_method = config.non_iid_method
        self.non_iid_alpha = config.non_iid_alpha if hasattr(config, 'non_iid_alpha') else 0.5
        self.classes_per_client = config.classes_per_client if hasattr(config, 'classes_per_client') else 2

        # 数据缓存
        self.client_data_cache = {}
        self.dataset_stats = {}

    def load_dataset(self):
        """加载指定数据集"""
        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        config = self.dataset_configs[self.dataset_name]

        # 定义数据预处理
        train_transform = self._get_train_transform(config)
        test_transform = self._get_test_transform(config)

        # 加载数据集
        if self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=test_transform)
        elif self.dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=test_transform)
        elif self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=test_transform)
        elif self.dataset_name == 'fashionmnist':
            train_dataset = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=test_transform)

        return train_dataset, test_dataset

    def _get_train_transform(self, config):
        """获取训练数据预处理"""
        if 'cifar' in self.dataset_name:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:  # MNIST, FashionMNIST
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])

    def _get_test_transform(self, config):
        """获取测试数据预处理"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])

    def create_federated_dataset(self, train_dataset, test_dataset):
        """创建联邦学习数据集"""
        # 提取数据和标签
        train_data, train_labels = self._extract_data_labels(train_dataset)
        test_data, test_labels = self._extract_data_labels(test_dataset)

        # 按非IID方法划分数据
        if self.non_iid_method == 'dirichlet':
            client_indices = self._dirichlet_split(train_labels, self.non_iid_alpha)
        elif self.non_iid_method == 'class_based':
            client_indices = self._class_based_split(train_labels, self.classes_per_client)
        elif self.non_iid_method == 'quantity_based':
            client_indices = self._quantity_based_split(len(train_labels))
        elif self.non_iid_method == 'mixed':
            client_indices = self._mixed_split(train_labels)
        else:  # IID
            client_indices = self._iid_split(len(train_labels))

        # 创建客户端数据加载器
        client_loaders = {}
        for client_id, indices in client_indices.items():
            client_dataset = data.Subset(train_dataset, indices)
            client_loader = data.DataLoader(
                client_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 0
            )
            client_loaders[client_id] = client_loader

        # 全局测试数据加载器
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size if hasattr(self.config, 'test_batch_size') else 100,
            shuffle=False,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 0
        )

        # 计算数据集统计信息
        self._compute_dataset_stats(client_indices, train_labels)

        return client_loaders, test_loader

    def _extract_data_labels(self, dataset):
        """提取数据和标签"""
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            return dataset.data, dataset.targets
        else:
            # 对于自定义数据集
            data_list = []
            labels_list = []
            for i in range(len(dataset)):
                data_point, label = dataset[i]
                data_list.append(data_point)
                labels_list.append(label)
            return data_list, labels_list

    def _dirichlet_split(self, labels, alpha):
        """Dirichlet分布的非IID划分"""
        labels = np.array(labels)
        num_classes = len(np.unique(labels))

        # 每个类别的样本索引
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        client_indices = {i: [] for i in range(self.num_clients)}

        for class_idx in range(num_classes):
            # 使用Dirichlet分布生成每个客户端的样本比例
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))

            # 根据比例分配样本
            class_samples = class_indices[class_idx]
            np.random.shuffle(class_samples)

            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + int(proportions[client_id] * len(class_samples))
                client_indices[client_id].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx

        # 打乱每个客户端的样本顺序
        for client_id in client_indices:
            np.random.shuffle(client_indices[client_id])

        return client_indices

    def _class_based_split(self, labels, classes_per_client):
        """基于类别的非IID划分"""
        labels = np.array(labels)
        num_classes = len(np.unique(labels))

        # 每个类别的样本索引
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        client_indices = {i: [] for i in range(self.num_clients)}

        # 为每个客户端分配指定数量的类别
        for client_id in range(self.num_clients):
            # 随机选择类别
            selected_classes = np.random.choice(
                num_classes,
                size=min(classes_per_client, num_classes),
                replace=False
            )

            for class_idx in selected_classes:
                # 将该类别的样本分配给客户端
                class_samples = class_indices[class_idx].copy()
                np.random.shuffle(class_samples)

                # 平均分配给有该类别的客户端
                clients_with_class = [cid for cid in range(self.num_clients)
                                      if class_idx in [
                                          np.random.choice(num_classes, size=min(classes_per_client, num_classes),
                                                           replace=False)
                                          for _ in range(1)][0]]

                samples_per_client = len(class_samples) // max(1, len(clients_with_class))
                start_idx = client_id * samples_per_client
                end_idx = min(start_idx + samples_per_client, len(class_samples))

                if start_idx < len(class_samples):
                    client_indices[client_id].extend(class_samples[start_idx:end_idx])

        return client_indices

    def _quantity_based_split(self, total_samples):
        """基于数量的非IID划分"""
        client_indices = {i: [] for i in range(self.num_clients)}

        # 生成不均匀的样本数量分布
        proportions = np.random.dirichlet(np.repeat(1.0, self.num_clients))

        indices = list(range(total_samples))
        np.random.shuffle(indices)

        start_idx = 0
        for client_id in range(self.num_clients):
            end_idx = start_idx + int(proportions[client_id] * total_samples)
            client_indices[client_id] = indices[start_idx:end_idx]
            start_idx = end_idx

        return client_indices

    def _mixed_split(self, labels):
        """混合非IID划分策略"""
        # 结合Dirichlet和类别划分
        labels = np.array(labels)
        num_classes = len(np.unique(labels))

        # 随机选择一部分客户端使用Dirichlet分布
        dirichlet_clients = np.random.choice(
            self.num_clients,
            size=self.num_clients // 2,
            replace=False
        )

        client_indices = {i: [] for i in range(self.num_clients)}

        # 对选中的客户端使用Dirichlet分布
        temp_config = type('Config', (), {'num_clients': len(dirichlet_clients)})()
        temp_loader = FedDataLoader(temp_config)
        temp_loader.num_clients = len(dirichlet_clients)
        dirichlet_indices = temp_loader._dirichlet_split(labels, self.non_iid_alpha)

        for i, client_id in enumerate(dirichlet_clients):
            client_indices[client_id] = dirichlet_indices[i]

        # 对其他客户端使用类别划分
        remaining_clients = [i for i in range(self.num_clients) if i not in dirichlet_clients]
        remaining_labels = labels.copy()

        # 移除已分配的样本
        used_indices = set()
        for indices in client_indices.values():
            used_indices.update(indices)

        remaining_indices = [i for i in range(len(labels)) if i not in used_indices]
        remaining_labels = labels[remaining_indices]

        if len(remaining_clients) > 0:
            temp_config.num_clients = len(remaining_clients)
            temp_loader.num_clients = len(remaining_clients)
            class_indices = temp_loader._class_based_split(remaining_labels, self.classes_per_client)

            for i, client_id in enumerate(remaining_clients):
                # 将相对索引转换为绝对索引
                relative_indices = class_indices[i]
                absolute_indices = [remaining_indices[idx] for idx in relative_indices]
                client_indices[client_id] = absolute_indices

        return client_indices

    def _iid_split(self, total_samples):
        """IID均匀划分"""
        indices = list(range(total_samples))
        np.random.shuffle(indices)

        client_indices = {}
        samples_per_client = total_samples // self.num_clients

        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            if client_id == self.num_clients - 1:  # 最后一个客户端获得剩余样本
                end_idx = total_samples
            client_indices[client_id] = indices[start_idx:end_idx]

        return client_indices

    def _compute_dataset_stats(self, client_indices, labels):
        """计算数据集统计信息"""
        labels = np.array(labels)
        num_classes = len(np.unique(labels))

        self.dataset_stats = {
            'total_samples': len(labels),
            'num_classes': num_classes,
            'num_clients': self.num_clients,
            'non_iid_method': self.non_iid_method,
            'client_stats': {}
        }

        # 计算每个客户端的统计信息
        for client_id, indices in client_indices.items():
            client_labels = labels[indices]
            class_counts = Counter(client_labels)

            self.dataset_stats['client_stats'][client_id] = {
                'num_samples': len(indices),
                'class_distribution': dict(class_counts),
                'num_classes': len(class_counts),
                'dominant_class': max(class_counts, key=class_counts.get) if class_counts else None
            }

        # 计算非IID程度指标
        self.dataset_stats['non_iid_metrics'] = self._compute_non_iid_metrics(client_indices, labels)

    def _compute_non_iid_metrics(self, client_indices, labels):
        """计算非IID程度指标"""
        labels = np.array(labels)
        num_classes = len(np.unique(labels))

        # 计算每个客户端的类别分布
        client_class_distributions = []
        for client_id, indices in client_indices.items():
            client_labels = labels[indices]
            class_counts = np.bincount(client_labels, minlength=num_classes)
            class_dist = class_counts / (np.sum(class_counts) + 1e-10)
            client_class_distributions.append(class_dist)

        client_class_distributions = np.array(client_class_distributions)

        # 计算KL散度
        global_dist = np.bincount(labels, minlength=num_classes)
        global_dist = global_dist / np.sum(global_dist)

        kl_divergences = []
        for client_dist in client_class_distributions:
            kl_div = np.sum(client_dist * np.log((client_dist + 1e-10) / (global_dist + 1e-10)))
            kl_divergences.append(kl_div)

        # 计算Jensen-Shannon散度
        js_divergences = []
        for i in range(len(client_class_distributions)):
            for j in range(i + 1, len(client_class_distributions)):
                dist1 = client_class_distributions[i]
                dist2 = client_class_distributions[j]
                m = (dist1 + dist2) / 2
                js_div = 0.5 * np.sum(dist1 * np.log((dist1 + 1e-10) / (m + 1e-10))) + \
                         0.5 * np.sum(dist2 * np.log((dist2 + 1e-10) / (m + 1e-10)))
                js_divergences.append(js_div)

        return {
            'avg_kl_divergence': np.mean(kl_divergences),
            'max_kl_divergence': np.max(kl_divergences),
            'avg_js_divergence': np.mean(js_divergences) if js_divergences else 0,
            'client_class_entropy': [self._entropy(dist) for dist in client_class_distributions]
        }

    def _entropy(self, distribution):
        """计算分布的熵"""
        return -np.sum(distribution * np.log(distribution + 1e-10))

    def get_dataset_info(self):
        """获取数据集信息"""
        if self.dataset_name in self.dataset_configs:
            config = self.dataset_configs[self.dataset_name]
            return {
                'dataset_name': self.dataset_name,
                'num_classes': config['num_classes'],
                'input_size': config['input_size'],
                'normalization': {
                    'mean': config['mean'],
                    'std': config['std']
                }
            }
        return None

    def get_dataset_stats(self):
        """获取数据集统计信息"""
        return self.dataset_stats

    def save_dataset_partition(self, filepath):
        """保存数据集划分"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'dataset_stats': self.dataset_stats,
                'config': {
                    'dataset_name': self.dataset_name,
                    'num_clients': self.num_clients,
                    'non_iid_method': self.non_iid_method,
                    'non_iid_alpha': self.non_iid_alpha,
                    'classes_per_client': self.classes_per_client
                }
            }, f)

    def load_dataset_partition(self, filepath):
        """加载数据集划分"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.dataset_stats = data['dataset_stats']
            return data['config']

    def visualize_data_distribution(self):
        """可视化数据分布"""
        if not self.dataset_stats:
            print("No dataset statistics available")
            return

        print(f"=== Dataset: {self.dataset_name.upper()} ===")
        print(f"Total samples: {self.dataset_stats['total_samples']}")
        print(f"Number of classes: {self.dataset_stats['num_classes']}")
        print(f"Number of clients: {self.dataset_stats['num_clients']}")
        print(f"Non-IID method: {self.dataset_stats['non_iid_method']}")
        print()

        # 显示客户端统计
        print("Client Statistics:")
        for client_id, stats in self.dataset_stats['client_stats'].items():
            print(f"  Client {client_id}: {stats['num_samples']} samples, "
                  f"{stats['num_classes']} classes, "
                  f"dominant class: {stats['dominant_class']}")

        print()

        # 显示非IID指标
        if 'non_iid_metrics' in self.dataset_stats:
            metrics = self.dataset_stats['non_iid_metrics']
            print("Non-IID Metrics:")
            print(f"  Average KL divergence: {metrics['avg_kl_divergence']:.4f}")
            print(f"  Maximum KL divergence: {metrics['max_kl_divergence']:.4f}")
            print(f"  Average JS divergence: {metrics['avg_js_divergence']:.4f}")


def create_federated_dataloaders(config):
    """创建联邦学习数据加载器的便捷函数"""
    fed_loader = FedDataLoader(config)
    train_dataset, test_dataset = fed_loader.load_dataset()
    client_loaders, test_loader = fed_loader.create_federated_dataset(train_dataset, test_dataset)

    return client_loaders, test_loader, fed_loader.get_dataset_stats()