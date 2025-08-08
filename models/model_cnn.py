import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """基础残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    """CNN模型 - 适用于MNIST/FashionMNIST"""

    def __init__(self, num_classes=10, input_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # 计算全连接层输入维度
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class CIFARCNN(nn.Module):
    """CNN模型"""

    def __init__(self, num_classes=10, input_channels=3):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    """ResNet模型"""

    def __init__(self, block, layers, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LeNet(nn.Module):
    """LeNet-5模型 - 适用于MNIST"""

    def __init__(self, num_classes=10, input_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGBlock(nn.Module):
    """VGG基础块"""

    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                    out_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):
    """VGG模型"""

    def __init__(self, num_classes=10, input_channels=3):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(input_channels, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(dataset_name, model_type='auto', num_classes=None):
    """创建适合指定数据集的模型"""
    # 数据集配置
    dataset_configs = {
        'mnist': {'num_classes': 10, 'input_channels': 1, 'input_size': 28},
        'fashionmnist': {'num_classes': 10, 'input_channels': 1, 'input_size': 28},
        'cifar10': {'num_classes': 10, 'input_channels': 3, 'input_size': 32},
        'cifar100': {'num_classes': 100, 'input_channels': 3, 'input_size': 32}
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = dataset_configs[dataset_name]
    if num_classes is not None:
        config['num_classes'] = num_classes

    # 自动选择模型
    if model_type == 'auto':
        if dataset_name in ['mnist', 'fashionmnist']:
            model_type = 'cnn'
        elif dataset_name in ['cifar10', 'cifar100']:
            model_type = 'cifar_cnn'

    # 创建模型
    if model_type == 'cnn':
        model = CNN(config['num_classes'], config['input_channels'])
    elif model_type == 'cifar_cnn':
        model = CIFARCNN(config['num_classes'], config['input_channels'])
    elif model_type == 'lenet':
        model = LeNet(config['num_classes'], config['input_channels'])
    elif model_type == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], config['num_classes'], config['input_channels'])
    elif model_type == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], config['num_classes'], config['input_channels'])
    elif model_type == 'vgg':
        model = VGG(config['num_classes'], config['input_channels'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def get_model_info(model):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'model_name': model.__class__.__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
    }


def count_parameters_by_layer(model):
    """按层统计参数数量"""
    layer_params = {}
    for name, param in model.named_parameters():
        layer_params[name] = param.numel()
    return layer_params


def initialize_weights(model, init_type='xavier'):
    """权重初始化"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def freeze_layers(model, freeze_layers):
    """冻结指定层"""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in freeze_layers):
            param.requires_grad = False


def get_feature_extractor(model, layer_name):
    """获取特征提取器"""
    layers = dict(model.named_modules())
    if layer_name not in layers:
        raise ValueError(f"Layer {layer_name} not found in model")

    feature_extractor = nn.Sequential()
    for name, module in model.named_modules():
        feature_extractor.add_module(name, module)
        if name == layer_name:
            break

    return feature_extractor


# 便捷函数
def create_mnist_model(model_type='cnn'):
    """创建MNIST模型"""
    return create_model('mnist', model_type)


def create_fashionmnist_model(model_type='cnn'):
    """创建FashionMNIST模型"""
    return create_model('fashionmnist', model_type)


def create_cifar10_model(model_type='cifar_cnn'):
    """创建CIFAR-10模型"""
    return create_model('cifar10', model_type)


def create_cifar100_model(model_type='cifar_cnn'):
    """创建CIFAR-100模型"""
    return create_model('cifar100', model_type)