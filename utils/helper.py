import os
import random
import numpy as np
import torch
import pickle
import json
import time
import hashlib
from datetime import datetime
from collections import OrderedDict
import shutil
import logging


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id=None):
    """获取计算设备"""
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    return device


def save_model(model, filepath, additional_info=None):
    """保存模型"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }

    if additional_info:
        save_dict.update(additional_info)

    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device=None):
    """加载模型"""
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded from {filepath}")
    return checkpoint


def save_checkpoint(model, optimizer, round_num, filepath, **kwargs):
    """保存训练检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }

    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device=None):
    """加载训练检查点"""
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {filepath}, round: {checkpoint['round']}")
    return checkpoint


def save_config(config, filepath):
    """保存配置"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 转换为可序列化的字典
    config_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            config_dict[key] = value
        else:
            config_dict[key] = str(value)

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Config saved to {filepath}")


def load_config(filepath):
    """加载配置"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    # 转换为命名空间对象
    class Config:
        pass

    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)

    return config


def save_results(results, filepath):
    """保存实验结果"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    print(f"Results saved to {filepath}")


def load_results(filepath):
    """加载实验结果"""
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_experiment_dir(base_dir, experiment_name):
    """创建实验目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)

    return experiment_dir


def get_model_size(model):
    """计算模型大小"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算模型大小（MB）
    model_size = total_params * 4 / (1024 * 1024)  # 假设float32

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size
    }


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage():
    """获取内存使用情况"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB

        return {
            'gpu_memory_allocated': memory_allocated,
            'gpu_memory_reserved': memory_reserved,
            'gpu_memory_free': torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        }
    else:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'cpu_memory_used': memory.used / (1024 ** 3),
            'cpu_memory_total': memory.total / (1024 ** 3),
            'cpu_memory_percent': memory.percent
        }


def time_str():
    """获取时间字符串"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def create_hash(data):
    """创建数据哈希"""
    if isinstance(data, str):
        data = data.encode()
    elif isinstance(data, dict):
        data = json.dumps(data, sort_keys=True).encode()
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy().tobytes()

    return hashlib.md5(data).hexdigest()


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def backup_file(filepath, backup_dir=None):
    """备份文件"""
    if not os.path.exists(filepath):
        return

    if backup_dir is None:
        backup_dir = os.path.dirname(filepath)

    ensure_dir(backup_dir)

    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)

    shutil.copy2(filepath, backup_path)
    print(f"File backed up to {backup_path}")


def setup_logger(name, log_file, level=logging.INFO):
    """设置日志器"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def merge_dicts(dict1, dict2):
    """合并字典"""
    result = dict1.copy()
    result.update(dict2)
    return result


def flatten_dict(d, parent_key='', sep='_'):
    """展平嵌套字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_namespace(d):
    """字典转命名空间"""

    class Namespace:
        pass

    ns = Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, dict_to_namespace(value))
        else:
            setattr(ns, key, value)

    return ns


def average_weights(weights_list):
    """平均权重"""
    if not weights_list:
        return {}

    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = torch.stack([w[key] for w in weights_list]).mean(0)

    return avg_weights


def weighted_average_weights(weights_list, weight_factors):
    """加权平均权重"""
    if not weights_list or not weight_factors:
        return {}

    # 归一化权重
    total_weight = sum(weight_factors)
    normalized_weights = [w / total_weight for w in weight_factors]

    avg_weights = {}
    for key in weights_list[0].keys():
        weighted_tensors = [w * weights_list[i][key] for i, w in enumerate(normalized_weights)]
        avg_weights[key] = torch.stack(weighted_tensors).sum(0)

    return avg_weights


def compare_models(model1, model2):
    """比较两个模型的差异"""
    diff_dict = {}

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    for name in params1:
        if name in params2:
            diff = torch.norm(params1[name] - params2[name])
            diff_dict[name] = diff.item()

    total_diff = sum(diff_dict.values())

    return {
        'layer_differences': diff_dict,
        'total_difference': total_diff,
        'avg_difference': total_diff / len(diff_dict) if diff_dict else 0
    }


def estimate_time_remaining(current_step, total_steps, start_time):
    """估计剩余时间"""
    if current_step == 0:
        return float('inf')

    elapsed_time = time.time() - start_time
    time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    remaining_time = remaining_steps * time_per_step

    return remaining_time


def format_time(seconds):
    """格式化时间"""
    if seconds == float('inf'):
        return "Unknown"

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def print_config(config):
    """打印配置信息"""
    print("=" * 50)
    print("Configuration:")
    print("=" * 50)

    for key, value in vars(config).items():
        print(f"{key}: {value}")

    print("=" * 50)


def validate_config(config, required_keys):
    """验证配置"""
    missing_keys = []

    for key in required_keys:
        if not hasattr(config, key):
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    return True


def get_file_size(filepath):
    """获取文件大小"""
    size_bytes = os.path.getsize(filepath)

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.1f} GB"


def cleanup_old_files(directory, keep_n=5, pattern="*.pkl"):
    """清理旧文件"""
    import glob

    files = glob.glob(os.path.join(directory, pattern))
    files.sort(key=os.path.getmtime, reverse=True)

    if len(files) > keep_n:
        for file_path in files[keep_n:]:
            os.remove(file_path)
            print(f"Removed old file: {file_path}")


def progress_bar(current, total, bar_length=50):
    """进度条"""
    progress = current / total
    filled_length = int(bar_length * progress)

    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = f"{progress * 100:.1f}%"

    print(f'\r|{bar}| {percent} ({current}/{total})', end='', flush=True)

    if current == total:
        print()


def tensor_to_numpy(tensor):
    """张量转numpy"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def numpy_to_tensor(array, device=None):
    """numpy转张量"""
    if device is None:
        device = get_device()

    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)
    return array