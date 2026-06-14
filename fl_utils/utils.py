import torch, os, sys
import numpy as np
from torchvision import datasets, transforms
import random
from collections import OrderedDict
import logging
import time

import colorlog
import torch

from fl_utils.parameters import Params
np.random.seed(42)
random.seed(42)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config):
    import logging
    log_dir = config.folder_path
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy


def compute_model_diff(model_a, model_b):
    diff = OrderedDict()
    for name, data in model_a.state_dict().items():
        if name in model_b.state_dict():
            diff[name] = data - model_b.state_dict()[name]
    return diff


def flatten_update(update_dict, layer_names=None):
    """
    展平特定层并返回np一维向量

    Args:
        update_dict (_type_): _description_
        layer_names (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    flat = []
    for name, data in update_dict.items():
        if layer_names and not any(l in name for l in layer_names):
            continue
        if 'num_batches_tracked' in name:
            continue
        flat.append(data.cpu().numpy().flatten())
    if flat:
        return np.concatenate(flat)
    return np.array([])


def get_layer_name_for_dataset(dataset):
    if dataset in ['cifar10', 'cifar100']:
        return 'linear'
    elif dataset == 'tiny-imagenet-200':
        return 'fc'
    else:
        return 'fc2'


def compute_cosine_similarity_matrix(updates):
    n = len(updates)
    cs = np.zeros((n, n))
    for i in range(n):
        norm_i = np.linalg.norm(updates[i]) + 1e-8
        for j in range(n):
            norm_j = np.linalg.norm(updates[j]) + 1e-8
            cs[i, j] = np.dot(updates[i], updates[j]) / (norm_i * norm_j)
    return cs


def gap_statistic(data, num_sampling=5, K_max=10, n=None):
    from sklearn.cluster import KMeans
    if n is None:
        n = len(data)
    W_k = np.zeros(K_max)
    W_kb = np.zeros((num_sampling, K_max))
    sk = np.zeros(K_max)

    for k in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
        kmeans.fit(data)
        W_k[k - 1] = np.sum([np.sum((data[kmeans.labels_ == i] -
                                      kmeans.cluster_centers_[i]) ** 2)
                             for i in range(k)])

        for b in range(num_sampling):
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            random_data = np.random.uniform(min_vals, max_vals, size=data.shape)
            kmeans_b = KMeans(n_clusters=k, init='k-means++', n_init=10)
            kmeans_b.fit(random_data)
            W_kb[b, k - 1] = np.sum([np.sum((random_data[kmeans_b.labels_ == i] -
                                              kmeans_b.cluster_centers_[i]) ** 2)
                                     for i in range(k)])

        if k == 1:
            sk[k - 1] = 0
        else:
            sk[k - 1] = np.std(np.log(W_kb[:, k - 1]))

    Gap_k = np.mean(np.log(W_kb), axis=0) - np.log(W_k)

    for k in range(1, K_max):
        if Gap_k[k - 1] >= Gap_k[k] - sk[k]:
            return k

    return K_max


def get_ood_data(dataset):

    transform_ood = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar100':
        ood_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_ood)
        num_classes = 100
    elif dataset == 'cifar10':
        ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
        num_classes = 10
    elif dataset == 'tiny-imagenet-200':
        ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
        num_classes = 200
    else:
        ood_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform_ood)
        num_classes = 10

    ood_data_sample_lens = 1000
    indices = list(range(ood_data_sample_lens))
    ood_dataloader = torch.utils.data.DataLoader(ood_dataset,
                                                 batch_size=64,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                                 drop_last=True)
    ood_datalist = list(ood_dataloader)
    ood_datalist_shape = ood_data_sample_lens // 64 * 64

    assigned_labels = np.array([i for i in range(num_classes)] * \
                               (ood_datalist_shape // num_classes) + [i for i in range(
        ood_datalist_shape % num_classes)])
    np.random.shuffle(assigned_labels)
    assigned_labels = assigned_labels.reshape(ood_data_sample_lens // 64 ,64)

    for batch_id, batch in enumerate(ood_datalist):
        data, targets = batch
        for ind in range(len(targets)):
            targets[ind] = assigned_labels[batch_id][ind]
    ood_dataloader = iter(ood_datalist)
    ood_test_loader = iter(ood_datalist)
    return ood_dataloader, ood_test_loader

# def _get_ood_dataloader(dataset):
#     """
#     生成一个包含分布外数据(OOD)的数据加载器, 和 attacker.py -> getdata()/getdata2() 函数不同, 二者采用随机数据做OOD, 这里采用不同的数据集
#     """
#     r'''
#     sample limited ood data as open set noise
#         '''

#     # 根据主任务数据集(dataset)的类型设置 OOD 数据的预处理标准. 如果主任务是单通道的 MNIST, 则使用单通道归一化；否则使用三通道(RGB)归一化
#     if dataset == 'mnist':
#         transform_ood = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ])
#     else:
#         transform_ood = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])

#     # 如果主任务是 CIFAR100, OOD 就选 CIFAR10
#     if dataset == 'cifar100':
#         ood_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_ood)
#         num_classes = 100
#     elif dataset == 'cifar10':
#         ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
#         num_classes = 10
#     elif dataset == 'tiny-imagenet-200':
#         ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
#         num_classes = 200
#     else:
#         ood_dataset = datasets.FashionMNIST("../data", train=True, download=True, transform=transform_ood)
#         num_classes = 10


#     ood_data_sample_lens = 500
#     # indices = list(range(ood_data_sample_lens))
#     # 从 OOD 数据集中随机抽取 500 个样本
#     # 话说下面的做法怎么这么眼熟, 好像两个getdata/getdata2函数差不多就是这么干的
#     indices = random.sample(range(len(ood_dataset)), ood_data_sample_lens)
#     ood_dataloader =  torch.utils.data.DataLoader(ood_dataset,
#                                 batch_size=64,
#                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
#                                 drop_last=True)
#     ood_datalist = list(ood_dataloader)
#     ood_datalist_shape = ood_data_sample_lens//64 * 64
#     assigned_labels = np.array([i for i in range(num_classes)] * \
#             (ood_datalist_shape//num_classes) + [i for i in range(ood_datalist_shape%num_classes)])
#     np.random.shuffle(assigned_labels)
#     assigned_labels = assigned_labels.reshape(ood_data_sample_lens//64, 64)
#     for batch_id, batch in enumerate(ood_datalist):
#         data, targets = batch
#         for ind in range(len(targets)):
#             targets[ind] = assigned_labels[batch_id][ind]
#     ood_dataloader=iter(ood_datalist)
#     return ood_dataloader

def _get_ood_dataloader(dataset_name, num_classes=10):
    """
    生成更具鲁棒性的 OOD DataLoader
    """
    # 1. 确定主任务需要的通道数
    # GTSRB, CIFAR, Tiny-ImageNet 通常是 3 通道；MNIST 是 1 通道
    main_channels = 1 if dataset_name == 'mnist' else 3
    # 统一尺寸（根据你的模型输入调整，假设为 32x32 或 28x28）
    target_size = 28 if dataset_name == 'mnist' else 32

    # 2. 构建鲁棒的 Transform
    transform_list = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ]

    if main_channels == 3:
        # 如果主任务是彩色，但 OOD 是灰度图，强制转为 3 通道
        transform_list.insert(0, transforms.Lambda(lambda x: x.convert("RGB")))
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    else:
        # 如果主任务是单通道，强制转为 L (灰度)
        transform_list.insert(0, transforms.Lambda(lambda x: x.convert("L")))
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

    transform_ood = transforms.Compose(transform_list)

    # 3. 选择 OOD 数据集
    data_path = "../data"
    if dataset_name == 'cifar100':
        ood_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_ood)
    elif dataset_name == 'cifar10':
        ood_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_ood)
    elif 'tiny-imagenet' in dataset_name:
        ood_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_ood)
    else:
        # 对于 GTSRB 或其他，默认选 FashionMNIST，但经过上面的 transform 会自动转为 3 通道
        ood_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform_ood)

    # 4. 采样与标签重分配
    ood_sample_size = 500
    batch_size = 64
    
    # 随机采样索引
    indices = random.sample(range(len(ood_dataset)), ood_sample_size)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    
    # 使用 DataLoader 提取数据
    temp_loader = torch.utils.data.DataLoader(
        ood_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
    )

    processed_batches = []
    
    # 计算总共能组成多少个完整 batch
    num_batches = len(temp_loader)
    total_allowed_samples = num_batches * batch_size

    # 生成打乱的伪标签 (确保覆盖所有类别)
    pseudo_labels = np.array([i % num_classes for i in range(total_allowed_samples)])
    np.random.shuffle(pseudo_labels)
    pseudo_labels = pseudo_labels.reshape(num_batches, batch_size)

    # 5. 修改标签并存入列表
    for i, (data, _) in enumerate(temp_loader):
        new_targets = torch.from_numpy(pseudo_labels[i]).long()
        processed_batches.append((data, new_targets))

    return iter(processed_batches)

class NoiseDataset(torch.utils.data.Dataset):

    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        noise = noise.cuda()
        return noise, 0


def _get_ood_testloader(dataset):
    r'''
    sample limited ood data as open set noise
        '''

    transform_ood = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ood_data_sample_lens = 1000

    if dataset == 'cifar100':
        ood_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_ood)
        num_classes = 100
    elif dataset == 'cifar10':
        ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
        # ood_dataset = NoiseDataset(size=(3,32,32), num_samples=ood_data_sample_lens)
        num_classes = 10
    elif dataset == 'tiny-imagenet-200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_ood = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
        num_classes = 200
    else:
        ood_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform_ood)
        num_classes = 10


    # indices = list(range(ood_data_sample_lens))
    indices = random.sample(range(len(ood_dataset)), ood_data_sample_lens)
    ood_dataloader =  torch.utils.data.DataLoader(ood_dataset,
                                batch_size=64,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                drop_last=True)
    ood_datalist = list(ood_dataloader)
    ood_datalist_shape = ood_data_sample_lens//64 * 64
    assigned_labels = np.array([i for i in range(num_classes)] * \
            (ood_datalist_shape//num_classes) + [i for i in range(ood_datalist_shape%num_classes)])
    np.random.shuffle(assigned_labels)
    assigned_labels = assigned_labels.reshape(ood_data_sample_lens//64, 64)

    for batch_id, batch in enumerate(ood_datalist):
        data, targets = batch
        for ind in range(len(targets)):
            targets[ind] = assigned_labels[batch_id][ind]
        # print(targets)
    ood_dataloader=iter(ood_datalist)
    return ood_dataloader




def record_time(params: Params, t=None, name=None):
    if t and name and params.save_timing == name or params.save_timing is True:
        torch.cuda.synchronize()
        params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))


# 按表格打印数据
def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data


def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)
