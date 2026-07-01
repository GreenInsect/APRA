import sys

sys.path.append("../")
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from fl_utils.thrd import design_indicator, check_indicator, add_noise
import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from copy import deepcopy
from collections import defaultdict, OrderedDict
import random
from torch.utils.data import Dataset
import numpy as np
import copy
import logging
logger = logging.getLogger('logger')


class RandomImagesDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

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

class Attacker:
    indicators: dict = None
    alpha = 0.5

    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.rate1 = self.helper.config["trigger_loss_rate"]
        self.rate2 = self.helper.config["main_loss_rate"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.helper.config["mia_class_method"] == "random":
            self.getdata_random()
        else:
            self.getdata()
        
        # 1. 动态获取数据集的真实长宽（完美适配 28 或 32）
        # 如果 helper 中没有封装真实的图像长宽变量，可以通过判断数据集名称来决策
        dataset_name = self.helper.config["dataset"].lower()
        if 'mnist' in dataset_name:
            self.img_size = 28  
        elif dataset_name in ['cifar10', 'cifar100', 'emnist', 'gtsrb']:
            self.img_size = 32
        else:
            self.img_size = 224

        # 2. 统一初始化
        self.trigger = torch.ones((1, self.helper.channel, self.img_size, self.img_size), requires_grad=False, device='cuda') * 0.5
        if self.helper.config["trigger_pattern"] == 'fixed':
            self.trigger = torch.ones((1, self.helper.channel, self.img_size, self.img_size), requires_grad=False, device='cuda') * self.helper.config["trigger_max"]
        
        self.mask = torch.zeros_like(self.trigger)
        trig_s = self.helper.config["trigger_size"]
        self.mask[:, :, 2:2+trig_s, 2:2+trig_s] = 1
        self.mask = self.mask.cuda()
        self.trigger0 = copy.deepcopy(self.trigger)      
        
        self.setup()           
          

    # def setup(self):
    #     self.handcraft_rnds = 0
    #     if self.helper.config["dataset"] == 'cifar10' or self.helper.config["dataset"] == 'cifar100':
    #         self.trigger = torch.ones((1, 3, 32, 32), requires_grad=False, device='cuda') * 0.5
    #     elif self.helper.config['dataset'] == 'mnist':
    #         self.trigger = torch.ones((1, 1, 28, 28), requires_grad=False, device='cuda') * 0.5
    #     else:
    #         self.trigger = torch.ones((1, 3, 224, 224), requires_grad=False, device='cuda') * 0.5

    #     self.mask = torch.zeros_like(self.trigger)

    #     self.mask[:, :, 2:2 + self.helper.config['trigger_size'], 2:2 + self.helper.config['trigger_size']] = 1

    #     transform_ood = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     num_images = 1000 # 2000 1000
    #     # self.ood_dataset 是 噪声数据集, 作为分布外数据集使用
    #     self.ood_dataset = NoiseDataset(size=(3, 32, 32), num_samples=num_images)

    #     # if self.helper.config["dataset"] == 'cifar100':
    #     #     self.ood_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_ood)
    #     # elif self.helper.config["dataset"] == 'cifar10':
    #     #     self.ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
    #     # elif self.helper.config["dataset"] == 'tiny-imagenet-200':
    #     #     self.ood_dataset = datasets.CIFAR100("../data", train=True, download=True, transform=transform_ood)
    #     # else:
    #     #     logger.error("Don't support this dataset")

    # def setup(self):
    #     self.handcraft_rnds = 0
    #     dataset_name = self.helper.config["dataset"].lower()

    #     if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    #         self.trigger = torch.ones((1, 3, 32, 32), requires_grad=False, device='cuda') * 0.5
    #         ood_size = (3, 32, 32)
    #     elif 'mnist' in dataset_name:
    #         self.trigger = torch.ones((1, 1, 28, 28), requires_grad=False, device='cuda') * 0.5
    #         ood_size = (1, 28, 28)
    #     elif dataset_name == 'emnist':
    #         self.trigger = torch.ones((1, 1, 32, 32), requires_grad=False, device='cuda') * 0.5
    #         ood_size = (1, 32, 32)            
    #     elif 'gtsrb' in dataset_name:
    #         self.trigger = torch.ones((1, 3, 32, 32), requires_grad=False, device='cuda') * 0.5
    #         ood_size = (3, 32, 32)
    #     else:
    #         # 针对 Tiny-ImageNet 等数据集
    #         self.trigger = torch.ones((1, 3, 224, 224), requires_grad=False, device='cuda') * 0.5
    #         ood_size = (3, 224, 224)
            
    #     # TODO 待修改触发器设置

    #     self.mask = torch.zeros_like(self.trigger)
        
    #     trig_s = self.helper.config['trigger_size']
    #     self.mask[:, :, 2:2 + trig_s, 2:2 + trig_s] = 1

    #     transform_ood = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     num_images = 1000 
    #     self.ood_dataset = NoiseDataset(size=ood_size, num_samples=num_images)
        # if self.helper.config["attacker_method"] == "reba":
        #     self.trigger = torch.ones((1,self.helper.channel,32,32), requires_grad=False, device = 'cuda')*0.5
        #     if self.helper.config["trigger_pattern"] == 'fixed':
        #         self.trigger = torch.ones((1, self.helper.channel, 32, 32), requires_grad=False,device='cuda') * self.helper.config.trigger_max
        #     self.mask = torch.zeros_like(self.trigger)
        #     self.mask[:, :, 2:2+self.helper.config["trigger_size"], 2:2+self.helper.config["trigger_size"]] = 1
        #     self.mask = self.mask.cuda()
        #     self.trigger0 = copy.deepcopy(self.trigger)
        
    def setup(self):
        self.handcraft_rnds = 0
        dataset_name = self.helper.config["dataset"].lower()

        if dataset_name == 'cifar10' or dataset_name == 'cifar100':
            ood_size = (3, 32, 32)
        elif 'mnist' in dataset_name:
            # 根据真实的长宽动态分配给 OOD 噪声数据集
            ood_size = (1, self.img_size, self.img_size)
        elif dataset_name == 'emnist':
            ood_size = (1, 32, 32)            
        elif 'gtsrb' in dataset_name:
            ood_size = (3, 32, 32)
        else:
            ood_size = (3, 224, 224)
            
        # 3. 重新规范化配置，杜绝硬编码覆盖
        self.trigger = torch.ones((1, self.helper.channel, self.img_size, self.img_size), requires_grad=False, device='cuda') * 0.5
        if self.helper.config["trigger_pattern"] == 'fixed':
            self.trigger = torch.ones((1, self.helper.channel, self.img_size, self.img_size), requires_grad=False, device='cuda') * self.helper.config["trigger_max"]
            
        self.mask = torch.zeros_like(self.trigger)
        trig_s = self.helper.config['trigger_size']
        self.mask[:, :, 2:2 + trig_s, 2:2 + trig_s] = 1
        self.mask = self.mask.cuda()

        transform_ood = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        num_images = 1000 
        self.ood_dataset = NoiseDataset(size=ood_size, num_samples=num_images)
        
        if self.helper.config["attacker_method"] == "reba":
            self.trigger = torch.ones((1, self.helper.channel, self.img_size, self.img_size), requires_grad=False, device='cuda') * 0.5
            if self.helper.config["trigger_pattern"] == 'fixed':
                self.trigger = torch.ones((1, self.helper.channel, self.img_size, self.img_size), requires_grad=False, device='cuda') * self.helper.config.trigger_max
            self.mask = torch.zeros_like(self.trigger)
            self.mask[:, :, 2:2 + trig_s, 2:2 + trig_s] = 1
            self.mask = self.mask.cuda()
            self.trigger0 = copy.deepcopy(self.trigger)        

    def generate_random_label(self):
        """
        攻击者用来生成一个非目标的随机标签
        """
        number = random.randint(1, self.helper.num_classes)
        while number == self.helper.config["target_class"]:
            number = random.randint(1, self.helper.num_classes)
        return number

    # def getdata(self):
    #     """
    #     生成纯随机的噪声图像, 并挂载到 self.miadate 上 
    #     """
    #     num_images = 1000  # 2000 1000


    #     dataset_name = self.helper.config.get("dataset", "").lower()
        
    #     if 'cifar' in dataset_name:
    #         size = 32
    #         num_channels = 3
    #     elif 'mnist' in dataset_name:
    #         size = 28
    #         num_channels = 1
    #     elif 'imagenet' in dataset_name: # 捕获 tiny-imagenet-200
    #         size = 224  # Tiny-ImageNet 标准尺寸
    #         num_channels = 3
    #     else:
    #         # 兜底方案：从 helper 的数据加载器中自动获取
    #         size = 64 
    #         num_channels = 3
    #         print(f"[*] klog Warning: Unknown dataset {dataset_name}, defaulting to size 64")


    #     label = self.generate_random_label()
    #     print(f"生成的随机图片的标签为:{label}")
    #     # 创建一个形状为 (1000,) 的张量, 全部填充为 label
    #     labels = torch.full((num_images,), label)  # 所有图片的标签都是label

    #     random_images = torch.rand((num_images, num_channels, size, size))  # 生成随机图片(均匀分布)

    #     # mu = 0  # 均值
    #     # sigma = 1  # 标准差
    #     # random_images = torch.randn((num_images, num_channels, size, size))
    #     # # 正态分布
    #     # random_images = mu + sigma * random_images

    #     random_dataset = RandomImagesDataset(random_images, labels)
    #     num_clients = self.helper.config["num_adversaries"]
    #     images_per_client = num_images // num_clients
    #     # 列表推导式生成索引范围列表
    #     client_indices = [list(range(i * images_per_client, (i + 1) * images_per_client)) for i in range(num_clients)]
    #     # 为每个攻击者创建 DataLoader , self.miadate 是 噪声数据加载器列表
    #     self.miadate = [DataLoader(Subset(random_dataset, indices), batch_size=32, shuffle=True) for indices in
    #                     client_indices]
    #     self.mia_test_loader = DataLoader(random_dataset, batch_size=64, shuffle=False)

    def getdata(self):
        """
        生成纯随机的噪声图像, 并挂载到 self.miadate 上 
        """
        num_images = 1000  # 2000 1000

        dataset_name = self.helper.config.get("dataset", "").lower()
        
        # --- 修改部分：增强对不同数据集尺寸和通道的判断 ---
        if 'cifar' in dataset_name:
            size = 32
            num_channels = 3
        elif 'mnist' in dataset_name:
            size = 28
            num_channels = 1
        elif 'imagenet' in dataset_name: # 捕获 tiny-imagenet-200
            size = 224  # Tiny-ImageNet 标准尺寸
            num_channels = 3
        elif 'gtsrb' in dataset_name:
            # 针对 GTSRB 的处理，确保为 3 通道，尺寸通常为 32
            size = 32
            num_channels = 3
        else:
            # 兜底方案：从 helper 的数据加载器中自动获取
            size = 64 
            num_channels = 3
            print(f"[*] klog Warning: Unknown dataset {dataset_name}, defaulting to size 64")
        # ----------------------------------------------

        # if self.helper.config["mia_class_method"] == "nobackdoor_random":
        #     # 生成一个非目标的随机标签
        #     label = self.generate_random_label()
        # elif self.helper.config["mia_class_method"] == "backdoor":
        #     label = self.helper.config["target_class"]
        # self.helper.config["mia_class"] = label
        # print(f"生成的随机图片的标签为:{label}")
        # 创建一个形状为 (1000,) 的张量, 全部填充为 label
        label = self.helper.config["mia_class"]
        labels = torch.full((num_images,), label)  # 所有图片的标签都是label

        random_images = torch.rand((num_images, num_channels, size, size))  # 生成随机图片(均匀分布)

        # mu = 0  # 均值
        # sigma = 1  # 标准差
        # random_images = torch.randn((num_images, num_channels, size, size))
        # # 正态分布
        # random_images = mu + sigma * random_images

        random_dataset = RandomImagesDataset(random_images, labels)
        num_clients = self.helper.config["num_adversaries"]
        images_per_client = num_images // num_clients
        # 列表推导式生成索引范围列表
        client_indices = [list(range(i * images_per_client, (i + 1) * images_per_client)) for i in range(num_clients)]
        # 为每个攻击者创建 DataLoader , self.miadate 是 噪声数据加载器列表
        self.miadate = [DataLoader(Subset(random_dataset, indices), batch_size=32, shuffle=True) for indices in
                        client_indices]
        self.mia_test_loader = DataLoader(random_dataset, batch_size=64, shuffle=False)    

    def getdata_random(self):
        """
        与 getdata 保持相同的数据构造逻辑, 仅将标签改为逐样本随机生成
        """
        num_images = 1000

        dataset_name = self.helper.config.get("dataset", "").lower()

        if 'cifar' in dataset_name:
            size = 32
            num_channels = 3
        elif 'mnist' in dataset_name:
            size = 28
            num_channels = 1
        elif 'imagenet' in dataset_name:
            size = 224
            num_channels = 3
        elif 'gtsrb' in dataset_name:
            size = 32
            num_channels = 3
        else:
            size = 64
            num_channels = 3
            print(f"[*] klog Warning: Unknown dataset {dataset_name}, defaulting to size 64")

        random_images = torch.rand((num_images, num_channels, size, size))
        labels = torch.randint(0, self.helper.num_classes, (num_images,), dtype=torch.long)

        random_dataset = RandomImagesDataset(random_images, labels)
        num_clients = self.helper.config["num_adversaries"]
        images_per_client = num_images // num_clients
        client_indices = [list(range(i * images_per_client, (i + 1) * images_per_client)) for i in range(num_clients)]
        self.miadate = [DataLoader(Subset(random_dataset, indices), batch_size=32, shuffle=True) for indices in
                        client_indices]
        self.mia_test_loader = DataLoader(random_dataset, batch_size=64, shuffle=False)

    def getdata2(self):
        """
        利用已有的分布外(OOD)数据集(预定义的噪声集 self.ood_dataset ), 并对其进行标签重分布
        """
        # 采样总长度
        ood_data_sample_lens = 1000
        # 攻击目标类别, 需排除
        target_label = self.helper.config["target_class"]
        indices = list(range(ood_data_sample_lens))
        # self.ood_dataset 是噪声集, 作为预先加载的分布外数据集
        ood_dataloader = torch.utils.data.DataLoader(self.ood_dataset,
                                                     batch_size=64,
                                                     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                                     drop_last=True)
        # 将加载器转为列表
        ood_datalist = list(ood_dataloader)
        # TOANSWER 1000 == 64 * 64 ? 
        ood_datalist_shape = ood_data_sample_lens // 64 * 64

        # 生成除了目标类别以外的所有合法标签列表
        valid_range = [i for i in range(self.helper.num_classes) if i != target_label]

        # 构造一组均匀分布的假标签. 通过重复 valid_range 并拼接余数部分, 确保每个非目标类出现的次数尽量相等
        assigned_labels = np.array(
            [i for i in valid_range] * (ood_datalist_shape // (self.helper.num_classes - 1)) +
            [i for i in valid_range[:ood_datalist_shape % (self.helper.num_classes - 1)]]
        )
        # 随机打乱标签顺序
        np.random.shuffle(assigned_labels)
        # 重构为 [BatchSize, 64]
        assigned_labels = assigned_labels.reshape(ood_data_sample_lens // 64,64)
        for batch_id, batch in enumerate(ood_datalist):
            data, targets = batch
            for ind in range(len(targets)):
                # 将原本的 OOD 标签改为上面生成的平衡假标签
                targets[ind] = assigned_labels[batch_id][ind]
        ood_dataloader = iter(ood_datalist)
        self.mia_test_loader = ood_dataloader
        self.mia_ood_datalist = ood_datalist
        num_clients = self.helper.config["num_adversaries"]
        # self.miadate 是数据加载器列表, 每个元素都是指向同一个修改后 OOD 列表的迭代器
        self.miadate = [ood_dataloader for _ in range(num_clients)]

    def save_data(self, random_images, labels):
        data_to_save = {'images': random_images, 'labels': labels}
        torch.save(data_to_save, 'data.pt')

    def dirichlet_distribution_noniid(self, train_dataset, num_clients, alpha):
        class_indices = [[] for _ in range(10)]
        for idx, (image, label) in enumerate(train_dataset):
            class_indices[label].append(idx)
        client_indices = [[] for _ in range(num_clients)]
        for class_idx in class_indices:
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            client_split = np.split(class_idx, proportions)
            for client_idx, client_split_indices in enumerate(client_split):
                client_indices[client_idx].extend(client_split_indices)
        return client_indices

    def check(self, model, epoch):
        if epoch <= self.helper.config["poison_epochs"]/2:
            thre = 0.3
        else:
            thre = 0.5
        mia_model = copy.deepcopy(model)
        mia_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # for images, labels in iter(self.mia_ood_datalist):
            for images, labels in self.mia_test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = mia_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if(accuracy>=thre):
            return 1
        else:
            return 0

    def noise(self, model, sampled_participants, epoch, weight_accumulator_by_client):
        """
        self.attacker.noise(self.helper.global_model, sampled_participants, epoch, weight_accumulator_by_client)\n
        随机挑选一个攻击者的更新量注入噪声
        """
        if epoch >= self.helper.config["poison_epochs"]:
            return
        # TOANSWER poison_epochs 和 attack_epoch 分别是啥意思? FCBA 的特殊性决定了它需要在多轮进行分布式攻击?   
        if self.helper.config["sample_method"] == 'fcba' and epoch not in self.helper.config["attack_epoch"]:
            self.previous_global_model = deepcopy(model)
            return
        # noise_model 是全局模型(未训练)的副本(名字叫 noise_model, 但是是当 global_model 用的)
        noise_model = copy.deepcopy(model)
        num = self.helper.config["num_adversaries"]
        choose_id = np.random.randint(num)
        # 见 fler.py -> train_once() -> self.helper.save_update(model=single_wa, userID=participant_id/i) 两行 
        update_name = '{0}/saved_updates/update_{1}.pth'.format(self.helper.config["folder_path"], choose_id)
        # 随机选择一个攻击者的恶意训练过后的模型增量
        backdoor_update = torch.load(update_name)
        dl = self.helper.train_data[choose_id]
        # 基于全局模型在当前本地数据 dl 上训练一个纯良性的模型副本
        benign_model = self.get_ben_model(noise_model, dl)
        benign_update = self.get_fl_update(benign_model, noise_model)
        # 见论文 P27 DOBA 的步骤4, P30 NDOBA 的步骤4, 即输出层噪声添加环节(步骤 4:输出层添加噪声. 结合投毒指示反馈,自适应的在输出层添加噪声,避免模型  更新过度集中)
        if 'cifar' in self.helper.config["dataset"]:
            ind_layer = 'layer4.1.conv2.weight'
            layer_name = 'linear'
        elif self.helper.config['dataset'] == 'tiny-imagenet-200':
            layer_name = 'fc'
        else:
            ind_layer = 'conv2.weight'
            layer_name = 'fc2'

        # 机制调整(暂时注释)
        # if epoch > 1 and epoch < self.helper.config["poison_epochs"]:
            # global_update = self.get_fl_update(noise_model, self.previous_global_model)
            # ac = self.check(noise_model, epoch)
            # if ac == 1:
            #     # print(f"feedback: success")
            #     self.rate1 = self.rate1 + 0.01
            #     if self.rate1 > 0.85:
            #         self.rate1 = 0.85
            # else:
            #     if epoch >= self.helper.config["poison_epochs"]/4:
            #         # print(f"feedback: fail")
            #         self.alpha = self.alpha + 0.01
            #         if self.alpha > 0.85:
            #             self.alpha = 0.85
                # else:
                #     print(f"feedback: can not judge")

        add_noise(self.helper, sampled_participants, weight_accumulator_by_client, backdoor_update, layer_name,
                      noise_model, self.alpha)
        # TOANSWER 这玩意又是不知道干什么用的, 代码里只有上面注释掉的部分用到了, 还是用来得到 get_fl_update 的
        self.previous_global_model = deepcopy(model)

    def get_fl_update(self, local_model, global_model):
        """
        计算并返回模型参数更新差值
        """
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if name == 'decoder.weight' or '__' in name:
                continue
            local_update[name] = (data - global_model.state_dict()[name])
        return local_update

    def get_adv_model(self, model, dl, trigger, mask):
        """
        模拟当前模型在植入后门后的下一步状态(即生成一个"对抗模型"), 并计算这个演化模型与原始模型在梯度方向上的相似度\n
        model: 下发到本地的模型\n
        Return:\n
            adv_model: 模拟更新后的模型\n
            sim_sum/sim_count: model和adv_model所有卷积层梯度的平均余弦相似度(标量 float)
        """       
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        # 使用毒化数据更新adv_model的权重进行训练, 生成 "对抗模型"?
        for _ in range(self.helper.config['dm_adv_epochs']):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        # 用于累加卷积层的梯度的余弦相似度
        sim_sum = 0.
        sim_count = 0.
        # 计算两个向量之间的余弦夹角
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    """
    FOCUS 
                search_trigger_re                             search_trigger   
    参考对象:    未来的良性模型 (G_(stop_t))                     后门输入训练后的中毒/对抗模型(adv_model)    
                模拟"如果没有攻击, 模型本该有的特征"               模拟"全局模型被攻击并更新后, 未来可能的权值"
    损失函数:    Loss = a * L_backdoor + (1 - a) * L_trigger    Loss = L_(local_model) + noise_loss_lambda * ∑ (adv_w * L_(adv_model)) / dm_adv_model_count  
    """


    def search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        """
        model: 下发到本地的模型\n
        确保触发器不仅要在当前模型上攻击成功, 还要保证其在经过后门输入训练后的对抗模型上依旧有效\n
        """
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        # 存储生成的对抗模型列表
        adv_models = []
        # 存储每个对抗模型的相似度权重(sim_sum/sim_count: 所有卷积层梯度的平均余弦相似度(标量 float))
        adv_ws = []

        def val_asr(model, dl, t, m):
            """
            测试毒化数据(触发器)在模型上的攻击效果\n
            应该是用于在优化过程中实时监控触发器的效果\n
            Return:\n
                asr, total_loss
            """
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    labels[:] = self.helper.config['target_class']
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config['trigger_lr']
        # trigger_outter_epochs: 20 #50 200 20
        K = self.helper.config['trigger_outter_epochs']
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr=alpha*10, weight_decay=0)
        for iter in range(K):
            # 
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            """
            dm_adv_K: 1
            dm_adv_epochs: 5
            dm_adv_model_count: 1
            TOANSWER 这里为什么设置得那么小
            """
            # 生成对抗模型
            if iter % self.helper.config['dm_adv_K'] == 0 and iter != 0:
                if len(adv_models) > 0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.helper.config['dm_adv_model_count']):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
                    
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.helper.config['target_class']
                outputs = model(inputs)
                # 后门输入在local_model上的损失
                loss = ce_loss(outputs, labels)
                # TOANSWER 式子的原理? 为了迫使触发器在所有对抗模型上都能达到目标?
                # 即: 最终损失 Loss = L_(local_model) + noise_loss_lambda * ∑ (adv_w * L_(adv_model)) / dm_adv_model_count
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        # 后门输入在生成的对抗模型上的损失
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.helper.config['noise_loss_lambda']*adv_w*nm_loss/self.helper.config['dm_adv_model_count']
                        else:
                            # yaml : noise_loss_lambda: 0.01
                            loss += self.helper.config['noise_loss_lambda']*adv_w*nm_loss/self.helper.config['dm_adv_model_count']
                if loss != None:
                    loss.backward()
                    # TOANSWER 这一行把 t.grad 张量中的所有数值求和, 意义是什么?
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()

    # ben -> Benign  dl -> DataLoader 
    def get_ben_model(self, model, dl):
        """
        在本地数据集上对当前模型进行短期的正常训练从而获得一个良性模型, 应该是指论文中的G_(stop_t)
        """
        ben_model = copy.deepcopy(model)
        ben_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        ben_opt = torch.optim.SGD(ben_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.helper.config['dm_adv_epochs']):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = ben_model(inputs)
                loss = ce_loss(outputs, labels)
                ben_opt.zero_grad()
                loss.backward()
                ben_opt.step()
        return ben_model

    def search_trigger_re(self, model, dl, epoch=0):
        """
        通过对抗训练生成满足论文中公式的简单后门触发器, 损失函数满足: Loss = a * L_backdoor + (1 - a) * L_trigger\n
        即需要保证后门在未被攻击的未来模型上的有效性, 并需要满足当前全局模型与模拟出的未来未中毒模型之间的差异较小
        """
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config['trigger_lr']
        K = self.helper.config['trigger_outter_epochs']
        # K = 10
        t = self.trigger.clone()
        m = self.mask.clone()
        count = 0
        # trigger_rate:{self.rate1} == trigger_loss_rate: 0.8
        logging.info(f"epoch:{epoch} trigger_rate:{self.rate1}")

        for iter in range(K):
            ben_model = self.get_ben_model(model, dl)
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, be_inputs, labels, be_labels = inputs.cuda(), inputs.cuda(),labels.cuda(), labels.cuda()
                # 
                inputs = t * m + (1 - m) * inputs
                labels[:] = self.helper.config['target_class']
                # model = self.helper.local_model
                outputs = model(inputs)
                # 这里的 loss 就是 中 L_backdoor = L_ce(x*, y*;Gt)表示触发器在当前全局模型 Gt (下发到客户端的) 的攻击任务损失
                loss = ce_loss(outputs, labels)
                # 这里的 diff_trigger_loss 就是 L_trigger, 即 model 指 Gt, ben_model 值 G_stop_t
                diff_trigger_loss = self.trigger_loss_diff(model, ben_model, inputs)
                # Loss = a * L_backdoor + (1 - a) * L_trigger, 见论文 P25 (3.1)
                loss = self.rate1 * diff_trigger_loss + (1 - self.rate1) * loss
                if loss != None:
                    loss.backward()
                    # alpha = self.helper.config['trigger_lr']
                    # sign() 的作用是丢弃梯度的具体数值大小, 只保留梯度下降的方向, 这就有点像FGSM
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m

    def search_trigger_k_combined(self, model, dl, epoch=0):
        """
        联立 LOSS 优化
        """
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config['trigger_lr']
        K = self.helper.config['trigger_outter_epochs']
        
        t = self.trigger.clone()
        m = self.mask.clone()
        
        noise_lambda = self.helper.config.get('noise_loss_lambda')
        adv_models = []
        adv_ws = []

        logging.info(f"Epoch:{epoch} - Combined Optimization Start")

        for iter in range(K):
            # 获取 re 所需的良性预测模型
            ben_model = self.get_ben_model(model, dl)
            
            # 获取函数 A3FL 所需的对抗模型
            if iter % self.helper.config.get('dm_adv_K', 10) == 0:
                # 清理旧模型释放内存
                for am in adv_models: del am
                adv_models, adv_ws = [], []
                for _ in range(self.helper.config['dm_adv_model_count']):
                    adv_m, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_m)
                    adv_ws.append(adv_w)

            for inputs, labels in dl:
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # 生成投毒样本
                poison_inputs = t * m + (1 - m) * inputs
                target_labels = labels.clone().fill_(self.helper.config['target_class'])
                
                # 计算 Loss 1: 基础后门损失 (L_backdoor)
                outputs = model(poison_inputs)
                l_backdoor = ce_loss(outputs, target_labels)
                
                # 计算 Loss 2: 触发器差异损失 (L_trigger)
                l_diff = self.trigger_loss_diff(model, ben_model, poison_inputs)
                
                # 初始加权 (re)
                loss = self.rate1 * l_diff + (1 - self.rate1) * l_backdoor
                
                # 计算 Loss 3: 对抗模型损失 (L_adv)
                if len(adv_models) > 0:
                    adv_loss_sum = 0
                    for am_idx, adv_m in enumerate(adv_models):
                        adv_outputs = adv_m(poison_inputs)
                        adv_loss_sum += adv_ws[am_idx] * ce_loss(adv_outputs, target_labels)
                    
                    # 累加到总损失 (函数 A3FL 的逻辑)
                    loss += (noise_lambda * adv_loss_sum / len(adv_models))

                # 反向传播与更新 
                if loss is not None:
                    loss.backward()
                    
                    # 优化
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()

        self.trigger = t.detach()
        self.mask = m.detach()
        logging.info("Combined trigger search completed.")    




    def search_trigger_re_a3fl(self, model, dl, epoch=0):
        """
        通过对抗训练生成满足论文中公式的简单后门触发器, 损失函数满足: Loss = a * L_backdoor + (1 - a) * L_trigger\n
        即需要保证后门在未被攻击的未来模型上的有效性, 并需要满足当前全局模型与模拟出的未来未中毒模型之间的差异较小
        """
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config['trigger_lr']
        K = self.helper.config['trigger_outter_epochs']
        # K = 10
        t = self.trigger.clone()
        m = self.mask.clone()
        count = 0
        # trigger_rate:{self.rate1} == trigger_loss_rate: 0.8
        logging.info(f"epoch:{epoch} trigger_rate:{self.rate1}")

        for iter in range(K):
            ben_model = self.get_ben_model(model, dl)
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, be_inputs, labels, be_labels = inputs.cuda(), inputs.cuda(),labels.cuda(), labels.cuda()
                # 
                inputs = t * m + (1 - m) * inputs
                labels[:] = self.helper.config['target_class']
                # model = self.helper.local_model
                outputs = model(inputs)
                # 这里的 loss 就是 中 L_backdoor = L_ce(x*, y*;Gt)表示触发器在当前全局模型 Gt (下发到客户端的) 的攻击任务损失
                loss = ce_loss(outputs, labels)
                # 这里的 diff_trigger_loss 就是 L_trigger, 即 model 指 Gt, ben_model 值 G_stop_t
                diff_trigger_loss = self.trigger_loss_diff(model, ben_model, inputs)
                # Loss = a * L_backdoor + (1 - a) * L_trigger, 见论文 P25 (3.1)
                loss = self.rate1 * diff_trigger_loss + (1 - self.rate1) * loss
                if loss != None:
                    loss.backward()
                    # alpha = self.helper.config['trigger_lr']
                    # sign() 的作用是丢弃梯度的具体数值大小, 只保留梯度下降的方向, 这就有点像FGSM
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m

        """
        model: 下发到本地的模型\n
        确保触发器不仅要在当前模型上攻击成功, 还要保证其在经过后门输入训练后的对抗模型上依旧有效\n
        """
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        # 存储生成的对抗模型列表
        adv_models = []
        # 存储每个对抗模型的相似度权重(sim_sum/sim_count: 所有卷积层梯度的平均余弦相似度(标量 float))
        adv_ws = []

        def val_asr(model, dl, t, m):
            """
            测试毒化数据(触发器)在模型上的攻击效果\n
            应该是用于在优化过程中实时监控触发器的效果\n
            Return:\n
                asr, total_loss
            """
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    labels[:] = self.helper.config['target_class']
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config['trigger_lr']
        # trigger_outter_epochs: 20 #50 200 20
        K = self.helper.config['trigger_outter_epochs']
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr=alpha*10, weight_decay=0)
        for iter in range(K):
            # 
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            """
            dm_adv_K: 1
            dm_adv_epochs: 5
            dm_adv_model_count: 1
            TOANSWER 这里为什么设置得那么小
            """
            # 生成对抗模型
            if iter % self.helper.config['dm_adv_K'] == 0 and iter != 0:
                if len(adv_models) > 0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.helper.config['dm_adv_model_count']):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
                    
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.helper.config['target_class']
                outputs = model(inputs)
                # 后门输入在local_model上的损失
                loss = ce_loss(outputs, labels)
                # TOANSWER 式子的原理? 为了迫使触发器在所有对抗模型上都能达到目标?
                # 即: 最终损失 Loss = L_(local_model) + noise_loss_lambda * ∑ (adv_w * L_(adv_model)) / dm_adv_model_count
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        # 后门输入在生成的对抗模型上的损失
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.helper.config['noise_loss_lambda']*adv_w*nm_loss/self.helper.config['dm_adv_model_count']
                        else:
                            # yaml : noise_loss_lambda: 0.01
                            loss += self.helper.config['noise_loss_lambda']*adv_w*nm_loss/self.helper.config['dm_adv_model_count']
                if loss != None:
                    loss.backward()
                    # TOANSWER 这一行把 t.grad 张量中的所有数值求和, 意义是什么?
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()


    def search_trigger_combined(self, model, dl, epoch=0):
        """
        联立 LOSS 优化
        """
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config['trigger_lr']
        K = self.helper.config['trigger_outter_epochs']
        
        t = self.trigger.clone()
        m = self.mask.clone()
        
        noise_lambda = self.helper.config.get('noise_loss_lambda', 0.01)
        adv_models = []
        adv_ws = []

        logging.info(f"Epoch:{epoch} - Combined Optimization Start")

        for iter in range(K):
            # 获取 re 所需的良性预测模型
            ben_model = self.get_ben_model(model, dl)
            
            # 获取函数 A3FL 所需的对抗模型
            if iter % self.helper.config.get('dm_adv_K', 10) == 0:
                # 清理旧模型释放内存
                for am in adv_models: del am
                adv_models, adv_ws = [], []
                for _ in range(self.helper.config.get('dm_adv_model_count', 1)):
                    adv_m, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_m)
                    adv_ws.append(adv_w)

            for inputs, labels in dl:
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # 生成投毒样本
                poison_inputs = t * m + (1 - m) * inputs
                target_labels = labels.clone().fill_(self.helper.config['target_class'])
                
                # 计算 Loss 1: 基础后门损失 (L_backdoor)
                outputs = model(poison_inputs)
                l_backdoor = ce_loss(outputs, target_labels)
                
                # 计算 Loss 2: 触发器差异损失 (L_trigger)
                l_diff = self.trigger_loss_diff(model, ben_model, poison_inputs)
                
                # 初始加权 (re)
                loss = self.rate1 * l_diff + (1 - self.rate1) * l_backdoor
                
                # 计算 Loss 3: 对抗模型损失 (L_adv)
                if len(adv_models) > 0:
                    adv_loss_sum = 0
                    for am_idx, adv_m in enumerate(adv_models):
                        adv_outputs = adv_m(poison_inputs)
                        adv_loss_sum += adv_ws[am_idx] * ce_loss(adv_outputs, target_labels)
                    
                    # 累加到总损失 (函数 A3FL 的逻辑)
                    loss += (noise_lambda * adv_loss_sum / len(adv_models))

                # 反向传播与更新 
                if loss is not None:
                    loss.backward()
                    
                    # 优化
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()

        self.trigger = t.detach()
        self.mask = m.detach()
        logging.info("Combined trigger search completed.")    


    def trigger_loss(self, model, backdoor_inputs, clean_inputs):
        model.train()
        """
        对比后门输入和干净输入在同一模型内部产生的特征激活(第一个激活层, 见论文 P25 式(3.2) )差异(差的平方和)\n
        在各种Net的定义中, 该函数大概长这个样子\n
        def first_activations(self, x):\n
            # x = F.relu(self.conv1(x)) / out = self.conv1(x)\n
            x = F.relu(self.bn1(self.conv1(x)))
        """
        backdoor_activations = model.first_activations(backdoor_inputs).mean([0, 1])
        clean_activations = model.first_activations(clean_inputs).mean([0, 1])
        difference = backdoor_activations - clean_activations
        loss = torch.sum(difference * difference)
        return loss

    def trigger_loss_diff(self, model, sim_model, backdoor_inputs):
        """
        对比后门输入在不同模型内部产生的特征激活(第一个激活层, 见论文 P25 式(3.2) )差异(差的平方和)\n
        即论文中的L_trigger\n
        在各种Net的定义中, 该函数大概长这个样子\n
        def first_activations(self, x):\n
            # x = F.relu(self.conv1(x)) / out = self.conv1(x)\n
            x = F.relu(self.bn1(self.conv1(x)))
        """        
        model.train()
        sim_model.train()
        before_activations = model.first_activations(backdoor_inputs).mean([0, 1])
        after_activations = sim_model.first_activations(backdoor_inputs).mean([0, 1])
        difference = before_activations - after_activations
        loss = torch.sum(difference * difference)
        return loss

    # 原始测量方法 (太好了又是不知所谓的注释, 我们有救了)
    def poison_input(self, inputs, labels, eval=False):
        """
        实施数据毒化, 根据预设的比例, 将后门触发器植入到输入的图像批次中, 并篡改相应的标签\n
        如果 eval=True(测试阶段), 则将当前批次中的所有图片都进行毒化\n
        如果 eval=False(训练阶段), 根据配置比例 bkd_ratio来计算毒化样本数
        """
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config['bkd_ratio'] * inputs.shape[0])
        # x^* = x * (1 - m) + ξ ⊙ m
        #  ⊙ 表示矩阵的按元素乘,m 表示一个尺寸与输入图像 x 相同的图像掩码 (详见论文 P15 式(2.3))
        # inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        # 1. 获取当前这批输入图片的实际长宽（可能是 28，也可能是 32）
        img_h, img_w = inputs.shape[2], inputs.shape[3]

        # 2. 检查当前的 trigger 和 mask 尺寸是否与 inputs 一致
        if self.trigger.shape[2] != img_h or self.trigger.shape[3] != img_w:
            # 如果不一致，说明运行中发生了 28 和 32 的混淆，动态对其进行空间切片或插值
            # 鉴于只相差 28 和 32，最安全的做法是直接切片对齐，或者用 F.interpolate 缩放
            import torch.nn.functional as F
            current_trigger = F.interpolate(self.trigger, size=(img_h, img_w), mode='bilinear', align_corners=False)
            current_mask = F.interpolate(self.mask, size=(img_h, img_w), mode='nearest')
        else:
            current_trigger = self.trigger
            current_mask = self.mask

        # 3. 使用对齐后的触发器和掩码进行投毒
        inputs[:bkd_num] = current_trigger * current_mask + inputs[:bkd_num] * (1 - current_mask)        
        # 见 yaml target_class: 2
        labels[:bkd_num] = self.helper.config['target_class']
        return inputs, labels

    def poison_input_test(self, inputs, labels):
        bkd_num = inputs.shape[0]
        new_inputs = inputs
        new_targets = labels
        adversarial_index = -1
        if self.helper.config["attacker_method"] == 'fcba':
            for index in range(0, len(new_inputs)):
                new_inputs[index] = self.add_pixel_pattern(inputs[index], adversarial_index)
                new_targets[index] = self.helper.config['target_class']
        else:
            new_inputs[:bkd_num] = self.trigger * self.mask + inputs[:bkd_num] * (1 - self.mask)
            new_targets[:bkd_num] = self.helper.config['target_class']
        new_inputs = new_inputs.to(self.helper.device)
        new_targets = new_targets.to(self.helper.device).long()
        new_inputs.requires_grad_(False)
        new_targets.requires_grad_(False)
        return new_inputs, new_targets


    def add_triggers(self, data):
        new_data = data
        new_data = self.trigger * self.mask + new_data * (1 - self.mask)
        new_data = new_data.to(self.helper.device)
        return new_data

    def poison_input_train(self, inputs, labels, adversarial_index):
        """
        将一个 batch 的原始数据转化为包含触发器的中毒数据, 并返回毒化数据与标签
        """
        # print(f'data_size: {len(inputs)}')
        # 确定当前 Batch 中有多少张图片需要被下毒
        if self.helper.config["sample_method"] == 'fcba':
            # TOANSWER 为什么是14张?
            bkd_num = 14
            # bkd_num = int(self.helper.config['bkd_ratio'] * inputs.shape[0])
        else:
            # inputs.shape[0] 应该是 BatchSize
            bkd_num = int(self.helper.config['bkd_ratio'] * inputs.shape[0])
            # FOCUS 直接引用
        new_inputs = inputs
        new_targets = labels

        if self.helper.config["attacker_method"] == 'fcba':
            for index in range(0, len(new_inputs)):
                if index < bkd_num:
                    new_inputs[index] = self.add_pixel_pattern(inputs[index], adversarial_index)
                    new_targets[index] = self.helper.config['target_class']
                else:
                    new_inputs[index] = inputs[index]
                    new_targets[index] = labels[index]
        else:
            # 除FCBA, 其他方法植入定义并在先前代码中完成的触发器
            new_inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
            new_targets[:bkd_num] = self.helper.config['target_class']

        new_inputs = new_inputs.to(self.helper.device)
        new_targets = new_targets.to(self.helper.device).long()

        return new_inputs, new_targets


    def get_fisher(self, model, dl, poison=1):
        copy_model = copy.deepcopy(model)
        ce_loss = torch.nn.CrossEntropyLoss()
        copy_model.train()
        copy_opt = torch.optim.SGD(copy_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        fisher = []
        if poison == 1:
            for i in range(self.helper.config["retrain_times"]):
                for inputs, labels in dl:
                    copy_opt.zero_grad()
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = self.poison_input(inputs, labels, False)
                    output = copy_model(inputs)
                    loss = ce_loss(output, labels)
                    loss.backward()
                    copy_opt.step()
        copy_model.eval()
        for inputs, labels in dl:
            inputs, labels = inputs.cuda(), labels.cuda()
            if poison == 1:
                inputs, labels = self.poison_input(inputs, labels,False)
            output = copy_model(inputs)
            loss = ce_loss(output, labels)
            loss.backward()
        for _, params in copy_model.named_parameters():
            if params.requires_grad and len(params.shape) != 1:
                fisher.append((params.grad.data ** 2).view(-1))
        fisher_list = torch.cat(fisher).cuda()
        return fisher_list
    
    
    def reshape_DI(self, model, Durable_Importance):
        copy_model = copy.deepcopy(model)
        DI_list = Durable_Importance.detach().cpu().numpy()
        mask_DI_list = []
        count = 0
        for _, params in copy_model.named_parameters():
            if params.requires_grad and len(params.shape) != 1:
                DI = DI_list[count:count + len(params.data.view(-1))]
                mask = list(DI.reshape(params.data.size()))
                mask = torch.from_numpy(np.array(mask, dtype='float32')).cuda()
                mask_DI_list.append(mask)
                count += len(params.data.view(-1))
        return mask_DI_list

    def opt_ReBA(self, model, dl):
        print("[*] #######  klog Into opt_ReBA() #########")
        ce_loss = torch.nn.CrossEntropyLoss()
        model.eval()
        alpha = self.helper.config["trigger_lr"]
        K = self.helper.config["trigger_outter_epochs"]
        t = self.trigger.clone()
        m = self.mask.clone()
        normal_grad = 0.
        count = 0
        for iter in range(K):
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t * m + (1 - m) * inputs
                labels[:] = self.helper.config["target_class"]
                outputs = model(inputs)
                loss_adv = ce_loss(outputs, labels)
                loss=loss_adv
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min =self.helper.config["trigger_min"], max = self.helper.config["trigger_max"])
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m

    def reba_poison_input_train(self, inputs, labels):
        """
        将一个 batch 的原始数据转化为包含左上角 5*5 触发器的中毒数据。
        """
        bkd_num = int(self.helper.config['bkd_ratio'] * inputs.shape[0])
        new_inputs = inputs.clone().to(self.helper.device)
        new_targets = labels.clone().to(self.helper.device)

        if bkd_num > 0:
            # self.reba_trigger 是经过 ATP 优化后的 5x5 patch [3, 5, 5]
            t_patch = torch.tanh(self.reba_trigger) 
            new_inputs[:bkd_num, :, 0:5, 0:5] = t_patch
            new_targets[:bkd_num] = self.helper.config['target_class']

        return new_inputs, new_targets

    # def poison_input(self, inputs, labels, eval=False):
    #     # print("into poison")
    #     if eval:
    #         bkd_num = inputs.shape[0]
    #     else:
    #         bkd_num = int(self.helper.config['bkd_ratio'] * inputs.shape[0])
    #         print(f"ratio bkd_num: {bkd_num}")
    #
    #     inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
    #     labels[:bkd_num] = self.helper.config['target_class']
    #     return inputs, labels

    def modelreplace(self):
        folder_name = f'{self.helper.config["folder_path"]}/saved_updates'
        user = self.helper.config["num_sampled_participants"]
        id = user[0]
        file_name = f'{folder_name}/update_{id}.pth'
        loaded_params = torch.load(file_name)
        for name, value in loaded_params.items():
            value.mul_(self.helper.config["fl_weight_scale"])
        for i in range(self.helper.config["num_sampled_participants"]):
            file_name = f'{folder_name}/update_{i}.pth'
            torch.save(loaded_params, file_name)

    def contain_adversary(self, epoch, sampled_participants):
        if self.helper.config["is_poison"] and \
            epoch < self.helper.config["poison_epochs"] and epoch >= 0:
            if self.helper.config["sample_method"] == 'random':
                for p in sampled_participants:
                    if p < self.helper.config["num_adversaries"]:
                        return p
        return -1

    # fcba 分布式后门
    def add_pixel_pattern(self, ori_image, adversarial_index):
        """
        基于cifar10_fcba.yaml定义好的模式, 在图片不同位置植入触发器
        """
        image = copy.deepcopy(ori_image)
        poison_patterns = self.get_pattern(adversarial_index)
        # if adversarial_index != -1:
        #     print(f"adversarial_index: {adversarial_index}  patterns:{poison_patterns}")
        # image[c][r][c] rgb行列
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1

        return image

    def get_pattern(self, adversarial_index):
        """
        基于cifar10_fcba.yaml中定义的参数获取后门触发器模式列表
        """
        poison_patterns = []
        if adversarial_index == -1:
            # trigger_num 只在 cifar10_fcba.yaml 中定义, 同理, 下面的i_poison_pattern也仅在cifar10_fcba.yaml中硬编码
            for i in range(0, self.helper.config['trigger_num']):
                poison_patterns = poison_patterns + self.helper.config[str(i) + '_poison_pattern']
        elif self.helper.config["trigger_num"] == 5:
            if adversarial_index < 5:
                poison_patterns = self.helper.config[str(adversarial_index) + '_poison_pattern']
            elif adversarial_index < 10:
                poison_patterns = poison_patterns + self.helper.config[str(adversarial_index % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 1) % 5) + '_poison_pattern']
            elif adversarial_index < 14:
                poison_patterns = poison_patterns + self.helper.config[str(adversarial_index % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 2) % 5) + '_poison_pattern']
            elif adversarial_index < 20:
                poison_patterns = poison_patterns + self.helper.config[str((adversarial_index) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 1) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 2) % 5) + '_poison_pattern']
            elif adversarial_index < 25:
                poison_patterns = poison_patterns + self.helper.config[str((adversarial_index) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 1) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 3) % 5) + '_poison_pattern']
            else:
                poison_patterns = poison_patterns + self.helper.config[str((adversarial_index) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 1) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 2) % 5) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 3) % 5) + '_poison_pattern']
        elif self.helper.config["trigger_num"] == 4:
            if adversarial_index < 4:
                poison_patterns = self.helper.config[str(adversarial_index) + '_poison_pattern']
            elif adversarial_index < 8:
                poison_patterns = poison_patterns + self.helper.config[str(adversarial_index % 4) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 1) % 4) + '_poison_pattern']
            elif adversarial_index < 10:
                poison_patterns = poison_patterns + self.helper.config[str(adversarial_index % 4) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 2) % 4) + '_poison_pattern']
            else:
                poison_patterns = poison_patterns + self.helper.config[str((adversarial_index) % 4) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 1) % 4) + '_poison_pattern']
                poison_patterns = poison_patterns + self.helper.config[
                    str((adversarial_index + 2) % 4) + '_poison_pattern']

        return poison_patterns

    def add_cen_pattern(self, ori_image):
        image = copy.deepcopy(ori_image)
        poison_patterns = self.pos
        # image[c][r][c] rgb行列
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
        return image
    







