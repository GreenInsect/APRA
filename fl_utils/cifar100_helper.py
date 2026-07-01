import math
import sys
import datetime

from model.alxnet import Alxnet
from model.mobilenet import Mobilenet
from model.vgg16 import VGG16

sys.path.append("../")
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
from model.resnet import ResNet18, SupConResNet18, ResNet34, SupConResNet34
import logging
import pandas as pd
from utils.utils import create_logger
import os
import yaml

logger = logging.getLogger('logger')


# random.seed(42)

class Cifar100_Helper:
    def __init__(self, config):
        self.config = config
        self.num_classes = 100
        
        label = None
        if self.config["mia_class_method"] == "nobackdoor_random":
            # 生成一个非目标的随机标签
            label = self.generate_random_label()
        elif self.config["mia_class_method"] == "backdoor":
            label = self.config["target_class"]
        self.config["mia_class"] = label
        if self.config["mia_class_method"] == "random":
            print("生成的随机图片标签模式为: per-sample random")
        else:
            print(f"生成的随机图片的标签为:{label}")

        mia = "mia" if self.config["mia"] else "no-mia"
        noise = "noise" if self.config["noise"] else "no-noise"
        mia_class = "???" if self.config["mia"] == True else "no-mia-class"
        if self.config["mia_class_method"] == "random":
            mia_class = "random"
        else:
            mia_class = str(self.config["mia_class"])
        if not self.config["mia"]:
            mia_class = "no-mia-class"            
        self.config["folder_path"] = f'../main/re_result_{self.config["comment"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["sample_method"]}_{self.config["is_poison"]}_{mia}_{mia_class}_{noise}_{self.config["attacker_method"]}_DOBA'
        if self.config["attacker_method"] != "sin-adv":
            self.config["folder_path"] = f'../main/re_result_{self.config["comment"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["sample_method"]}_{self.config["is_poison"]}_{mia}_{mia_class}_{noise}_{self.config["attacker_method"]}'
        if self.config["attacker_method"] == "modelreplace":
            self.config["folder_path"] = f'../main/re_result_{self.config["comment"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["sample_method"]}_{self.config["is_poison"]}_{mia}_{mia_class}_{noise}_{self.config["attacker_method"]}_{self.config["fl_weight_scale"]}'          
        self.config["data_folder"] = '../data/'
        self.make_folders()
        self.local_model = None
        self.global_model = None
        self.contrastive_model = None
        self.num_classes = 100
        self.client_models = []
        self.setup_all()
        self.accuracy = [[], [], [], [], []]
        self.train_log = [[], [], [], [], [], [], []]
        self.distance = [[],[],[]]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the logger
        fh = logging.FileHandler(
            filename=f'{self.config["folder_path"]}/log.txt')
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def make_folders(self):
        log = create_logger()
        try:
            os.makedirs(self.config["folder_path"], exist_ok=True)
        except FileExistsError:
            log.info('Folder already exists')

        fh = logging.FileHandler(
            filename=f'{self.config["folder_path"]}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s '
                                      '- %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        log.addHandler(fh)

        with open(f'{self.config["folder_path"]}/params.yaml.txt', 'w') as f:
            yaml.dump(self.config, f)

    def setup_all(self):
        self.load_data()
        self.load_model()
        self.config_adversaries()

    def load_model(self):
        """
        模型初始化, 在内存(及显存)中构建起联邦学习所需的所有角色模型, 包括本地模型(self.local_model)、全局模型(self.global_model)、用于攻击的对比学习模型(self.contrastive_model), 以及所有参与者的本地副本(self.client_models)
        """
        self.local_model = ResNet18(num_classes=self.num_classes)
        # self.local_model = Mobilenet()
        # self.local_model = ResNet34(num_classes=self.num_classes)
        # self.local_model = VGG16()
        self.local_model.cuda()
        self.global_model = ResNet18(num_classes=self.num_classes)
        # self.global_model = ResNet34(num_classes=self.num_classes)
        # self.global_model = Mobilenet()
        # self.global_model = VGG16()
        self.global_model.cuda()
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        # 对比模型, 在 fler.py -> chameleon_train() 中使用
        self.contrastive_model = SupConResNet18(name='Contrastive',
                                                created_time=current_time)
        # self.contrastive_model = SupConResNet34(name='Contrastive',
        #                                         created_time=current_time)
        self.contrastive_model.cuda()
        for i in range(self.config["num_total_participants"]):
            t_model = ResNet18(num_classes=self.num_classes)
            # t_model = ResNet34(num_classes=self.num_classes)
            # t_model = Mobilenet()
            t_model.cuda()
            self.client_models.append(t_model)

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
        Params:\n
            no_participants: 参与者总数\n
        该函数最终返回一个字典 per_participant_list,结构如下:\n

        Key: 用户 ID (0, 1, 2...)

        Value: 该用户拥有的所有数据索引列表, 每个用户拥有的样本总数以及各类别比例是不均衡的       
        """
        cifar_classes = {}
        # 这里通过遍历, 将所有属于同一类别的图片索引收集到 cifar_classes 字典中(例如: 键 0 对应所有"飞机"的索引列表?)
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            # 狄利克雷分布采样. 它会返回一个长度等于参与者数量的概率向量, 向量各元素之和为 1, 乘以 class_size: 将概率转换为实际分配给每个用户的样本数量
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user])) # 计算该用户应分得的数量
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list) # 添加到用户的数据列表中
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        return per_participant_list

    def get_train(self, indices):
        """
        从原始数据集中根据给定的索引indices使用SubsetRandomSampler创建一个用于训练的子集数据加载器DataLoader(train_loader)
        """
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            # 随机子集采样
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config["num_worker"])
        return train_loader

    def get_test(self):
        """
        创建测试数据加载器Test DataLoader
        """
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["test_batch_size"],
            shuffle=False,
            num_workers=self.config["num_worker"])

        return test_loader

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def load_data(self):
        print('into load_data')
        print(f'{self.config["data_folder"]}')
        self.num_classes = 100
        self.channel=3
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = datasets.CIFAR100(
            self.config["data_folder"],
            train=True,
            download=True,
            transform=transform_train)

        self.test_dataset = datasets.CIFAR100(
            self.config["data_folder"],
            train=False,
            transform=transform_test)

        self.classes_dict = self.build_classes_dict()

        # 利用 狄利克雷分布（Dirichlet Distribution） 将全局训练集的索引（Indices）分配给每一个参与者
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config["num_total_participants"],
            alpha=self.config["dirichlet_alpha"])

        # train_loaders(== self.train_data) 比 self.train_neur 少了每个参与者的pos( pos == ID )
        train_loaders = [self.get_train(indices)
                         for pos, indices in indices_per_participant.items()]
        # (pos, pos对应的参与者的DataLoader) 的元组列表
        self.train_neur = [(pos, self.get_train(indices)) for pos, indices in
                           indices_per_participant.items()]

        # 针对每一个参与者(Participant)的索引数据, 批量生成对应的训练数据加载器(DataLoader)
        self.train_data = train_loaders
        # 数据集的dataloader
        self.test_data = self.get_test()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_worker"])

    def config_adversaries(self):
        if self.config["is_poison"]:
            self.adversary_list = list(range(self.config["num_adversaries"]))
        else:
            self.adversary_list = list()

    def record_acc(self, epoch, loss, acc, bkd_loss, bkd_ac, lr):
        print('into acc')

        self.accuracy[0].append(acc)
        self.accuracy[1].append(loss)
        self.accuracy[2].append(bkd_ac)
        self.accuracy[3].append(bkd_loss)
        self.accuracy[4].append(lr)
        name = ['main', 'main_loss', 'backdoor', 'backdoor_loss', 'lr']
        acc_frame = pd.DataFrame(columns=name, data=zip(*self.accuracy),
                                 index=range(self.config['start_epoch'], epoch + 1))
        mia = "mia" if self.config["mia"] else "no-mia"
        noise = "noise" if self.config["noise"] else "no-noise"
        mia_class = "???" if self.config["mia"] == True else "no-mia-class"
        if self.config["mia_class_method"] == "random":
            mia_class = "random"
        else:
            mia_class = str(self.config["mia_class"])       
        if not self.config["mia"]:
            mia_class = "no-mia-class"             
        filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["sample_method"]}_{self.config["is_poison"]}_{mia}_{mia_class}_{noise}_{self.config["attacker_method"]}_DOBA_accuracy.csv'
        if self.config["attacker_method"] != "sin-adv":
            filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["sample_method"]}_{self.config["is_poison"]}_{mia}_{mia_class}_{noise}_{self.config["attacker_method"]}_accuracy.csv'
        if self.config["attacker_method"] == "modelreplace":       
            filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["sample_method"]}_{self.config["is_poison"]}_{mia}_{mia_class}_{noise}_{self.config["attacker_method"]}_{self.config["fl_weight_scale"]}_accuracy.csv'                         
        acc_frame.to_csv(filepath)
        print(f"Saving accuracy record to {filepath}")

    def record_norm(self, epoch, particpant, distance):
        self.distance[0].append(epoch)
        self.distance[1].append(particpant)
        self.distance[2].append(distance)
        name = ['epoch','particpant', 'distance']
        acc_frame = pd.DataFrame(columns=name, data=zip(*self.distance))
        filepath = f"{self.config['folder_path']}/distance.csv"
        acc_frame.to_csv(filepath)

    def record_train_acc(self, epoch, participant, is_attack, loss, acc, bkd_loss, bkd_ac):
        self.train_log[0].append(epoch)
        self.train_log[1].append(participant)
        self.train_log[2].append(is_attack)
        self.train_log[3].append(acc)
        self.train_log[4].append(loss)
        self.train_log[5].append(bkd_ac)
        self.train_log[6].append(bkd_loss)

        name = ['epoch','client_id','is_attack','main', 'main_loss', 'backdoor', 'backdoor_loss']
        acc_frame = pd.DataFrame(columns=name, data=zip(*self.train_log))
        filepath = f"{self.config['folder_path']}/training_accuracy.csv"
        acc_frame.to_csv(filepath)
        # print(f"Saving accuracy record to {filepath}")

    def record_mia(self, epoch, acc):
        filepath = f"{self.config['folder_path']}/mia_acc.csv"
        data = [[epoch] + [acc]]
        if epoch == 1:
            columns = ['epoch'] + ['mia_acc']
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False)
            self.header_written = True
        else:
            df = pd.DataFrame(data)
            df.to_csv(filepath, mode='a', header=False, index=False)

    def record_trigger(self, epoch, trigger):
        """
        持久化记录后门攻击中的"触发器"在每一轮训练中的状态, 但是只保存到一个文件中, 即只保存最后一次调用
        """
        trigger_flattened = trigger.view(-1).cpu().numpy()
        # 将当前轮次 epoch 和展平后的触发器数值拼接成一行数据
        data = [[epoch] + list(trigger_flattened)]
        filepath = f"{self.config['folder_path']}/trigger.csv"
        # 如果是第一轮训练, 需要创建文件并写入表头
        if epoch == 1:
            # 触发器的元素个数
            num_trigger_elements = trigger.numel()
            # 生成 CSV 的列名. 第一列叫 epoch, 后面的列用数字索引 0, 1, 2... 命名
            columns = ['epoch'] + [f'{i}' for i in range(num_trigger_elements)]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False)
            self.header_written = True
        # 如果轮次大于 1, 则以追加模式写入
        else:
            df = pd.DataFrame(data)
            df.to_csv(filepath, mode='a', header=False, index=False)

    def read_trigger(self, filepath, epoch):
        df = pd.read_csv(filepath)
        if 'epoch' not in df.columns:
            raise ValueError(f"CSV file {filepath} does not have an 'epoch' column.")
        try:
            epoch_row = df[df['epoch'] == epoch]
            num_elements = len(epoch_row.iloc[0, 1:])
            trigger_data = epoch_row.iloc[0, 1:].values.flatten()
            return torch.tensor(trigger_data, dtype=torch.float32, device='cuda').reshape(1, 3, 32, 32)
        except IndexError:
            raise ValueError(f"Epoch {epoch} not found in the CSV file.")

    def save_update(self, model=None, userID=0):
        """
        保存模型至self.config["folder_path"]/saved_updates
        """
        folderpath = '{0}/saved_updates'.format(self.config["folder_path"])
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)

    def model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            # print(name)
            if name in target_params:
                # 计算平方差并累加
                squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))

        return math.sqrt(squared_sum)
        #     squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        # return math.sqrt(squared_sum)

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def model_dist_norm_var(self, model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)
    
    
    def generate_random_label(self):
        """
        攻击者用来生成一个非目标的随机标签
        """
        number = random.randint(1, self.num_classes)
        while number == self.config["target_class"]:
            number = random.randint(1, self.num_classes)
        return number        

    # def remove_update(self):
    #     for i in (self.config["num_sampled_participants"]):
    #         file_name = '{0}/saved_updates/update_{1}.pth'.format(self.config["folder_path"], i)
    #         if os.path.exists(file_name):
    #             os.remove(file_name)
    #     os.rmdir('{0}/saved_updates'.format(self.config["folder_path"]))
    #     if self.config["agg_method"] == 'Foolsgold':
    #         for i in (self.config["num_sampled_participants"]):
    #             file_name = '{0}/foolsgold/history_{1}.pth'.format(self.config["folder_path"], i)
    #             if os.path.exists(file_name):
    #                 os.remove(file_name)
    #         os.rmdir('{0}/foolsgold'.format(self.config["folder_path"]))

