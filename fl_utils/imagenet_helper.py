import datetime
import json
import math
import pickle
import sys
sys.path.append("../")
sys.path.append("./")
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
from model.pytorch_resnet import pt_resnet18, SupConResNet18
import logging
import pandas as pd
from fl_utils.utils import create_logger
import os
import yaml
logger = logging.getLogger('logger')


# random.seed(42)

# CIFAR10
class ImageNet_Helper:
    def __init__(self, config):
        self.config = config
        mia = "mia" if self.config["mia"] else "no-mia"
        noise = "noise" if self.config["noise"] else "no-noise"
        self.config["folder_path"] = f'../main/re_result/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}_DOBA'
        if self.config["attacker_method"] != "sin-adv":
            self.config["folder_path"] = f'../main/re_result/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}'
        self.config["data_folder"] = '../data/'
        self.make_folders()
        self.num_classes = 200
        self.local_model = None
        self.global_model = None
        self.client_models = []
        self.setup_all()
        self.accuracy = [[], [], [], [], []]
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
            os.mkdir(self.config["folder_path"])
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
        self.local_model = pt_resnet18(num_classes=self.num_classes)
        # self.local_model = resnet18()
        self.local_model.cuda()
        self.global_model = pt_resnet18(num_classes=self.num_classes)
        # self.global_model = resnet18()
        self.global_model.cuda()

        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        self.contrastive_model = SupConResNet18(name='Contrastive',
                                           created_time=current_time)
        # self.contrastive_model = SupConVGG16()
        self.contrastive_model.cuda()
        for i in range(self.config["num_total_participants"]):
            t_model = pt_resnet18(num_classes=self.num_classes)
            # t_model = resnet18()
            t_model.cuda()
            self.client_models.append(t_model)
        
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        print("into sample")
        # cifar_classes = {}
        # print(len(self.train_dataset))
        # for ind, x in enumerate(self.train_dataset):
        #     _, label = x
        #     # print(label)
        #     if label in cifar_classes:
        #         cifar_classes[label].append(ind)
        #     else:
        #         cifar_classes[label] = [ind]
        # print("f1 sample")
        # class_size = len(cifar_classes[0])
        # per_participant_list = defaultdict(list)
        # no_classes = len(cifar_classes.keys())
        # print("begin sample")
        # for n in range(no_classes):
        #     random.shuffle(cifar_classes[n])
        #     sampled_probabilities = class_size * np.random.dirichlet(
        #         np.array(no_participants * [alpha]))
        #     for user in range(no_participants):
        #         no_imgs = int(round(sampled_probabilities[user]))
        #         sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
        #         per_participant_list[user].extend(sampled_list)
        #         cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        # dict_to_save = dict(per_participant_list)
        # with open("per_participant_list.json", 'w') as file:
        #     json.dump(dict_to_save, file)
        with open("per_participant_list.json", 'r') as file:
            loaded_dict = json.load(file)
        # 将读取回来的普通字典转换为defaultdict
        per_participant_list = defaultdict(list, loaded_dict)
        print("end sample and save")
        return per_participant_list
    
    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config["num_worker"])
        return train_loader

    def load_data(self):
        print('into load_data')
        print(f'{self.config["data_folder"]}')
        self.num_classes = 200
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_dataset = datasets.ImageFolder(
            os.path.join(self.config["data_folder"], "tiny-imagenet-200", 'train'),
            transform_train)
        print("train data ok")
        self.test_dataset = datasets.ImageFolder(
            os.path.join(self.config["data_folder"], 'tiny-imagenet-200', 'val'),
            transform_test)
        print("test data ok")
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config["num_total_participants"],
            alpha=self.config["dirichlet_alpha"])
        # print(indices_per_participant)
        print("participant ok")

        self.train_neur = [(pos, self.get_train(indices)) for pos, indices in
                           indices_per_participant.items()]
        print("load data ok")
        self.train_data = [self.get_train(indices)
            for pos, indices in indices_per_participant.items()]

        self.test_data = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["test_batch_size"],
            shuffle=False,
            num_workers=self.config["num_worker"], pin_memory=True)
        print("get data ok")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_worker"], pin_memory=True)
        print("data ok")
    
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
        filepath = f"""{self.config['folder_path']}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}_DOBA_accuracy.csv"""
        if self.config["attacker_method"] != "sin-adv":
            filepath = f"""{self.config['folder_path']}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}_accuracy.csv"""
        acc_frame.to_csv(filepath)
        print(f"Saving accuracy record to {filepath}")

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

    def record_trigger(self, epoch, trigger, mask):
        trigger_flattened = trigger.view(-1).cpu().numpy()
        mask_flattened = mask.view(-1).cpu().numpy()
        data = [[epoch] + list(trigger_flattened) + list(mask_flattened)]
        filepath = f"{self.config['folder_path']}/trigger.csv"
        if not self.header_written:
            num_trigger_elements = trigger.numel()
            num_mask_elements = mask.numel()
            columns = ['epoch'] + [f'trigger_{i}' for i in range(num_trigger_elements)] + [f'mask_{i}' for i in
                                                                                           range(num_mask_elements)]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False)
            self.header_written = True
        else:
            df = pd.DataFrame(data)
            df.to_csv(filepath, mode='a', header=False, index=False)

    def save_update(self, model=None, userID = 0):
        folderpath = '{0}/saved_updates'.format(self.config["folder_path"])
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)

    def model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

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
