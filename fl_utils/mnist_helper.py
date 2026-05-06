import sys

sys.path.append("../")
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
from model.simple import SimpleMnist, SupSimpleMnist
import logging
import pandas as pd
from fl_utils.utils import create_logger
import os
import yaml
import math
logger = logging.getLogger('logger')


class Mnist_Helper:
    def __init__(self, config):
        self.config = config
        self.config["folder_path"] = f'../main/re_result/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{self.config["attacker_method"]}_DOBA'
        if self.config["attacker_method"] != "sin-adv":
            self.config["folder_path"] = f'../main/re_result/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{self.config["attacker_method"]}'
        self.config["data_folder"] = '../data/MNIST'
        self.make_folders()
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
        self.local_model = SimpleMnist()
        self.local_model.cuda()
        self.global_model = SimpleMnist()
        self.global_model.cuda()
        self.contrastive_model = SupSimpleMnist()
        self.contrastive_model.cuda()
        for i in range(self.config["num_total_participants"]):
            t_model = SimpleMnist()
            t_model.cuda()
            self.client_models.append(t_model)

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
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
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config["num_worker"])
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["test_batch_size"],
            shuffle=False,
            num_workers=self.config["num_worker"])
        return test_loader

    def load_data(self):
        print('into load_data')
        print(f'{self.config["data_folder"]}')
        self.num_classes = 10
        normalize = transforms.Normalize((0.1307,), (0.3081,))

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.train_dataset = datasets.MNIST(
            self.config["data_folder"],
            train=True,
            download=True,
            transform=transform_train)

        self.test_dataset = datasets.MNIST(
            self.config["data_folder"],
            train=False,
            transform=transform_test)

        indices_per_participant = self.sample_dirichlet_train_data(
            self.config["num_total_participants"],
            alpha=self.config["dirichlet_alpha"])

        train_loaders = [self.get_train(indices)
                         for pos, indices in indices_per_participant.items()]
        self.train_neur = [(pos, self.get_train(indices)) for pos, indices in
                           indices_per_participant.items()]

        self.train_data = train_loaders
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
        filepath = f"{self.config['folder_path']}/accuracy.csv"
        filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{self.config["attacker_method"]}_DOBA_accuracy.csv'
        if self.config["attacker_method"] != "sin-adv":
            filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{self.config["attacker_method"]}_accuracy.csv'
        acc_frame.to_csv(filepath)                              
        print(f"Saving accuracy record to {filepath}")

    def save_update(self, model=None, userID=0):
        folderpath = '{0}/saved_updates'.format(self.config["folder_path"])
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)

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

    def record_trigger(self, epoch, trigger):
        trigger_flattened = trigger.view(-1).cpu().numpy()
        data = [[epoch] + list(trigger_flattened)]
        filepath = f"{self.config['folder_path']}/trigger.csv"
        if epoch == 1:
            num_trigger_elements = trigger.numel()
            columns = ['epoch'] + [f'{i}' for i in range(num_trigger_elements)]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False)
            self.header_written = True
        else:
            df = pd.DataFrame(data)
            df.to_csv(filepath, mode='a', header=False, index=False)

    def model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            # print(name)
            if name in target_params:
                # 计算平方差并累加
                squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))

        return math.sqrt(squared_sum)

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