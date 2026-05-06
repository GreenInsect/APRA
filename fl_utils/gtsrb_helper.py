import sys

sys.path.append("../")
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
from model.resnet import ResNet18
import logging
import pandas as pd
from utils.utils import create_logger
import os
import yaml

logger = logging.getLogger('logger')


class GTSRB_Helper:
    def __init__(self, config):
        self.config = config
        mia = "mia" if self.config["mia"] else "no-mia"
        noise = "noise" if self.config["noise"] else "no-noise"
        self.config["folder_path"] = f'../main/re_result/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}_DOBA'
        if self.config["attacker_method"] != "sin-adv":
            self.config["folder_path"] = f'../main/re_result/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}'
        self.config["data_folder"] = '../data/'
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
        self.local_model = ResNet18(num_classes=self.num_classes)
        self.local_model.cuda()
        self.global_model = ResNet18(num_classes=self.num_classes)
        self.global_model.cuda()
        for i in range(self.config["num_total_participants"]):
            t_model = ResNet18(num_classes=self.num_classes)
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
        print('into load_gtstrb_data')
        print(f'{self.config["data_folder"]}')
        self.num_classes = 43
        means = [0.3337, 0.3064, 0.3171]
        std = [0.2672, 0.2564, 0.2629]
        size = (32, 32, 3)

        transform_train = transforms.Compose([
            transforms.Resize(size[0]),
            transforms.CenterCrop(size[0]),
            transforms.ToTensor(),
            transforms.Normalize(means, std)
        ])

        transform_test = transform_train

        self.train_dataset = datasets.GTSRB(root=self.config["data_folder"], split='train',transform=transform_train, download=True)
        self.test_dataset = datasets.GTSRB(root=self.config["data_folder"], split='test',transform=transform_test, download=True)

        indices_per_participant = self.sample_dirichlet_train_data(
            self.config["num_total_participants"],
            alpha=self.config["dirichlet_alpha"])

        train_loaders = [self.get_train(indices)
                         for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_worker"])

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
        filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}_DOBA_accuracy.csv'
        if self.config["attacker_method"] != "sin-adv":
            filepath = f'{self.config["folder_path"]}/{self.config["dataset"]}_{self.config["epochs"]}_{self.config["current_time"]}_{self.config["agg_method"]}_{self.config["is_poison"]}_{mia}_{noise}_{self.config["attacker_method"]}_accuracy.csv'
        acc_frame.to_csv(filepath)
        print(f"Saving accuracy record to {filepath}")

    def config_adversaries(self):
        if self.config["is_poison"]:
            self.adversary_list = list(range(self.config["num_adversaries"]))
        else:
            self.adversary_list = list()

    def save_update(self, model=None, userID=0):
        folderpath = '{0}/saved_updates'.format(self.config["folder_path"])
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)


