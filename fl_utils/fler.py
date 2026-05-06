import sys

from fl_utils.losses import SupConLoss
from fl_utils.utils import get_ood_data, _get_ood_dataloader

sys.path.append("../")
import time
import torch

import random
import numpy as np
import copy
import os

from .attacker import Attacker
from .aggregator import Aggregator
from math import ceil
import pickle
import logging
import pandas as pd


logger = logging.getLogger('logger')


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_flattened_weights(model):
    """提取模型中所有参数并展平为一维向量"""
    return torch.cat([p.data.view(-1) for p in model.values()])

def record_trajectory_to_csv(round_idx, current_sd, base_weights, asr, main_acc, csv_path):
    """计算当前轮次指标并追加到CSV"""
    current_weights = get_flattened_weights(current_sd).detach()
    
    # 计算指标
    cos_sim = F.cosine_similarity(base_weights.unsqueeze(0), current_weights.unsqueeze(0)).item()
    euc_dist = torch.norm(base_weights - current_weights, p=2).item()
    
    # 构造单行数据
    data = {
        "epoch": round_idx,
        "cosine_similarity": cos_sim,
        "l2_distance": euc_dist,
        "backdoor": asr,
        "main": main_acc
    }
    
    # 转换为DataFrame并追加写入CSV
    df = pd.DataFrame([data])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)


class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.helper = helper
        self.SupConLoss = SupConLoss().cuda()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attack_sum = 0
        #  self.ood_train 是一个包含分布外数据(OOD)的数据加载器
        self.ood_train= _get_ood_dataloader(self.helper.config["dataset"])
        self.aggregator = Aggregator(self.helper, self.ood_train)
        self.start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        self.attacker = Attacker(self.helper)
        # self.kappa = torch.tensor(self.helper.config["kappa_0"], requires_grad=True, device=self.device)
        self.phi = None     
        if self.helper.config["sample_method"] == 'random_updates':
            self.init_advs()           
        
        # Load benign model ../saved/pretrain/cifar100_500_resnet18.pt,下面的代码:根据数据集的选用对应加载预训练模型
        print(self.helper.config['load_benign_model'])
        
        model_path = ''
        # TOANSWER self.helper.config.is_poison( 表示该联邦学习进程存在攻击?):
        if self.helper.config['load_benign_model']:  # and self.helper.config.is_poison( TOANSWER 表示该联邦学习进程存在攻击?):
            if self.helper.config["dataset"] == 'mnist':
                model_path = f'../saved/pretrain/{self.helper.config["dataset"]}_10_{self.helper.config["model"]}.pt'
            elif self.helper.config["dataset"] == 'cifar100':
                model_path = f'../saved/pretrain/{self.helper.config["dataset"]}_500_{self.helper.config["model"]}.pt'
                # model_path = f'../saved/benign_new/{self.helper.config["dataset"]}_100_sin-adv_avg.pt'
            elif self.helper.config["dataset"] == 'tiny-imagenet-200':
                model_path = f'../saved/pretrain/{self.helper.config["dataset"]}_900_{self.helper.config["model"]}.pt'
            elif self.helper.config["dataset"] == 'cifar10':
                model_path = f'../saved/pretrain/{self.helper.config["dataset"]}_900_{self.helper.config["model"]}.pt'
                # model_path = f'../saved/pretrain/cifar10_600_avg.pt'
                if self.helper.config["attacker_method"] == "reba":
                    model_path = f'../saved/pretrain/{self.helper.config["dataset"]}_1900_{self.helper.config["model"]}.pt'
            elif self.helper.config["dataset"] == 'gtsrb':
                model_path = f'../saved/pretrain/{self.helper.config["dataset"]}_500_{self.helper.config["model"]}.pt'            
            else:
                logger.error("no model")
            # FOCUS  FLer 和 Aggregator都要基于 helper 进行构造, 而global_model正位于self.helper.global_model
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location='cuda')['model'])
            print(f'Load benign model {model_path}')
            loss, acc = self.test_once()
            print(f'Load benign model {model_path}, acc {acc:.3f}')
        return
    
    def init_advs(self):
        num_updates = self.helper.config["num_sampled_participants"] * self.helper.config["poison_epochs"]
        num_poison_updates = ceil(self.helper.config["sample_poison_ratio"] * num_updates)
        updates = list(range(num_updates))
        advs = np.random.choice(updates, num_poison_updates, replace=False)
        print(f'Using random updates, sampled {",".join([str(x) for x in advs])}')
        adv_dict = {}
        for adv in advs:
            epoch = adv // self.helper.config["num_sampled_participants"]
            idx = adv % self.helper.config["num_sampled_participants"]
            if epoch in adv_dict:
                adv_dict[epoch].append(idx)
            else:
                adv_dict[epoch] = [idx]
        self.advs = adv_dict    

    def test_once(self, poison=False):
        """
        对 self.helper.global_model 进行一次测试\n
        根据poison选择是否毒化输入: data, targets = self.attacker.poison_input(data, targets, eval=True)
        """
        # print("begin_malicious testing")
        # global_model 整个代码中使用的 model,基于预训练模型
        model = self.helper.global_model
        model.eval()
        with torch.no_grad():
            # data_source 数据集的dataloader
            data_source = self.helper.test_data  # data_source 数据集的 test dataloader ,也有 train_data 的
            """data_source 数据集的dataloader"""
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0) 
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        print(f"[*] klog : acc : {acc}, loss : {loss}\n")
        # [*] klog : acc : 67.11, loss : 0.0012455023050308228
        return loss, acc
    
    def test_local_once(self, model, poison = False):
        """
        这个函数没用到? 用到的全注释了, 两个的区别只在于上面用的model = self.helper.global_model, 这里是传进来的model
        """
        model.eval()
        with torch.no_grad():
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        return loss, acc

    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc):
        print(f"{epoch} log:")
        log_dict = {
            'epoch': epoch, 
            'test_acc': acc,
            'test_loss': loss, 
            'bkd_acc': bkd_acc, # 后门触发成功率
            'bkd_loss': bkd_loss # 后门任务损失
            }
        print(f"epoch:{epoch}  main:{acc}  bkd:{bkd_acc}")
        print(f"epoch:{epoch}  main_acc :{acc}  bkd_acc:{bkd_acc}")
        if not self.helper.config["is_poison"]:
            self.save_benign_model(epoch, log_dict) # 保存了模型的权重, 还将 log_dict 一并存入文件

    def save_benign_model(self, epoch, log_dict):
        """
        保存全局模型
        """
        print(f"[**] klog enter save_benign_model ")
        # self.helper.config["save_on_epochs"] 代码这里在cifar100下应该为空啊(见 cifar100.yaml line: 55, 且搜索无相关改变代码), 那你这里执行不了, /data/datasets/home/jyf/tmp/pycharm_project_297/saved/benign_new亦为空
        # 确实, 能进入函数, 但没用, 下面不运行
        if epoch in self.helper.config["save_on_epochs"]:
            print(f"[**] klog epoch : {epoch} ")
            print(f'Saving model on epoch {epoch}')
            log_dict['model'] = self.helper.global_model.state_dict()
            save_path = f'../saved/benign_new/{self.helper.config["dataset"]}_{epoch}_{self.helper.config["attacker_method"]}_{self.helper.config["agg_method"]}.pt'
            torch.save(log_dict, save_path)
            print(f'Model saved at {save_path}')
    
    def save_local_model(self, participant, model, epoch, poison):
        """
        实际执行的保存函数\n
        这里的poison是first_adversary 代表当前轮次中第一个被抽样到的攻击者的ID  participant应该是模型拥有者ID, 即当前正在被处理的那个客户端的 ID
        """
        log_dict = {}
        log_dict['model'] = model.state_dict()
        save_path = f'../saved/local_model/{epoch}_{poison}_{participant}.pt'
        torch.save(log_dict, save_path)

    def normalize_vector(self, vector, base=0):
        min_vec = torch.min(vector)
        max_vec = torch.max(vector)
        normalized_vector = (vector - min_vec) / (max_vec - min_vec)
        return normalized_vector + base


    def train(self):
        """
        在一个完整的生命周期内, 循环执行"采样-训练-攻击(如有)-聚合-测试"这五个核心阶段
        """
        print('Training')
        accs = []
        asrs = [] # Attack Success Rate
        self.local_asrs = {} # 这个玩意在这里没用到
        # DIFF A3FL epoch -> for epoch in range(-2, self.helper.config["epochs"])
        for epoch in range(1, self.helper.config['epochs']+1):
            # sampled_participants 即每一轮中参与训练的客户端
            sampled_participants = self.sample_participants(epoch)

# DIFF ++++++++++++++++++++++++++++++++++++++++++++++++  DIFF  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # choose user id: [0, 71, 13, 69, 24, 55, 66, 76, 26, 64]
            print(f"choose user id: {sampled_participants}")
            time1 = time.time();
            # indicator预处理再下发模型 aggregator: 聚合
            # TOANSWER 这里的indicator是指: server利用指示机制的原理进行防御?给全局模型加水印
            # second 啥意思?
            if self.helper.config["agg_method"] == 'indicator' or self.helper.config["agg_method"] == 'second':
                self.aggregator.pre_process(self.helper.test_data, epoch)
#-------------------------------------------------------------------------------------------------------------- 
            # 执行一轮 train_once()
            weight_accumulator, weight_accumulator_by_client = self.train_once(epoch, sampled_participants)
# DIFF ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
            time2 = time.time();
            print(f"training time: {time2 - time1}s")
            # TOANSWER 为什么只在 self.helper.config["attacker_method"] == "sin-adv" 时添加噪声? 
            # DIFF 2/12 去除加噪 重复训练
            if self.helper.config["attacker_method"] == "sin-adv" and self.helper.config["is_poison"] and self.helper.config["noise"]:
                print(f"#########  into noise ##########")
                self.attacker.noise(self.helper.global_model, sampled_participants, epoch, weight_accumulator_by_client)

# ------------------------------------------------------------------------------------------------------------------------

            # 下面的 agg 函数我都不想看了, 应该是执行了服务器端聚合与安全检测策略
            """
            服务器执行聚合与防御. 它会根据 agg_method(如 FedAvg, DeepSight 等)对所有客户端的更新进行过滤、加权, 并更新 global_model
            """
            self.aggregator.agg(self.helper.global_model, weight_accumulator, weight_accumulator_by_client, self.helper.client_models, sampled_participants, epoch)
            # 测试聚合后的全局模型的表现(毒化/非毒化)
            loss, acc = self.test_once()
            bkd_loss, bkd_acc = self.test_once(poison=True)
            self.log_once(epoch, loss, acc, bkd_loss, bkd_acc)                        
            self.helper.record_acc(epoch, loss, acc, bkd_loss, bkd_acc, self.get_lr(epoch))
            # 记录trigger和mask(k: mask 记录了吗?哪呢?)
            # if epoch <= self.helper.config['poison_epochs'] and self.helper.config["attacker_method"] == "sin-adv":
            #     self.helper.record_trigger(epoch, self.attacker.trigger)
            accs.append(acc)
            asrs.append(bkd_acc)

            mia = "mia" if self.helper.config["mia"] else "no-mia"
            noise = "noise" if self.helper.config["noise"] else "no-noise"
            csv_log_path = f'{self.helper.config["folder_path"]}/{self.helper.config["dataset"]}_{self.helper.config["epochs"]}_{self.helper.config["current_time"]}_{self.helper.config["agg_method"]}_{self.helper.config["is_poison"]}_{mia}_{noise}_{self.helper.config["attacker_method"]}_DOBA_trajectory.csv'
            if self.helper.config["attacker_method"] != "sin-adv":
                csv_log_path = f'{self.helper.config["folder_path"]}/{self.helper.config["dataset"]}_{self.helper.config["epochs"]}_{self.helper.config["current_time"]}_{self.helper.config["agg_method"]}_{self.helper.config["is_poison"]}_{mia}_{noise}_{self.helper.config["attacker_method"]}_trajectory.csv'            
            
            # FOCUS 震荡调试逻辑
            if epoch == self.helper.config["poison_epochs"]:
                base_model_weights = get_flattened_weights(copy.deepcopy(self.helper.global_model.state_dict())).detach()
                print(f"--- Round {epoch}: 攻击停止，已保存基准模型权重 ---")
            if epoch > self.helper.config["poison_epochs"]:
                record_trajectory_to_csv(
                    round_idx=epoch,
                    current_sd=self.helper.global_model.state_dict(),
                    base_weights=base_model_weights,
                    asr=bkd_acc,      
                    main_acc=acc,  
                    csv_path=csv_log_path
                )                

        print(f"all attacker num {self.attack_sum}")

    def train_once(self, epoch, sampled_participants):
        """
        单轮的train, 详细内容太多了看注释, 下面是AI的总结, 还行\n
        单轮迭代的核心执行函数 它模拟了在一个训练轮次中, 服务器选中的客户端(包括良性客户端和可能的恶意攻击者)进行本地训练的过程 

        该函数的主要职责包括: 

        攻击准备: 如果本轮包含攻击者, 根据配置生成优化的触发器(如 sin-adv 或 a3fl 方法)或计算 Neurotoxin 攻击所需的梯度掩码(Mask) 

        模型分发: 将当前的全局模型参数下发给选中的参与者 

        本地训练: 

        良性客户端: 执行标准的本地监督学习

        恶意客户端: 根据指定的攻击算法(如 Chameleon, Neurotoxin, 或基础的投毒训练)执行恶意更新 

        更新收集: 汇总并计算每个客户端的权重更新量, 并支持"模型替换"

        持久化: 保存本地模型的状态及更新量, 以便后续聚合或分析 
        
        Return:\n
            weight_accumulator (dict): 全局累加器. 一个字典, 结构与模型参数一致, 存储了本轮所有选中客户端更新量的总和\n
            weight_accumulator_by_client (list of dict): 个体更新列表. 按顺序存储了每个被选中客户端独立的更新字典\n
        与 A3FL 相比, 仅在其基础上添加其他攻击的代码, A3FL部分代码一致, 但 train_malicious 的代码中使用的 poison_input_train / poison_input(A3FL)\n 
        具体而言, DOBA 的 poison_input_train 添加了针对 FCBA 的逻辑      
        """
        print(f"epoch {epoch} begin train")
        # 全局累加器, 用于汇总所有客户端的更新
        weight_accumulator = self.create_weight_accumulator()
        # 列表, 按顺序存储每个客户端独立的更新字典
        weight_accumulator_by_client = []
        # 计数器和用于记录本轮攻击者在列表中的索引位置
        client_count = 0
        attacker_idxs = []
        global_model_copy = self.create_global_model_copy()
        # 检查本轮选中的客户端列表中是否包含攻击者, 如果有, 返回第一个攻击者的 ID；如果没有, 返回 -1
        first_adversary = self.contain_adversary(epoch, sampled_participants)
        # 下面进行触发器的生成
        # 这里的 sin 的全称应该是 sin-adv
        if first_adversary >= 0 and ('sin' in self.helper.config["attacker_method"]):
            time1 = time.time();
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            # self.helper.train_data 是 self.train_data = train_loaders, 即针对每一个参与者(Participant)的索引数据, 批量生成对应的训练数据加载器(DataLoader)
            # CHANGE 2/15 该处将re暂时修改为 search_trigger 以查看实验结果
            self.attacker.search_trigger_re(model, self.helper.train_data[first_adversary], epoch)
            # CHANGE 2/19 修改为 search_trigger_re_a3fl(先运行DOBA优化, 再运行A3FL优化) 函数, 且加噪
            # TODO
            # self.attacker.search_trigger_re_a3fl(model, self.helper.train_data[first_adversary], epoch)
            time2 = time.time();
            print(f"optimize trigger ok, time: {time2-time1}s")
        # 和上面唯一的不同就在于触发器的生成, 我怀疑是直接复制过来的
        if first_adversary >= 0 and self.helper.config["attacker_method"] == 'a3fl':
            time1 = time.time();
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            self.attacker.search_trigger(model, self.helper.train_data[first_adversary], 'outter', first_adversary, epoch)
            time2 = time.time();
            print(f"optimize a3fl trigger ok, time: {time2 - time1}s")
        if first_adversary >= 0 and self.helper.config["attacker_method"] == 'reba':
            time1 = time.time();
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            if self.helper.config["trigger_pattern"] == 'adaptive':
                self.attacker.opt_ReBA(model, self.helper.train_data[first_adversary])
            time2 = time.time();
            print(f"optimize reba trigger ok, time: {time2 - time1}s")
        # 如果本轮存在攻击者, 打印攻击日志
        if first_adversary >= 0:
            self.attack_sum += 1
            print(f'Epoch {epoch}, poisoning by {first_adversary}, attack sum {self.attack_sum}.')
        else:
            # TOANSWER 没看懂, 为什么要特别判断需要是FCBA方法? first_adversary要么 >= 0 要么 == -1
            if self.helper.config["attacker_method"] != 'fcba':
                print(f'Epoch {epoch}, no adversary.')

        adv_index = -1
        # FOCUS 这里更实锤了攻击者就是 0 ~ num_adversaries - 1
        adv_num = [i for i in range(self.helper.config["num_adversaries"])]

        # 准备 neurotoxin 攻击需要的掩码
        if first_adversary >= 0 and self.helper.config["attacker_method"] == 'neurotoxin':
            print(f"neurotoxin mask_grad_list")
            time1 = time.time();
            # 决定攻击者用来模拟良性更新的"干净数据块"数量
            num_clean_data = 30
            # TOANSWER  是因为 mnist 是很简单的任务吗?
            if self.helper.config['dataset']=='mnist':
                num_clean_data = 15
            # 存储了被选中用于模拟良性更新的数据分片编号, 即从参与者的ID列表中随机选取 num_clean_data 个参与者, 后面用他们的数据
            subset_data_chunks_mask = random.sample(range(1, self.helper.config['num_total_participants']),
                                                    num_clean_data)
            # sampled_data 是一个DataLoader列表, 包含了由 subset_data_chunks_mask 决定的各个参与者的数据加载器
            sampled_data = [self.helper.train_neur[pos] for pos in subset_data_chunks_mask]
            # 通过分析良性数据在模型上产生的梯度, 识别出那些"不活跃"(更新频率低、梯度幅值小)的参数, 并生成一个掩码( mask_grad_list, 结构与 self.helper.global_model 参数结构一致 ) 攻击者随后只在这些掩码标记的参数上植入后门, 以确保后门不会被良性训练覆盖, 从而实现持久性 
            mask_grad_list = self.grad_mask_cv(self.helper.global_model, sampled_data, ratio=0.95)
            print(f"neurotoxin mask_grad_list: {time.time()-time1}")

        # FOCUS 训练开始, 按参与者ID在 sampled_participants 中的顺序进行(注意: 这里的 sampled_participants 中仅可能存在攻击者)
        for participant_id in sampled_participants:
            model = self.helper.local_model
            # 实际上的模型下发步骤
            self.copy_params(model, global_model_copy)
            # 无攻击者的正常训练
            if not self.if_adversary(epoch, participant_id) and adv_index == -1:
                model.train()
                print(f"epoch {epoch} client {participant_id} benign train")
                self.train_benign(participant_id, model, epoch)       
            # chameleon 和 neurotoxin 或者 一般的恶意攻击训练
            else:
                # print(f"[*] klog !!! then? ")
                print(f"epoch {epoch} client {participant_id} malicious train")
                attacker_idxs.append(client_count)
                if self.helper.config["attacker_method"] == 'cham':
                    # model.train()记得注释掉(非k写)
                    if self.helper.config["mia"]:
                        self.train_mia(participant_id, model, epoch)                    
                    self.chameleon_train(participant_id, model, epoch)
                elif self.helper.config["attacker_method"] == 'neurotoxin':
                    model.train()
                    if self.helper.config["mia"]:
                        self.train_mia(participant_id, model, epoch)
                    self.neurotoxin_train(participant_id, model, epoch, mask_grad_list)
                # elif self.helper.config["attacker_method"] == 'reba':
                #     model.train()
                #     if self.helper.config["mia"]:
                #         self.train_mia(participant_id, model, epoch)
                #     self.reba_train(participant_id, model, epoch)    
                elif self.helper.config["attacker_method"] == 'reba':
                    attacker_idxs.append(client_count)
                    if self.helper.config["mia"]:
                        self.train_mia(participant_id, model, epoch)                
                    self.train_ReBA(participant_id, model, epoch)                                     
                else:
                    model.train()
                    # 英改
                    # if 'sin' in self.helper.config["attacker_method"]:
                    #     self.train_mia(participant_id,model,epoch)
                    # 噪声训练 FOCUS 
                    # FOCUS 模块重点 !!! 
                    # CHANGE
                    if self.helper.config["mia"]:
                        self.train_mia(participant_id, model, epoch)
                    # 基本后门恶意训练
                    self.train_malicious(participant_id, model, epoch)

            weight_accumulator, single_wa, modelreplace_weight, fcba_weight = self.update_weight_accumulator(model, weight_accumulator, adv_index)

            # 模型替换攻击
            if self.helper.config["attacker_method"] == 'modelreplace' and participant_id in adv_num:
                single_wa = modelreplace_weight
            # 将该客户端的更新存入列表(按照 sampled_participants 的顺序)
            weight_accumulator_by_client.append(single_wa)
            self.helper.client_models[participant_id].load_state_dict(model.state_dict())
            client_count += 1
            # 对于cifar100_helper, 保存模型至self.helper.config["folder_path"]/saved_updates/update_{id}.pth
            # 实际上该路径存储了所有参与者的更新(无论是否恶意)
            self.helper.save_update(model=single_wa, userID=participant_id)
            # self.save_local_model(participant_id, model, epoch, first_adversary)

        # cham   sin-adv  加noise的时候需要用 self.helper.config["attacker_method"] == 'cham' or  (非 k 写)
        # 执行self.helper.config["attacker_method"] == 'sin-adv' 时的攻击(上面涉及该攻击方法的代码只进行了触发器的生成)
        # 但是并不在 weight_accumulator, weight_accumulator_by_client 中记录更新, 但是会调用 self.helper.save_update 保存更新量至 self.helper.config["folder_path"]/saved_updates
        if epoch <= self.helper.config["poison_epochs"] and self.helper.config["is_poison"] and \
                (self.helper.config["attacker_method"] == 'sin-adv'):
            for i in range(self.helper.config["num_adversaries"]):
                if i in sampled_participants:
                    continue
                model = self.helper.local_model
                self.copy_params(model, global_model_copy)
                model.train()
                # 仅执行简单的投毒
                self.train_malicious(i, model, epoch)
                update_wa = self.update_weight(model)
                self.helper.save_update(model=update_wa, userID=i)
                # self.save_local_model(i, model)
        return weight_accumulator, weight_accumulator_by_client

    def update_weight(self, model):
        """
        计算并返回模型参数的更新量
        """
        single_weight_accumulator = dict()
        for name, data in model.state_dict().items():
            if name == 'decoder.weight' or '__'in name:
                continue
            single_weight_accumulator[name] = data - self.helper.global_model.state_dict()[name]
        return single_weight_accumulator

    # fcba需要改
    def contain_adversary(self, epoch, sampled_participants):
        """
        TOANSWER 不知道这个函数是不是我认为的作用和过程\n
        逆天函数, 所有变量全靠猜\n
        检测当前轮次被选中的客户端中是否包含攻击者, 如果包含攻击者, 它会sampled_participants中的第一个(是索引上的第一个)攻击者的ID; 如果不包含, 则返回 -1\n
        is_poison: 表示整个实验是否配置了后门攻击\n
        poison_epochs: 到底是攻击持续的轮次范围, 还是开始攻击的轮次\n
        num_adversaries: 攻击者的总数, 0-4为攻击者?\n
        """
        if self.helper.config["is_poison"] and \
            epoch < self.helper.config["poison_epochs"] and epoch >= 0 :
            if self.helper.config["sample_method"] == 'random' or self.helper.config["sample_method"] == 'fix':
                # FOCUS sampled_participants 是无序的
                for p in sampled_participants:
                    if p < self.helper.config["num_adversaries"]:
                        return p
            elif self.helper.config["sample_method"] == 'random_updates':
                # TOANSWER self.advs 又是什么变量, 整个文件这里第一次出现
                # 从代码上看, self.advs是一个存储了预先分配好的每一轮攻击者 ID 的字典?
                assert(False)
                if epoch in self.advs:
                    return self.advs[epoch][0]
        return -1

    def if_adversary(self, epoch, participant_id):
        """
        判断一个特定的客户端是否是攻击者, 攻击者应该就是 0 ~ num_adversaries - 1
        
        是攻击者返回True
        """
        if self.helper.config["is_poison"] and epoch < self.helper.config["poison_epochs"] and epoch >= 0:
            if self.helper.config["sample_method"] == 'random' and participant_id < self.helper.config["num_adversaries"]:
                return True
            elif self.helper.config["sample_method"] =='fix' and participant_id < self.helper.config['num_adversaries']:
                return True
        else:
            return False

    def create_local_model_copy(self, model):
        """"
        为传入的特定本地模型创建一个不参与梯度的参数副本
        """
        model_copy = dict()
        for name, param in model.named_parameters():
            model_copy[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
        return model_copy

    def create_global_model_copy(self):
        """
        为中心服务器当前的全局模型(self.helper.global_model)创建一个参数副本
        """
        global_model_copy = dict()
        for name, param in self.helper.global_model.named_parameters():
            global_model_copy[name] = self.helper.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy

    def create_weight_accumulator(self):
        """
        初始化一个用于累加模型更新的累加器
        """
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            # 忽略解码器或私有属性
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator
    
    def update_weight_accumulator(self, model, weight_accumulator, adv_index):
        """
        计算并返回客户端训练后的模型更新, 各种更新都返回\n
        Params: \n
            model: 当前客户端训练完成后的本地模型对象(model = self.helper.local_model)\n
            weight_accumulator: 全局的累加器字典, 用于存储本轮所有客户端更新的总和 (weight_accumulator = self.create_weight_accumulator())\n
            adv_index: 用于指定FCBA的攻击模式(位置), 见本文件 neurotoxin_train() 函数中的 FOCUS \n
        Return: \n
            已更新的全局累加器 单体更新量(用于正常聚合或记录) 缩放后的模型替换权重 缩放后的 FCBA 权重 
        """
        # 存储原始更新量的字典, 即本地模型与全局模型之间的直接差值
        single_weight_accumulator = dict()
        # 专门为模型替换攻击计算的字典, 将更新量放大了self.helper.config["fl_weight_scale"]
        modelreplace_weight = dict()
        # 专门为 FCBA 攻击 计算的字典
        fcba_weight = dict()
        for name, data in model.state_dict().items():
            if name == 'decoder.weight' or '__'in name:
                continue
            # 与集中式对比的时候
            weight_accumulator[name].add_(data - self.helper.global_model.state_dict()[name])
            single_weight_accumulator[name] = data - self.helper.global_model.state_dict()[name]
            # 下面两行的区别就是*的东西不一样
            """
            fl_weight_scale: 80
            scale_weights_poison: 50
            """
            fcba_weight[name] = (data - self.helper.global_model.state_dict()[name]) * self.helper.config["scale_weights_poison"]
            # TODO modelreplace 攻击修改: 1. * 10 2. 学习率问题 
            modelreplace_weight[name] = (data - self.helper.global_model.state_dict()[name])*self.helper.config["fl_weight_scale"]
            # modelreplace_weight[name] = (data - self.helper.global_model.state_dict()[name])
        return weight_accumulator, single_weight_accumulator, modelreplace_weight, fcba_weight

    def train_benign(self, participant_id, model, epoch):
        """
        正常模型的训练过程
        """
        # 😓什么Ninja Code风格
        for params in model.named_parameters():
            params[1].requires_grad = True
        lr = self.get_lr(epoch)  # cifar10
        # lr = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config["momentum"],
            weight_decay=self.helper.config["decay"])
        # yaml : retrain_times: 2
        # print(f"[*] klog participant_id -> {participant_id}\nlen(self.helper.train_data) == {len(self.helper.train_data)}")
        for internal_epoch in range(self.helper.config["retrain_times"]):
            sample_dataset = self.helper.train_data[participant_id].dataset
            print("Dataset type:", type(sample_dataset))
            sample_img, sample_label = sample_dataset[0]
            print("Image type before transforms:", type(sample_img))
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = self.criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # loss, acc = self.test_local_once(model)
        # bkd_loss, bkd_acc = self.test_local_once(model, True)
        # self.helper.record_train_acc(epoch, participant_id, 0, loss, acc, bkd_loss, bkd_acc)

    def scale_up(self, model, curren_num_adv):
        clip_rate = 2/curren_num_adv
        for key, value in model.state_dict().items():
            if  key == 'decoder.weight' or '__'in key:
                continue
            target_value = self.helper.global_model.state_dict()[key]
            new_value = target_value + (value - target_value) * clip_rate

            model.state_dict()[key].copy_(new_value)
        return model

    def train_mia(self, participant_id, model, epoch):
        """
        在 self.attacker.miadate(两种方式构造的噪声集) 决定的数据集上对model进行噪声训练
        """
        lr = self.get_lr(epoch)
        #lr = 0.005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config["momentum"],
                                    weight_decay=self.helper.config["decay"])
        if self.helper.config["is_poison"] == False:
            assert(False)
            participant_id = random.randint(0,4)
            # TOANSWER  这里的 self.attacker.miadate 是哪来的? attacker.py -> getdata()/getdata2()都能
        for i, client_loader in enumerate(self.attacker.miadate):
            if i == participant_id:
                for internal_epoch in range(self.helper.config["retrain_times"]):
                    index = 0
                    for inputs, labels in client_loader:
                        inputs, labels = inputs.cuda(), labels.cuda()
                        # print(f" [*] klog DEBUG: inputs shape: {inputs.shape}")
                        output = model(inputs)
                        # print(f"[*] klog : DUBG  output shape: {output.shape}, labels shape: {labels.shape}")
                        """
                         [*] klog DEBUG: inputs shape: torch.Size([32, 1, 32, 32])
                        [*] klog : DUBG  output shape: torch.Size([50, 10]), labels shape: torch.Size([32])
                        """
                        loss = self.attacker_criterion(output, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        index += 1
        print("#########  train mia ###############")

    def train_malicious(self, participant_id, model, epoch):
        """
        本地投毒训练, 直接通过篡改本地数据集(植入触发器并翻转标签)来训练模型
        """
        adv_index = -1
        # lr = self.get_lr(epoch)/5  # cifar10 攻击
        lr = self.get_lr(epoch)
        # lr = 0.05   #0.05  0.01
        # lr = 0.1  # fenbushi 0.05可以
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config["momentum"],
            weight_decay=self.helper.config["decay"])
        for internal_epoch in range(self.helper.config["attacker_retrain_times"]):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                # DIFF 
                inputs, labels = self.attacker.poison_input_train(inputs, labels, adv_index)
                output = model(inputs)
                loss = self.attacker_criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # clean_model = self.helper.global_model
        # target_params_variables = dict()
        # for name, param in clean_model.named_parameters():
        #     target_params_variables[name] = clean_model.state_dict()[name].clone().detach().requires_grad_(False)
        #
        # model_norm = self.helper.model_dist_norm(model, target_params_variables)
        # self.helper.record_norm(epoch, participant_id, model_norm)
        # loss, acc = self.test_local_once(model)
        # bkd_loss, bkd_acc = self.test_local_once(model, True)
        # self.helper.record_train_acc(epoch, participant_id, 1, loss, acc, bkd_loss, bkd_acc)

        # FOCUS ReBA


    # def reba_train(self, participant_id, model, epoch):
    #     """
    #     ReBA train
    #     """
    #     # TODO trigger 还没用到
    #     best_trigger_patch = self.optimize_atp(model, k_atp=50)
    #     self.attacker.reba_trigger = best_trigger_patch
    #     print("ATP Trigger 优化完成。")
    #     self.compute_update_mask(model, participant_id)
    #     print("持久性重要度掩码 Phi 计算完成。")
    #     best_kappa = self.adaptive_search_kappa(epoch, model, self.helper.global_model, self.helper.train_data[participant_id])
    #     print(f"自适应搜索完成，当前最优放大倍数 kappa*: {best_kappa}")


    def reba_train(self, participant_id, model, epoch):
        """
        ReBA train: 整合 ATP, Phi, Kappa 并直接修改 model 权重返回
        """
        # ATP 优化 (5*5 触发器)
        best_trigger_patch = self.optimize_atp(model, self.helper.train_data[participant_id], k_atp=50)
        self.attacker.reba_trigger = best_trigger_patch
        print("ATP Trigger 优化完成。")

        # 计算持久性掩码 Phi (self.phi)
        self.compute_update_mask(model, participant_id)
        # print(f"持久性重要度掩码 Phi 计算完成。数值为 \n {self.phi}")

        # 搜索最优放大因子 kappa
        best_kappa = self.adaptive_search_kappa(epoch, model, participant_id, self.helper.global_model, self.helper.train_data[participant_id])
        print(f"自适应搜索完成，当前最优放大倍数 kappa*: {best_kappa}")

        # 获取后门梯度 g_back 
        g_back_dict = self._get_backdoor_gradient(model, participant_id)

        # 核心公式：θ_back = θ_G - η * κ * Φ * g_back
        lr = self.get_lr(epoch)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in g_back_dict:
                    phi_w = self.phi[name]
                    
                    # 计算恶意更新项: η * κ * Φ * g_back
                    malicious_update = lr * best_kappa * phi_w * g_back_dict[name]
                    
                    # 直接修改模型权重：θ_new = θ_old - malicious_update
                    param.copy_(param.data - malicious_update)



    def optimize_atp(self, model, benign_loader, k_atp=100, eps=0.05):
        """
        TOANSWER 论文里写的 t j = t j−1 − η∇θe  G L.\n
        这是不是有问题，Loss 关于全局模型参数的梯度? 参数又不变\n
        步骤 1: 对抗性触发器优化 (ATP)
        L = 1/|Db| * sum(CE(xt, ye))
        """
        # 初始化 5*5 的触发器 (C, H, W) -> (3, 5, 5)
        trigger = torch.randn((3, 5, 5), requires_grad=True, device=self.device)
            
            # 这里我们仍然需要优化 t，所以优化器还是针对 trigger
        optimizer_t = torch.optim.SGD([trigger], lr=0.01)
        
        model.eval() # 固定全局模型 theta_G
        
        for j in range(k_atp): # for j = 1 to Katp
            for images, _ in benign_loader:
                images = images.to(self.device)
                
                # 构造 xt (将触发器贴到良性样本上)
                poisoned_images = images.clone()
                poisoned_images[:, :, :5, :5] = torch.tanh(trigger) 
                
                # 7: 计算损失 L = CE(xt, ye)
                outputs = model(poisoned_images)
                labels = torch.full((images.size(0),), self.helper.config['target_class'], device=self.device)
                loss = F.cross_entropy(outputs, labels)
                
                # 8: 更新触发器
                optimizer_t.zero_grad()
                loss.backward() # 这会计算出 dL/dt
                optimizer_t.step() # t = t - eta * grad
                
        return torch.tanh(trigger).detach()

    def compute_update_mask(self, model, participant_id):
        """
        步骤 2: 计算持久性重要度 (SD) 并生成 Phi
        Phi = 1 + (SD - min) / (max - min)
        """
        model.eval()
        sd_dict = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        
        # 计算平方梯度 (Fisher Information )
        for images, labels in self.helper.train_data[participant_id]:
            images = images.to(self.device)
            labels = labels.to(self.device)
            images, labels = self.attacker.reba_poison_input_train(images, labels)
            outputs = model(images)
            labels = torch.full((images.size(0),), self.helper.config['target_class'], device=self.device)
            loss = F.cross_entropy(outputs, labels)
            
            grads = torch.autograd.grad(loss, model.parameters())
            for (name, param), grad in zip(model.named_parameters(), grads):
                sd_dict[name] += (grad.data ** 2)
        
        # Min-Max Scaling 映射到 [1, 2]
        phi_dict = {}
        for name, sd in sd_dict.items():
            sd_min = sd.min()
            sd_max = sd.max()
            phi_dict[name] = 1.0 + (sd - sd_min) / (sd_max - sd_min + 1e-9)
            
        self.phi = phi_dict
        return phi_dict

    def adaptive_search_kappa(self, epoch, model, participant_id, global_model, benign_loader, eta_kappa=0.1):
        """
        步骤 3: 自适应搜索放大因子 kappa
        kappa = kappa - eta * grad_kappa(||D_dummy - D_back||)
        """
        # 模拟良性更新 (D_dummy)
        # 计算良性更新距离基准 D_dummy (Line 14-16)
        theta_G = copy.deepcopy(global_model)
        theta_dummy = self._simulate_benign_update(copy.deepcopy(global_model), benign_loader)
        dist_dummy = self._get_model_dist(theta_G, theta_dummy)

        # 获取后门梯度 g_back (Line 11-12)
        g_back = self._get_backdoor_gradient(copy.deepcopy(global_model), participant_id)

        # 开启 Until 循环 (Line 19-22)
        max_search_steps = 200
        step = 0
        while True:
            print(f"[*] klog epoch | step --> {epoch} | {step}")
            # --- Line 19: 生成当前 kappa 下的攻击模型 ---
            dist_back_sq = 0
            for name, p in theta_G.named_parameters():
                # 恶意更新量：eta * kappa * phi * g_back
                update_vec = self.get_lr(epoch) * self.kappa * self.phi[name] * g_back[name]
                dist_back_sq += torch.sum(update_vec ** 2)
            
            # --- Line 20: 计算当前距离 Dback ---
            dist_back = torch.sqrt(dist_back_sq + 1e-10)

            # --- Line 22: Until 终止条件判断 ---
            if dist_back.item() <= dist_dummy:
                print("[*] woooooooooooooo ")
            if dist_back.item() <= dist_dummy or step >= max_search_steps:
                break

            # --- Line 21: 根据距离差值更新 kappa ---
            # 损失函数目标：让 Dback 逼近 Ddummy
            loss_kappa = torch.abs(dist_dummy - dist_back)
            
            # 清除之前的梯度
            if self.kappa.grad is not None:
                self.kappa.grad.zero_()
                
            loss_kappa.backward()
            
            with torch.no_grad():
                # 沿梯度方向调整 kappa：kappa = kappa - η * ∇κ
                self.kappa -= eta_kappa * self.kappa.grad
                # 物理约束，kappa 不能为负或过小
                self.kappa.clamp_(min=0.1)
            
            step += 1

        # --- Line 23: 找到最优 κ* ---
        return self.kappa.item()

    def _simulate_benign_update(self, model, loader):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.helper.config["lr"])
        model.train()
        for internal_epoch in range(self.helper.config["retrain_times"]):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
        return model

    # def _get_backdoor_gradient(self, model, participant_id):
    #     model.eval()
    #     images, labels = next(iter(self.helper.train_data[participant_id]))
    #     images = images.to(self.device)
    #     labels = labels.to(self.device)
    #     images, labels = self.attacker.poison_input_train(images, labels, -1)
    #     labels = torch.full((images.size(0),), self.helper.config['target_class'], device=self.device)
    #     loss = F.cross_entropy(model(images), labels)
    #     grads = torch.autograd.grad(loss, model.parameters())
    #     return {name: g.data for (name, p), g in zip(model.named_parameters(), grads)}

    def _get_backdoor_gradient(self, model, participant_id, num_batches=5):
        model.eval()
        all_grads = None
        count = 0
        
        # 迭代多次取样
        data_iter = iter(self.helper.train_data[participant_id])
        for _ in range(num_batches):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            # 中毒处理
            images, _ = self.attacker.reba_poison_input_train(images, labels)
            # 强制目标标签
            target_labels = torch.full((images.size(0),), self.helper.config['target_class'], device=self.device)
            
            loss = F.cross_entropy(model(images), target_labels)
            grads = torch.autograd.grad(loss, model.parameters())
            
            # 累加梯度
            if all_grads is None:
                all_grads = [g.detach() for g in grads]
            else:
                for i, g in enumerate(grads):
                    all_grads[i] += g.detach()
            count += 1
            
        # 取平均
        avg_grads = [g / count for g in all_grads]
        return {name: g.data for (name, p), g in zip(model.named_parameters(), avg_grads)}    

    def _get_model_dist(self, model_a, model_b):
        dist = 0
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            dist += torch.norm(p_a - p_b)**2
        return torch.sqrt(dist).item()        






    def neurotoxin_train(self, participant_id, model, epoch, mask_grad_list):
        """
        实现了 Neurotoxin 攻击在本地训练阶段的核心逻辑
        
        利用预先计算好的掩码 mask_grad_list 来限制梯度的更新方向, 从而将后门植入到那些良性更新不频繁的参数中\n
        Params:\n
            mask_grad_list: 预先计算好的掩码列表, 见fler.py/grad_mask_cv()函数
        """
        # FOCUS 这个 adv_index 最终在 attacker.py/get_pattern() 函数中使用, 应该是 FCBA 中用于指定后门植入的模式(应该是位置信息)
        print(f"[*] klog enter neurotoxin_train")
        # print(f"[**] klog  self.attacker.trigger: {self.attacker.trigger} ")
        adv_index = -1
        # lr = 0.05  # 0.025  0.1
        lr = 0.1
        # lr = 0.01
        # lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config["momentum"],
                                    weight_decay=self.helper.config["decay"])
        # TOANSWER  attacker_retrain_times 应该是指攻击训练的轮数, 这玩意不同攻击会有相同的轮数?
        for internal_epoch in range(self.helper.config["attacker_retrain_times"]):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                # adv_index 是定值 -1 ?
                inputs, labels = self.attacker.poison_input_train(inputs, labels, adv_index)
                output = model(inputs)
                # 就是 CrossEntropyLoss 损失函数
                loss = self.attacker_criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                self.apply_grad_mask(model, mask_grad_list)
                optimizer.step()
        # loss, acc = self.test_local_once(model)
        # bkd_loss, bkd_acc = self.test_local_once(model, True)
        # self.helper.record_train_acc(epoch, participant_id, 0, loss, acc, bkd_loss, bkd_acc)

    # 对比方法 Chameleon 使用对比学习 对等图像的思想
    def chameleon_train(self, participant_id, model, epoch):
        """
        TODO 未细看
        替换模型参数为"对比学习"和"分类训练"后的恶意参数, 并放大权重(new_value = target_value + (value - target_value) * self.helper.config["num_adversaries"])
        """
        global_model = copy.deepcopy(model)
        for params in model.named_parameters():
            params[1].requires_grad = True
        # self.train_mia(participant_id, model, epoch)
        target_params_variables = dict()
        for name, param in model.named_parameters():
            target_params_variables[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
        contrastive_model = self.helper.contrastive_model
        contrastive_model.copy_params(model.state_dict())
        # 恶意客户端可操作的数据
        poisoned_data = self.helper.train_data[participant_id]
        retrain_no_times = self.helper.config['retrain_poison_contrastive']
        step_lr = self.helper.config['poison_step_lr_contrastive']
        poison_lr = self.helper.config['poison_lr_contrastive']
        # 优化器
        poison_optimizer_contrastive = torch.optim.SGD(
            filter(lambda p: p.requires_grad, contrastive_model.parameters()), lr=poison_lr,
            momentum=self.helper.config['momentum_contrastive'],
            weight_decay=self.helper.config['decay_contrastive'])
        # 学习率调度器
        scheduler_contrastive = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer_contrastive,
                                                                     milestones=self.helper.config[
                                                                         'milestones_conrtastive'],
                                                                     gamma=self.helper.config['lr_gamma_contrastive'])
        print(f'嵌入训练')
        for internal_epoch in range(1, retrain_no_times + 1):
            if step_lr:
                scheduler_contrastive.step()
            data_iterator = poisoned_data
            for inputs, labels in (data_iterator):
                inputs, labels = inputs.cuda(), labels.cuda()
                # print(f"inputs: {len(inputs)}")
                bkd_num = int(self.helper.config['bkd_ratio'] * len(inputs))
                # print(f'bkd_num: {bkd_num}   times:{internal_epoch}')
                for pos in range(bkd_num):
                    poison_pos = pos
                    inputs[poison_pos] = self.attacker.add_triggers(inputs[poison_pos])
                    noise = torch.FloatTensor(inputs[poison_pos].shape).normal_(0,self.helper.config['noise_level']).to(inputs.device)
                    inputs[poison_pos].add_(noise)
                    inputs = inputs.cuda()
                    labels[poison_pos] = self.helper.config['target_class']
                    # print(inputs)
                data, targets = inputs.cuda(), labels.cuda()
                poison_optimizer_contrastive.zero_grad()
                # data = torch.clamp(data, 0, 1)  # 将数据限制在[0, 1]范围内
                output = contrastive_model(data)
                # print(f"label {labels}")
                # print(f"output: {output}")
                suploss = self.SupConLoss
                contrastive_loss = suploss(output, targets,
                                           poison_per_batch=self.helper.config['poisoning_per_batch_contrastive'],
                                           poison_images_len=bkd_num,
                                           scale_weight=self.helper.config['contrastive_loss_scale_weight'],
                                           down_scale_weight=self.helper.config['contrastive_loss_down_scale_weight'],
                                           helper=self.helper)
                loss = self.helper.config['contrastive_loss_weight'] * contrastive_loss
                loss.backward()
                if self.helper.config['diff_privacy_contrastive']:
                    poison_optimizer_contrastive.step()
                    model_norm = self.helper.model_dist_norm(contrastive_model, target_params_variables)
                    if model_norm > self.helper.config['s_norm_contrastive']:
                        norm_scale = self.helper.config['s_norm'] / ((model_norm))
                        for name, layer in contrastive_model.named_parameters():
                            # don't scale tied weights:
                            if self.helper.config.get('tied', False) and name == 'decoder.weight' or '__' in name:
                                continue
                            clipped_difference = norm_scale * (layer.data - global_model.state_dict()[name])
                            layer.data.copy_(global_model.state_dict()[name] + clipped_difference)
                else:
                    poison_optimizer_contrastive.step()

        model.copy_params(contrastive_model.state_dict())
        poisoned_data = self.helper.train_data[participant_id]

        # params E in local training 开始训练分类器
        retrain_no_times = self.helper.config['retrain_poison']
        step_lr = self.helper.config['poison_step_lr']

        poison_lr = self.helper.config['poison_lr']
        #  posion_opt and regular_opt are different in learning rate 编码器不参与反向传播
        # imagenet用fc   其他用linear  mnist用fc2
        if self.helper.config['is_frozen_params']:
            for params in model.named_parameters():
                if params[0] != 'linear.weight' and params[0] != 'linear.bias':
                # if params[0] != 'fc2.weight' and params[0] != 'fc2.bias':
                # if params[0] != 'fc.weight' and params[0] != 'fc.bias':
                    # print(params[1])
                    params[1].requires_grad = False
                # if params[0] != 'fc1.weight' and params[0] != 'fc1.bias':
                #     params[1].requires_grad = False

        poison_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=poison_lr,
                                           momentum=self.helper.config['momentum'],
                                           weight_decay=self.helper.config['decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                         milestones=self.helper.config['milestones'],
                                                         gamma=self.helper.config['lr_gamma'])
        print(f'分类训练')
        for internal_epoch in range(1, retrain_no_times + 1):
            if step_lr:
                scheduler.step()
            data_iterator = poisoned_data
            for inputs, labels in data_iterator:
                inputs, labels = inputs.cuda(), labels.cuda()
                bkd_num = int(self.helper.config['bkd_ratio'] * len(inputs))
                for pos in range(bkd_num):
                    poison_pos = pos
                    inputs[poison_pos] = self.attacker.add_triggers(inputs[poison_pos])
                    noise = torch.FloatTensor(inputs[poison_pos].shape).normal_(0,
                                                                                self.helper.config['noise_level']).to(
                        inputs.device)
                    inputs[poison_pos].add_(noise)
                    inputs = inputs.cuda()
                    labels[poison_pos] = self.helper.config['target_class']
                data, targets = inputs.cuda(), labels.cuda()
                poison_optimizer.zero_grad()
                output = model(data)
                class_loss = self.criterion(output, targets)
                distance_loss = self.helper.model_dist_norm_var(model, target_params_variables)
                loss = self.helper.config['alpha_loss'] * class_loss + (
                            1 - self.helper.config['alpha_loss']) * distance_loss
                loss.backward()
                if self.helper.config['diff_privacy']:
                    poison_optimizer.step()
                    model_norm = self.helper.model_dist_norm(model, target_params_variables)
                    if model_norm > self.helper.config['s_norm']:
                        norm_scale = self.helper.config['s_norm'] / ((model_norm))
                        for name, layer in model.named_parameters():
                            # don't scale tied weights:
                            if self.helper.config.get('tied', False) and name == 'decoder.weight' or '__' in name:
                                continue
                            clipped_difference = norm_scale * (
                                    layer.data - global_model.state_dict()[name])
                            layer.data.copy_(
                                global_model.state_dict()[name] + clipped_difference)
                else:
                    poison_optimizer.step()

        clip_rate = (self.helper.config['scale_weights'] / self.helper.config["num_adversaries"])
        print(f"Scaling by  {clip_rate}")
        for key, value in model.state_dict().items():
            if self.helper.config.get('tied', False) and key == 'decoder.weight' or '__' in key:
                continue
            target_value = global_model.state_dict()[key]
            new_value = target_value + (self.helper.config['new_model_scale_weights'] * value - target_value) * \
                        self.helper.config['scale_weights']
            model.state_dict()[key].copy_(new_value)
        if self.helper.config['diff_privacy']:
            model_norm = self.helper.model_dist_norm(model, target_params_variables)
            if model_norm > self.helper.config['s_norm']:
                norm_scale = self.helper.config['s_norm'] / (model_norm)
                for name, layer in model.named_parameters():
                    # don't scale tied weights:
                    if self.helper.config.get('tied', False) and name == 'decoder.weight' or '__' in name:
                        continue
                    clipped_difference = norm_scale * (layer.data - global_model.state_dict()[name])
                    layer.data.copy_(global_model.state_dict()[name] + clipped_difference)

        for key, value in model.state_dict().items():
            if self.helper.config.get('tied', False) and key == 'decoder.weight' or '__' in key:
                continue
            target_value = global_model.state_dict()[key]
            new_value = target_value + (value - target_value) * self.helper.config["num_adversaries"]
            model.state_dict()[key].copy_(new_value)


    def model_l2_distance(self, model1, model2):
        l2_distance = 0.0
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        for p1, p2 in zip(params1, params2):
            l2_distance += torch.sum((p1 - p2.detach()) ** 2)
        return torch.sqrt(l2_distance)
    def torch_l2_distance(self, t1, t2):

        l2_distance = torch.sum((t1 - t2) ** 2)
        return torch.sqrt(l2_distance)



    def Adaptive_kappa(self, model, indices, participant_id, lr):
        k = torch.ones(1, device='cuda') * self.helper.config["initial_kappa"]
        k.requires_grad_()
        optimizer_k = torch.optim.SGD([k, ], lr=0.05)

        res_ori = torch.tensor([]).cuda()
        for v in model.parameters():
            v = v.view(1, -1).squeeze()
            res_ori = torch.cat([res_ori.squeeze(), v], 0).view(-1)
        backdoor_model = copy.deepcopy(model)
        optimizer_backdoor = torch.optim.SGD(backdoor_model.parameters(), lr=lr, momentum=self.helper.config["momentum"],
                                             weight_decay=self.helper.config["decay"])
        clean_model = copy.deepcopy(model)
        optimizer_clean = torch.optim.SGD(clean_model.parameters(), lr=lr, momentum=self.helper.config["momentum"],
                                          weight_decay=self.helper.config["decay"])
        res_bkd = torch.tensor([]).cuda()
        res_cl = torch.tensor([]).cuda()
        for idx in range(self.helper.config["retrain_clean_model"]):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = clean_model(inputs)
                loss = self.criterion(output, labels)
                optimizer_clean.zero_grad()
                loss.backward()
                optimizer_clean.step()

        for inputs, labels in self.helper.train_data[participant_id]:
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = self.attacker.poison_input(inputs, labels)
            output = backdoor_model(inputs)
            loss = self.attacker_criterion(output, labels)
            optimizer_backdoor.zero_grad()
            loss.backward()
            self.reba_apply_grad_mask(backdoor_model, indices,1)
            optimizer_backdoor.step()


        for v in backdoor_model.parameters():
            v = v.view(1, -1).squeeze()
            res_bkd = torch.cat([res_bkd.squeeze(), v], 0).view(-1)
        res_bkd_ori = (res_ori - res_bkd)
        res_bkd_ori=res_bkd_ori.detach()


        for v in clean_model.parameters():
            v = v.view(1, -1).squeeze()
            res_cl = torch.cat([res_cl.squeeze(), v], 0).view(-1)
        res_cl = (res_ori - res_cl).detach()

        glo_l2 = self.model_l2_distance(model, clean_model)
        glo_l2 = glo_l2.detach()
        for i in range(200):

            res_bkd = res_bkd_ori * k
            loss_L2 = self.torch_l2_distance(res_bkd, torch.full_like(res_bkd, 0).detach()) - self.torch_l2_distance(res_cl, torch.full_like(res_cl, 0).detach())
            optimizer_k.zero_grad()
            loss_L2.backward()
            optimizer_k.step()
            backdoor_model_test = copy.deepcopy(model)
            optimizer_backdoor_copy = torch.optim.SGD(backdoor_model_test.parameters(), lr=lr,
                                                      momentum=self.helper.config["momentum"],
                                                      weight_decay=self.helper.config["decay"])
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.attacker.poison_input(inputs, labels)
                output = backdoor_model_test(inputs)
                loss = self.attacker_criterion(output, labels)
                optimizer_backdoor_copy.zero_grad()
                loss.backward()
                self.reba_apply_grad_mask(backdoor_model_test, indices,k)
                optimizer_backdoor_copy.step()

            loss_global = self.model_l2_distance(backdoor_model_test, model)
            if loss_global.detach().cpu() < glo_l2.cpu():
                break

        return k




    def train_ReBA(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config["momentum"],
                                    weight_decay=self.helper.config["decay"])
        clean_model = copy.deepcopy(model)
        optimizer_clean = torch.optim.SGD(clean_model.parameters(), lr=lr,
                                          momentum=self.helper.config["momentum"],
                                          weight_decay=self.helper.config["decay"])


        fisher_matrix = self.attacker.get_fisher(model, self.helper.train_data[participant_id], 1)
        Durable_Importance = self.normalize_vector(fisher_matrix,1)
        Durable_Importance = self.attacker.reshape_DI(model, Durable_Importance)

        if self.helper.config["adaptive_kappa"]:
            K = self.Adaptive_kappa(model,Durable_Importance,participant_id,lr)
        else:
            K = self.helper.config["initial_kappa"]

        for internal_epoch in range(self.helper.config["attacker_retrain_times"]):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = clean_model(inputs)
                loss = self.criterion(output, labels)
                optimizer_clean.zero_grad()
                loss.backward()
                optimizer_clean.step()
                inputs, labels = self.attacker.poison_input(inputs, labels)
                output = model(inputs)
                loss = self.attacker_criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                self.reba_apply_grad_mask(model, Durable_Importance,K)
                optimizer.step()
            loss_L2 = self.model_l2_distance(model, clean_model) * 0.1
            optimizer.zero_grad()
            loss_L2.backward()
            optimizer.step()

    # def reba_get_lr(self, epoch):
    #     if self.helper.config.lr_method == 'exp':
    #         tmp_epoch = epoch
    #         if self.helper.config.is_poison and self.helper.config.load_benign_model:
    #             tmp_epoch += self.helper.config.poison_start_epoch
    #         lr = self.helper.config.lr * (self.helper.config.gamma ** tmp_epoch)
    #     elif self.helper.config.lr_method == 'linear':
    #         if self.helper.config.is_poison or epoch > 1900:
    #             lr = 0.002
    #         else:
    #             lr_init = self.helper.config.lr
    #             target_lr = self.helper.config.target_lr
    #             if epoch <= self.helper.config.epochs / 2.:
    #                 lr = epoch * (target_lr - lr_init) / (self.helper.config.epochs / 2. - 1) + lr_init - (
    #                             target_lr - lr_init) / (self.helper.config.epochs / 2. - 1)
    #             else:
    #                 lr = (epoch - self.helper.config.epochs / 2) * (-target_lr) / (
    #                             self.helper.config.epochs / 2) + target_lr

    #             if lr <= 0.002:
    #                 lr = 0.002
    #     return lr


    def get_lr(self, epoch):
        """
        返回似乎每一轮都有的指定的学习率
        """
        if self.helper.config['dataset'] == 'mnist':
            # lr = 0.002
            lr = 0.01
            return lr
        if self.helper.config["is_poison"] or epoch > 1900:
            # lr = 0.002
            # TODO 学习率修改 C100 -> 0.01/0.002
            lr = 0.002
        else:
            # print("[*] 11111111")
            lr_init = self.helper.config["lr"]
            target_lr = self.helper.config["target_lr"]
            if epoch <= self.helper.config["epochs"] / 2.:
                # print("[*] 22222222222")
                lr = epoch * (target_lr - lr_init) / (self.helper.config["epochs"] / 2. - 1) + lr_init - (
                            target_lr - lr_init) / (self.helper.config["epochs"] / 2. - 1)
            else:
                # print("[*] 333333333")
                lr = (epoch - self.helper.config["epochs"] / 2) * (-target_lr) / (
                            self.helper.config["epochs"] / 2) + target_lr
            if lr <= 0.002:
                # print("[*] 44444444444")
                lr = 0.002
            # if lr <= 0.01:
            #         lr = 0.01
            # else:
            #     print("[*] !!!!!!!!!")
            #     print(f"[*] lr == {lr}")
            #     raise NotImplementedError
        return lr

    def sample_participants(self, epoch):
        """决定在当前的这一轮训练中, 哪些客户端被选中来上传它们的模型更新\n
            random 和 fix(带有固定范围偏好的随机)\n
            TOANSWER random 和 random_updates 啥区别, 采样的过程都是一样的, 但是下面 contain_adversary() 函数内表明二者有区别
        """
        if self.helper.config["sample_method"] in ['random', 'random_updates']:
            sampled_participants = random.sample(
                range(self.helper.config["num_total_participants"]),
                self.helper.config["num_sampled_participants"])
        elif self.helper.config["sample_method"] == 'fix':
            num_total = self.helper.config["num_total_participants"]
            num_sampled = self.helper.config["num_sampled_participants"]
            # 先从0-4中固定选一个
            # TOANSWER 为什么0-4?他们是攻击者?
            # FOCUS 待修改
            # fixed_selected = random.sample(range(5), 1)[0]
            fixed_selected = random.sample(range(self.helper.config["num_adversaries"]), 1)[0]
            remaining_participants = list(set(range(num_total)) - {fixed_selected})
            # 再从剩下的里面选剩余需要的数量
            num_remaining_to_select = num_sampled - 1
            sampled_remaining = random.sample(remaining_participants, num_remaining_to_select)
            # 将固定选的那一个和从剩下里面选的合并起来作为最终采样结果
            sampled_participants = [fixed_selected] + sampled_remaining
        else:
            raise NotImplementedError
        assert len(sampled_participants) == self.helper.config["num_sampled_participants"]
        return sampled_participants
    
    def copy_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])

    def apply_grad_mask(self, model, mask_grad_list):
        """
        应用mask_grad_list原地修改了 model 中各个参数的 .grad 属性(parms.grad = parms.grad * next(mask_grad_list_copy))
        
        迫使模型将只根据过滤后的梯度进行更新 
        """
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)
        
    def reba_apply_grad_mask(self, model, mask_grad_list,k):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, params in model.named_parameters():
            if params.requires_grad and len(params.shape) != 1:
                params.grad = params.grad * k * next(mask_grad_list_copy)        

    def grad_mask_cv(self, model, dataset_clearn, ratio=0.95):
        """Generate a gradient mask based on the given dataset(计算机视觉任务)
        
        Params:\n
            dataset_clearn 是一个DataLoader列表, 包含了由subset_data_chunks_mask决定的各个参与者的数据加载器
        
        Neurotoxin 攻击\n
            攻击者从服务器下载前一轮的全局梯度 g , 并计算 g 中绝对值最大的前 k% 个坐标, 将其存入集合 S = topₖ(g), 然后 攻击者在生成自己的恶意更新时, 会避开 S 中的这些位置, 只在剩余的、良性用户更新不频繁的"冷门"参数上植入后门
            
        通过分析良性数据在模型上产生的梯度, 识别出那些"不活跃"(更新频率低、梯度幅值小)的参数, 并生成一个掩码( mask_grad_list, 结构与模型参数结构一致 ) 攻击者随后只在这些掩码标记的参数上植入后门, 以确保后门不会被良性训练覆盖, 从而实现持久性 
        """
        model.train()
        model.zero_grad()
        ce_loss = torch.nn.CrossEntropyLoss()
        for participant_id in range(len(dataset_clearn)):

            train_data = dataset_clearn[participant_id]
            # train_data == (pos, pos对应的参与者的DataLoader) 的元组列表
            _, data = train_data
            # 来个一轮训练
            for inputs, labels in data:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = ce_loss(output, labels)
                loss.backward(retain_graph=True)
        # 用于存储每一层最终生成的张量掩码
        mask_grad_list = []
        # 二维列表, 用于临时存储所有层展平后的梯度绝对值一维长向量
        grad_list = []
        # 记录每一层梯度的绝对值之和
        grad_abs_sum_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_list.append(parms.grad.abs().view(-1))

                grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                k_layer += 1
        # 将 grad_list 中所有层的一维梯度向量连接成一个巨大的单一向量
        grad_list = torch.cat(grad_list).cuda()
        # topk 默认找最大的k个
        # indices 存储了那些梯度幅值最小的参数在长向量中的位置下标, 这就是我们要攻击的参数(不活跃的)
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        #  mask_flat_all_layer 是合并后的所有层的参数梯度掩码向量, 将选中的"不活跃"参数位置设为 1.0 这意味着掩码中 1 代表允许攻击, 0 代表禁止攻击(避开良性更新活跃区)
        mask_flat_all_layer[indices] = 1.0

        # 下面是一维长向量还原成原本形状的多维张量的过程
        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))
                mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                # 按层切割后将每一层的一维梯度掩码还原对应的参数梯度张量形状
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                count += gradients_length
                # 计算该层有多少比例的参数被选中作为掩码
                percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                percentage_mask_list.append(percentage_mask1)
                # 该层梯度绝对值之和占全模型梯度绝对值之和的权重比例
                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))
                k_layer += 1
        model.zero_grad()
        # 最终得到一个张量列表, 每个张量与模型参数一一对应(形状上)
        return mask_grad_list