from random import random

import numpy as np
from fl_utils.helper import Helper
from fl_utils.cifar100_helper import Cifar100_Helper
import torch
import logging
from copy import deepcopy
from torch import optim
logger = logging.getLogger('logger')


def check_indicator(helper, global_update, indicators, ind_layer):
    accept = []
    feedbacks = []
    num = helper.config["num_adversaries"]
    for adv_id in range(num):
        [I, ind_val] = indicators[adv_id]
        feedbacks.append(global_update[ind_layer]
            [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
    for [I, ind_val] in indicators[num:]:
        feedbacks.append(global_update[ind_layer]
            [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
    # print(f'feedbacks {feedbacks}')
    if helper.config["dataset"] == 'mnist':
        threshold = 1e-4
    else:
        threshold = 1e-5
    for feedback in feedbacks:
        if feedback <= threshold:
            accept.append('r')  # r = rejected
        elif feedback > threshold:
            accept.append('a')  # a = accepted
    return accept


def design_indicator(helper, model, sampled_participants, weight_accumulator_by_client, backdoor_update, benign_update,
            criterion, train_loader, trigger, mask):
    indicator_num = helper.config["num_adversaries"]
    if helper.config["dataset"] == 'cifar10':
        num_candidate = 10
        backdoor_update = abs(backdoor_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
        benign_update = abs(benign_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update + benign_update
        no_layer = 57
        gradient = np.zeros(shape=(256, 256, 3, 3))
        curvature = np.zeros(shape=(256, 256, 3, 3))
        ind_layer = 'layer4.1.conv2.weight'
    elif helper.config["dataset"] == 'cifar100':
        num_candidate = 100
        backdoor_update = abs(backdoor_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
        benign_update = abs(benign_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update + benign_update
        no_layer = 57
        gradient = np.zeros(shape=(256, 256, 3, 3))
        curvature = np.zeros(shape=(256, 256, 3, 3))
        ind_layer = 'layer4.1.conv2.weight'
    else:
        num_candidate = 10
        backdoor_update = abs(backdoor_update['conv2.weight'].cpu().numpy()) .flatten()
        benign_update = abs(benign_update['conv2.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update + benign_update
        no_layer = 2 # conv2.weight
        gradient = np.zeros(shape=(50, 20, 5, 5))
        curvature = np.zeros(shape=(50, 20, 5, 5))
        ind_layer = 'conv2.weight'

    t = trigger.clone()
    m = mask.clone()

    # Get gradient and curvature
    for inputs, labels in (train_loader):
        inputs, labels, be_labels = inputs.cuda(), labels.cuda(), labels.cuda()
        # be_labels = labels
        inputs = t * m + (1 - m) * inputs
        labels[:] = helper.config['target_class']
        # Compute gradient and curvature for normal loss
        outputs = model(inputs)
        loss = criterion(outputs, be_labels)
        grad = torch.autograd.grad(loss.mean(),
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    retain_graph=True,
                                    create_graph=True
                                    )[no_layer]
        grad.requires_grad_()
        grad_sum = torch.sum(grad)
        curv = torch.autograd.grad(grad_sum,
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    retain_graph=True
                                    )[no_layer]
        gradient += grad.detach().cpu().numpy()
        curvature += curv.detach().cpu().numpy()

        # Compute gradient and curvature for backdoor loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        grad = torch.autograd.grad(loss.mean(),
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    create_graph=True,
                                    retain_graph=True
                                    )[no_layer]
        grad.requires_grad_()
        grad_sum = torch.sum(grad)
        curv = torch.autograd.grad(grad_sum,
                                    [x for x in model.parameters() if
                                    x.requires_grad],
                                    retain_graph=True
                                    )[no_layer]
        gradient += grad.detach().cpu().numpy()
        curvature += curv.detach().cpu().numpy()

    update_val = []
    idx_candidate = []
    for i, grad in enumerate(analog_update):
        if len(idx_candidate) < num_candidate * indicator_num:
            update_val.append(grad)
            idx_candidate.append(i)
        elif grad < max(update_val):
            temp = update_val.index(max(update_val))
            update_val[temp] = grad
            idx_candidate[temp] = i

    index = []
    curv_val = []
    curvature = np.abs(curvature.flatten()).tolist()
    for idx in idx_candidate:
        if len(index) < indicator_num:
            curv_val.append(curvature[idx])
            index.append(idx)
        elif curvature[idx] == 0:
            temp = curv_val.index(max(curv_val))  # The index having max curvature
            if analog_update[idx] < analog_update[index[temp]]:
                curv_val[temp] = curvature[idx]
                index[temp] = idx
        elif curvature[idx] < max(curv_val):
            temp = curv_val.index(max(curv_val))
            curv_val[temp] = curvature[idx]
            index[temp] = idx

    if 'cifar' in helper.config["dataset"]:
        temp = []
        for i in range(len(curvature)):
            temp.append(i)
        temp = np.reshape(temp, (256, 256, 3, 3))
        for i in range(len(index)):
            index[i] = np.where(temp == index[i])
            index[i] = [index[i][0][0], index[i][1][0],
                        index[i][2][0], index[i][3][0]]
    else:
        temp = []
        for i in range(len(curvature)):
            temp.append(i)
        temp = np.reshape(temp, (50, 20, 5, 5))
        for i in range(len(index)):
            index[i] = np.where(temp == index[i])
            index[i] = [index[i][0][0], index[i][1][0],
                        index[i][2][0], index[i][3][0]]
    # index_I = []
    # logging.info(f"index : {index}")
    folder_path = helper.config["folder_path"]
    for i in range(indicator_num):
        if i not in sampled_participants:
            file_name = '{0}/saved_updates/update_{1}.pth'.format(folder_path, i)
            client_update = torch.load(file_name)
        else:
            j = sampled_participants.index(i)
            client_update = weight_accumulator_by_client[j]

        # client_update = weight_accumulator_by_client[i]  lipu de foolsgold dedaode
        I = index[i]

        client_update[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e5)

        if client_update[ind_layer][I[0]][I[1]][I[2]][I[3]] == 0:
            if helper.config["dataset"] == 'mnist':
                client_update[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-2)
            else:
                client_update[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-3)
        index[i] = [I, client_update[ind_layer][I[0]][I[1]][I[2]][I[3]].item()]

        if i not in sampled_participants:
            helper.save_update(client_update, i)
    return index


"""
论文 P27 (1)输出层噪声添加 : Deepsight[68]定义全连接层神经元的更新能量为 Ups,
即全局模型与当前模型权重和偏置项的  差之和. 因此我们将噪声视为可优化目标,
使用 Ups 代表恶意模型相对于全局模型的异常  程度,在模型的全连接层添加噪声,降低恶意模型的 Ups,
避免全连接层过度集中的神经元更新. 

在论文 DeepSight 中, UPS 是其核心检测指标之一, 全称为 Unusual Parameter Supplement ,
它衡量的是模型更新(梯度)的"能量"或"强度". DeepSight 认为, 攻击者为了让后门生效, 通常需要对模型的某些神经元(尤其是全连接层/输出层)进行剧烈的修改
计算方式: 对于某个神经元 m , 其 Upsₘ 是该神经元相关联的所有权重(Weights)和偏置(Bias)更新值的绝对值之和
即: Upsₘ = (∑|Δ Weightₘ|) + |Δ Biasₘ| 
防御逻辑: DeepSight 通过检测各个神经元 Ups 值的分布来识别异常. 如果发现某些神经元的 Ups 异常高, 或者分布极度不均匀, 就会判定该模型为恶意模型

输出层噪声添加:
由于攻击者直接注入后门会导致输出层某些神经元的 Ups 飙升, 被 DeepSight 轻易过滤, 因此攻击者可以在全连接层添加经过精心设计的噪声,
通过噪声抵消掉原本过高的 Ups, 同时通过在原本不活跃的神经元上添加噪声, 让 Ups 的方差变小
结果是: 恶意模型在 DeepSight 看来, 其神经元更新分布变得"平淡无奇", 从而绕过过滤
"""

"""
Params:\n
    noise_masks: 本文件 -> add_noise() -> noise_lists[] == 全局模型的副本列表(其指定的层(layer_name, 输出/全连接层)初始化为微小的随机噪声)    
    backdoor_update: 见 attacker.py -> noise() == 训练过后的模型增量
    random_neurons: 见 本文件 -> add_noise() 
    # 从所有类别中排除后门的目标类别, 随机选出 8 个索引
    random_neurons.append(temp[:8])
"""

def compute_lagrange_loss(helper, noise_masks, random_neurons):
    """
    计算噪声掩码的 L2 范数(欧几里得距离), 并将其作为 Loss 惩罚项,(不涉及更新量模型) \n
    具体而言, 是将 多个 noise_masks (基于 global_model 的噪声模型) 的输出层的 8 个神经元的噪声提取并累加到一维向量 sum_var 中并计算返回 L2 范数\n
    TOANSWER 为什么要特别地随机选择 random_neurons 这些类别呢\n
    最后返回范数列表, 一个噪声模型一个\n
    """
    # print(f"[*] klog enter compute_lagrange_loss ")
    # 存储每个噪声掩码计算出的最终 loss 值
    losses = []
    size = 0
    if helper.config["dataset"] == 'mnist':
        layer_name = 'fc2'
    elif helper.config['dataset'] == 'tiny-imagenet-200':
        layer_name = 'fc'
    else:
        layer_name = 'linear'
    # 计算输出层参数(展平后)的总大小
    for name, layer in noise_masks[0].named_parameters():
        # print(f"[*] klog name noise_masks[0].named_parameters() {name}")
        # 模型层级
        """
        [*] klog name noise_masks[0].named_parameters() conv1.weight
        [*] klog name noise_masks[0].named_parameters() bn1.weight
        [*] klog name noise_masks[0].named_parameters() bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer1.0.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer1.0.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer1.0.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer1.0.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer1.0.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer1.0.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer1.1.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer1.1.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer1.1.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer1.1.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer1.1.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer1.1.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer2.0.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer2.0.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer2.0.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer2.0.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer2.0.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer2.0.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer2.0.shortcut.0.weight
        [*] klog name noise_masks[0].named_parameters() layer2.0.shortcut.1.weight
        [*] klog name noise_masks[0].named_parameters() layer2.0.shortcut.1.bias
        [*] klog name noise_masks[0].named_parameters() layer2.1.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer2.1.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer2.1.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer2.1.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer2.1.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer2.1.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer3.0.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer3.0.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer3.0.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer3.0.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer3.0.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer3.0.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer3.0.shortcut.0.weight
        [*] klog name noise_masks[0].named_parameters() layer3.0.shortcut.1.weight
        [*] klog name noise_masks[0].named_parameters() layer3.0.shortcut.1.bias
        [*] klog name noise_masks[0].named_parameters() layer3.1.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer3.1.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer3.1.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer3.1.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer3.1.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer3.1.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer4.0.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer4.0.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer4.0.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer4.0.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer4.0.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer4.0.bn2.bias
        [*] klog name noise_masks[0].named_parameters() layer4.0.shortcut.0.weight
        [*] klog name noise_masks[0].named_parameters() layer4.0.shortcut.1.weight
        [*] klog name noise_masks[0].named_parameters() layer4.0.shortcut.1.bias
        [*] klog name noise_masks[0].named_parameters() layer4.1.conv1.weight
        [*] klog name noise_masks[0].named_parameters() layer4.1.bn1.weight
        [*] klog name noise_masks[0].named_parameters() layer4.1.bn1.bias
        [*] klog name noise_masks[0].named_parameters() layer4.1.conv2.weight
        [*] klog name noise_masks[0].named_parameters() layer4.1.bn2.weight
        [*] klog name noise_masks[0].named_parameters() layer4.1.bn2.bias
        [*] klog name noise_masks[0].named_parameters() linear.weight
        [*] klog name noise_masks[0].named_parameters() linear.bias
        """
        if layer_name in name:
            size += layer.view(-1).shape[0]
    # 初始化噪声累加张量( 0填充 ) => 线性排列的所有输出层参数
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    # print(f"[*] klog sum_var.shape {sum_var.shape} size {size} ")
    # [*] klog sum_var.shape torch.Size([25700]) size 25700 (256 * 100 + 100)
    # sum_var = torch.FloatTensor(size).fill_(0)
    # 在每个神经元对应的一维位置上 累加噪声模型的输出层的 random_neurons 所指定的8个神经元的 参数(包括 weight 和 bias)到sum_var
    # print(f"[*] klog len(noise_masks) {len(noise_masks)} ")
    # [*] klog len(noise_masks) 2 
    for i in range(len(noise_masks)):
        size = 0
        for name, layer in noise_masks[i].named_parameters():
            if layer_name in name:
                # print(f"[*] klog layer name {name} layer.shape {layer.shape} ")
                """
                [*] klog layer name linear.weight layer.shape torch.Size([100, 256]) 
                [*] klog layer name linear.bias layer.shape torch.Size([100])                 
                """
                # 讨论输出层的所有行(所有类别)
                for j in range(layer.shape[0]):
                    if j in random_neurons:
                        sum_var[size:size + layer[j].view(-1).shape[0]] += \
                            layer[j].view(-1)
                    # print(f"[*] layer[j] => \n {layer[j]} ")
                    # print(f"[*]  klog layer[j].view(-1).shape[0]] => \n {layer[j].view(-1).shape[0]} ")
                    """
                    [*]  klog layer[j].view(-1).shape[0]] => weight
                    256                     
                    [*]  klog layer[j].view(-1).shape[0]] => bias
                    1                     
                    """
                    """
                    layer[j] 应该不需要展平
                    [*] layer[j] => 
                    tensor([ 2.5937e-03, -6.1338e-03, -3.1274e-03,  4.5584e-03, -2.8434e-03,
                            ......
                            -4.7188e-03,  9.9371e-03,  4.3641e-03,  3.9803e-03,  5.1100e-04,
                            4.4164e-04], device='cuda:0', grad_fn=<SelectBackward0>)                     
                    """
                    size += layer[j].view(-1).shape[0]
    # 计算 L2 惩罚项(拉格朗日约束)
    if helper.config["dataset"] == 'mnist':
        # 计算向量的 L2 范数
        loss = 1e-1 * torch.norm(sum_var+0.00000001, p=2)
    else:
        loss = 1e-2 * torch.norm(sum_var + 0.00000001, p=2)
    
    # print(f"[***] klog loss => \n {loss} ")
    """
    [***] klog loss => 
    0.0032216692343354225     
    """
    
    for i in range(len(noise_masks)):
        losses.append(loss)
    return losses


def compute_noise_ups_loss(helper, backdoor_update, noise_masks, random_neurons):
    """
    如果攻击者只修改了目标类别的神经元, Vs 序列会呈现"一个极高值 + 多个低值", 方差极大. 通过最小化方差, 优化器会自动调整噪声, 使得原本高的变低, 原本低的(通过噪声)变高, 最终让所有神经元的能量看上去差不多\n
    该函数计算并返回添加噪声后的模型增量的输出层的 Ups 值的方差的列表, 每个噪声模型都有自己对应的方差值
   """
    
    # 用于存储每个噪声掩码对应的 Loss 值(输出层 Ups 值的方差)
    losses = []
    # neurons 是指 根据数据集确定输出层的神经元数量(即分类类别数)
    if helper.config["dataset"] == 'cifar100':
        neurons = 100
    elif helper.config['dataset'] == 'tiny-imagenet-200':
        neurons = 200
    else:
        neurons = 10
    # 遍历所有的噪声模型, 计算每个噪声模型对应的 Ups 方差
    for i in range(len(noise_masks)):
        # up_value 列表, 存储当前模型中输出层每个神经元的 Ups 值, 即整个输出层的 Ups 值(能量值)
        Vs = []
        # 遍历一个模型输出层的每一个神经元, 求出输出层总的 Ups 值
        for m in range(neurons):
            if helper.config["dataset"] == 'mnist':
                # print(f"[*] klog backdoor_update['fc2.weight'] : {backdoor_update['fc2.weight']} ")
                """
                backdoor_update['fc2.weight'][m]: 客户端增量模型的输出层的第 m 个神经元的参数 + 
                
                """
                up_value = (torch.abs(backdoor_update['fc2.weight'][m] + \
                                      noise_masks[i].fc2.weight[m]).sum() \
                            + torch.abs(backdoor_update['fc2.bias'][m] + \
                                        noise_masks[i].fc2.bias[m]))
            elif helper.config['dataset'] == 'tiny-imagenet-200':
                up_value = (torch.abs(backdoor_update['fc.weight'][m] + \
                                      noise_masks[i].fc.weight[m]).sum() \
                            + torch.abs(backdoor_update['fc.bias'][m] + \
                                        noise_masks[i].fc.bias[m]))
            else:
                """
                [*] klog backdoor_update['linear.weight'] : tensor([[ 1.9261e-04, -3.7852e-04,  8.6306e-05,  ..., -3.4180e-05,
                        5.9552e-05,  4.2762e-04],
                        [-6.3301e-04, -6.7492e-04, -3.2085e-04,  ..., -1.3131e-04,
                        -6.3534e-04, -4.7471e-04],
                        [ 2.2708e-02,  4.6425e-02,  1.6706e-02,  ...,  2.1995e-02,
                        3.1559e-02,  2.1249e-02],
                        ...,
                        [ 9.1010e-04, -1.0256e-04,  2.2677e-04,  ...,  1.0190e-03,
                        -7.9691e-05, -1.9729e-05],
                        [-9.3263e-05, -1.5459e-03, -2.8735e-04,  ..., -7.4308e-05,
                        -2.6092e-05,  1.5240e-03],
                        [-9.4769e-04, -1.8071e-04, -3.7317e-04,  ...,  2.2882e-04,
                        -1.4129e-03, -5.4972e-04]], device='cuda:0')    
                    [out_features, in_features]
                    out_features (行数): 对应分类任务的类别总数
                    in_features (列数): 对应上一层输出的特征维度
                    长度为 in_features 的一维向量, 代表了所有输入特征对第 m 个分类神经元的贡献权重
                    基础知识需要温习, 需要不断深化基础知识的记忆, 以便于抽象与理解代码   
                    实际上, 窃以为阅读代码时需要在脑中模拟某种相关结构的组织与运行以便于理解与记忆, 因为窃还以为理解是个从具象到抽象的过程, 同时编写代码是个从抽象到具象的过程  
                    窃又以为, 神经网络的本质是抽象的数学运算, 但需要被赋予具象的意义
                """
                # print(f"[*] klog backdoor_update['linear.weight'] : {backdoor_update['linear.weight']} ")
                # print(f"[**] klog backdoor_update['linear.weight'].shape {backdoor_update['linear.weight'].shape} ")
                # [**] klog backdoor_update['linear.weight'].shape torch.Size([100, 256]) 
                # up_value 即为 添加噪声后的模型增量的输出层的某个神经元的 Upsₘ 值
                # 即: Upsₘ = (∑|Δ Weightₘ|) + |Δ Biasₘ| 
                # TOANSWER 只可以加 8 个维度(random_neurons)
                up_value = (torch.abs(backdoor_update['linear.weight'][m] + \
                                      noise_masks[i].linear.weight[m]).sum() \
                            + torch.abs(backdoor_update['linear.bias'][m] + \
                                        noise_masks[i].linear.bias[m]))
            Vs.append(up_value.item())
        ups_tensor = torch.tensor(Vs)
        # 输出层能量值的方差(不是对每个神经元的 Upsₘ 加权求和什么的, 而是计算方差)
        variance = ups_tensor.var()
        # print(f"[*] klog variance {variance} ")
        # [*] klog variance 0.3741230070590973 
        UPs_loss = variance
        noise_masks[i].requires_grad_(True)
        UPs_loss.requires_grad_(True)
        losses.append(UPs_loss)
    return losses


def compute_ups_loss(helper,backdoor_update,noise_masks, random_neurons):
    """
    计算并返回每个噪声模型对应的 random_neurons 指定的各神经元的 UPs 倒数和(分子1e-1)
    """
    """
    与之前的 compute_noise_ups_loss(优化方差/平滑度)不同, 这个函数专门针对攻击者选定的 8个关键神经元(random_neurons). 
    它的核心逻辑是: 如果这些关键神经元的更新强度(UPS)太小, 就通过 Loss 强迫它们变大
    该函数是攻击者为了平衡"隐蔽性"与"攻击力"而设计的.  
    在 compute_noise_ups_loss 中, 攻击者试图减小方差使模型看起来平滑；但如果平滑过头, 后门神经元可能就"哑火"了. 
    因此, compute_ups_loss 通过倒数和的形式, 强迫选定的 8 (这里代码中的len(random_neurons) == 8) 个关键神经元维持一定的更新强度. 
    它确保了噪声虽然掩盖了特征, 但没有抹除后门信号
    """
    # 用于存储每个噪声模型计算出的损失
    losses = []
    for i in range(len(noise_masks)):
        UPs = []
        for j in random_neurons:
            if 'cifar' in helper.config["dataset"]:
                # 计算第 j 个关键神经元在"投毒梯度 + 噪声"后的总更新强度 UPsⱼ
                UPs.append(torch.abs(backdoor_update['linear.weight'][j] + \
                                     noise_masks[i].linear.weight[j]).sum() \
                           + torch.abs(backdoor_update['linear.bias'][j] + \
                                       noise_masks[i].linear.bias[j]))
            elif helper.config['dataset'] == 'tiny-imagenet-200':
                UPs.append(torch.abs(backdoor_update['fc.weight'][j] + \
                                     noise_masks[i].fc.weight[j]).sum() \
                           + torch.abs(backdoor_update['fc.bias'][j] + \
                                       noise_masks[i].fc.bias[j]))
            else:
                UPs.append(torch.abs(backdoor_update['fc2.weight'][j] + \
                                     noise_masks[i].fc2.weight[j]).sum() \
                           + torch.abs(backdoor_update['fc2.bias'][j] + \
                                       noise_masks[i].fc2.bias[j]))
        UPs_loss = 0
        # 通过梯度下降减小此 Loss, 会迫使优化器增大这些关键神经元的 UPS. 这样可以防止噪声添加过猛, 导致原本的后门信号被彻底抵消
        for j in range(len(UPs)):
            UPs_loss += 1e-1 / UPs[j]  # (UPs[j] * params.fl_num_neurons)
        noise_masks[i].requires_grad_(True)
        UPs_loss.requires_grad_(True)
        losses.append(UPs_loss)
    return losses

# def compute_noise_loss(helper:Cifar100_Helper, backdoor_update, noise_masks, alpha, random_neurons, lagrange_mul):

def compute_noise_loss(helper, backdoor_update, noise_masks, alpha, random_neurons, lagrange_mul):
    """
    losses = compute_noise_loss(helper, backdoor_update, noise_lists, alpha, random_neurons[0], 1)\n
    在 DeepSight 论文的背景下, 攻击者需要解决一个多目标优化问题: 生成的噪声既要能抹平异常特征(躲避检测), 又要保持攻击强度(后门有效), 还要足够微小(不破坏模型原本的分类精度). 这个函数通过加权组合三个子损失函数, 构建了最终的优化目标\n
    返回一个包含总 Loss 的列表(总共有多个噪声掩码/模型, 即 noise_masks / noise_lists). 每个 Loss 都代表一个噪声掩码的多目标优化得分. 数值越小, 代表该噪声越能完美平衡上述三个目标
    """
    # 初始化最终总损失列表
    loss = []
    # Compute UPs loss
    # 计算方差损失. 它让所有神经元的更新能量(UPS)趋于平均
    ups_loss = compute_noise_ups_loss(helper, backdoor_update, noise_masks, random_neurons)
    # 计算强度保障损失. 它强迫选定的 8 个神经元保持足够的更新能量
    ups = compute_ups_loss(helper, backdoor_update, noise_masks, random_neurons)
    # 将"隐蔽性优化(均分能量)"和"有效性优化(保障能量)"进行等权合并(取平均)
    for i in range(len(ups_loss)):
        loss.append((ups_loss[i]+ups[i])/2)
    # Compute lagrange constrain
    # 计算约束损失. 通过 L2 范数限制噪声的大小
    lagrange_loss = compute_lagrange_loss(helper, noise_masks, random_neurons)
    # Attacker 中定义为 alpha = 0.5, 对于每一个噪声掩码将 lagrange constrain 加权到原损失值中
    for i in range(len(ups_loss)):
        loss[i] = alpha * loss[i] + (1-alpha) * lagrange_loss[i]
    return loss


def add_noise(helper, sampled_participants, weight_accumulator_by_client, backdoor_update, layer_name, global_model, alpha):
    """
    add_noise(self.helper, sampled_participants, weight_accumulator_by_client, backdoor_update, layer_name,\n
                      noise_model, self.alpha)\n
    挑选了 8 个无关的神经元作为陪衬, 训练了一组特殊的噪声, 目的是让这 8 个神经元的能量看起来和目标神经元一样高(平滑方差), 同时又让所有噪声在所有攻击者之间互相抵消(中心化)\n
    并将优化好的噪声植入到客户端更新量模型的 random_neurons 指定的输出层的神经元中, 最后 save_update 保存修改后的更新量模型
    """
    noise_masks = []
    random_neurons = []
    if helper.config["dataset"] == 'cifar100':
        temp = list(range(100))
    elif helper.config['dataset'] == 'tiny-imagenet-200':
        temp = list(range(200))
    else:
        temp = list(range(10))  
    temp.remove(helper.config["target_class"])
    np.random.shuffle(temp)
    """
    TOANSWER 可以这么理解吗: 攻击后的模型在目标类上的神经元的参数必定会有异常, 
    攻击者不想让"目标类"看起来太突兀, 所以选了 8 个"陪跑"的神经元, 准备在它们身上也加点噪声, 把水搅浑
    """
    # 从所有类别中排除后门的目标类别, 随机选出 8 个索引
    random_neurons.append(temp[:8])
    # 全局模型的副本列表(其指定的层(layer_name)初始化为微小的随机噪声), 这创建了后面需要的噪声掩码
    # TOANSWER 每个噪声模型应该是不一样的吧?
    noise_lists = []

    # num 是噪声掩码的数量
    num = 2
    for i in range(num):
        noised_model = deepcopy(global_model)
        for name, data in noised_model.state_dict().items():
            if layer_name in name:
                noised_layer = torch.FloatTensor(data.shape).fill_(0)
                noised_layer = noised_layer.to(helper.device)
                # noised_layer.normal_(mean=0, std=0.05)
                # normal_ 用于原地生成高斯噪声
                noised_layer.normal_(mean=0, std=0.005)
                # data 被替换成了纯噪声
                data.add_(noised_layer - data)
        noise_lists.append(noised_model)

    # Centralize the noise mask 噪声中心化
    #  TOANSWER 让多个噪声模型的参数之和趋近于零或某个基准? 没均值啊
    avg_params = deepcopy(backdoor_update)
    # avg_params 借用了 backdoor_update 的结构(当然整个项目里所有模型的结构都一样), 其输出层参数归零
    # FOCUS 已修改
    for name, data in avg_params.items():
        # TOANSWER 这里的 name 是哪来的?
        # print(f"[*] klog name {name}")
        """
        全是:
        [*] klog name linear.bias
        """
        if layer_name in name:
            data.fill_(0)
    """
    应该是这样吧?
    A: a
    B: b
    avg = (a + b) / 2
    A_new = a - avg = (a - b) / 2
    B_new = b - avg = (b - a) / 2
    A_new + B_new == 0
    """
    # 噪声求和?求均值?
    for i in range(num):
        for name, data in noise_lists[i].state_dict().items():
            # avg_params 的输出层参数加上上面得到的噪声
            if layer_name in name:
                # TOANSWER 这里确定不是 / num 吗
                avg_params[name].add_(data / 1)
    # 中心化所有噪声模型的噪声值
    for i in range(num):
        for name, data in noise_lists[i].state_dict().items():
            # 又让每个噪声模型含噪声的输出层 - 噪声均值?
            if layer_name in name:
                data.add_(- avg_params[name])
    # 定义噪声掩码的优化器
    optimizer_lists = []
    for i in range(num):
        optimizer_lists.append(optim.SGD(
            noise_lists[i].parameters(),
            lr=0.1,
            weight_decay=helper.config["decay"],
            momentum=helper.config["momentum"]))
    # 30 轮优化噪声掩码
    for _ in range(30):
        for i in range(num):
            noise_lists[i].zero_grad()
        # 计算三大 Loss (方差 + 强度 + 约束), UPs 的相关内容在噪声计算中
        losses = compute_noise_loss(helper, backdoor_update, noise_lists, alpha, random_neurons[0], 1)
        # print(f" [*] klog losses {losses}, type(losses) {type(losses)}")
        """
        [*] klog losses [tensor(2.3080, device='cuda:0', grad_fn=<AddBackward0>), tensor(2.2242, device='cuda:0', grad_fn=<AddBackward0>)], type(losses) <class 'list'>
        """
        for i in range(num):
            losses[i].backward(retain_graph=True)
            optimizer_lists[i].step()
    # 得到最终的噪声掩码模型
    for temp in noise_lists:
        noise_masks.append(temp)

    for i in range(helper.config["num_adversaries"]):
        if i not in sampled_participants:
            continue
        """
        TOANSWER 见 train_once() 函数:
        sampled_participants: [0, 71, 13, 69, 24, 55, 66, 76, 26, 64] / [4, 29, 12, 7, 91, 88, 49, 52, 30, 98]
        for participant_id in sampled_participants:
            ......
            weight_accumulator_by_client.append(single_wa) 
        这段代码的意思是选中攻击者吧? weight_accumulator_by_client 是列表又不是字典, 直接 weight_accumulator_by_client[i] 有问题吧
        以 [4, 29, 12, 7, 91, 88, 49, 52, 30, 98] 为例, 最后会选择 91 吧, 这里如果要选择攻击者应该是 weight_accumulator_by_client[0] ?      
        """
        i = sampled_participants.index(i)
        client_weight = weight_accumulator_by_client[i]
        # 随机选一个优化好的噪声掩码
        j = np.random.randint(num)
        for name, data in noise_masks[j].state_dict().items():
            if layer_name in name:
                # 创建一个全 0 的掩码容器
                sum_var = torch.cuda.FloatTensor(data.shape).fill_(0)
                # 又来个 j, 这码风绝了 
                for j in range(sum_var.shape[0]):
                    if j in random_neurons[0]:
                        sum_var[j] = data[j]
                # 给 client_weight 的输出层的 random_neurons 指定的神经元添加噪声掩码
                client_weight[name].add_(sum_var)
        # 保存添加噪声的更新量模型
        helper.save_update(client_weight, i)
    return True