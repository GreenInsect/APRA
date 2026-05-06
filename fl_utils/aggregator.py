import math
import sys

from torch import nn

from model.mobilenet import Mobilenet
from model.pytorch_resnet import pt_resnet18
from model.resnet import ResNet18, ResNet34

sys.path.append("../")

import torch
from torch.utils.data import DataLoader, TensorDataset
# import hdbscan
from sklearn.cluster import KMeans
from defenses.fldetector import gap_statistics
from collections import defaultdict, OrderedDict, Counter
from sklearn.decomposition import PCA
import numpy as np
import copy
import os
from sklearn.cluster import DBSCAN
import logging
import sklearn.metrics.pairwise as smp
from model.simple import SimpleMnist
from fl_utils.apra import APRAAggregator
logger = logging.getLogger('logger')



class Aggregator:
    def __init__(self, helper, ood_set):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []
        # 水印数据, 也是 OOD 数据集
        self.wm_data = ood_set # Out-of-Distribution Dataset (分布外数据集), 指的是与模型正常训练任务(In-Distribution, ID)完全无关的数据
        # 添加水印的轮数
        self.watermarking_rounds = [round for round in range(1,100,10)]
        # 水印之后的bn层
        self.after_wm_injection_bn_stats_dict = dict()
        self.wm_mu = self.helper.config["watermarking_mu"] # $\mu$ ? 一个正则化超参数, 限制了模型修改的幅度, 防止植入水印导致模型在正常任务上彻底失效
        if self.helper.config["dataset"] == 'mnist':
            check_model = SimpleMnist()
        elif self.helper.config['dataset'] == 'tiny-imagenet-200':
            check_model = pt_resnet18(num_classes=self.helper.num_classes)
        else:
            check_model = ResNet18(num_classes=self.helper.num_classes)
            # check_model = ResNet34(num_classes=self.helper.num_classes)
            # check_model = Mobilenet()

        self.check_model = check_model.cuda()
        self.target_label = -1


    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants, epoch):
        """
        TODO 没看
        """
        if self.helper.config['agg_method'] == 'avg':
            return self.average_models(global_model, weight_accumulator,  weight_accumulator_by_client, sampled_participants)
        elif self.helper.config['agg_method'] == 'apra':
            apraAggregator = APRAAggregator(self.helper)
            return apraAggregator.aggregate(global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants, epoch)
        elif self.helper.config['agg_method'] == 'clip':
            self.clip_updates(weight_accumulator, weight_accumulator_by_client, sampled_participants)
            return self.average_shrink_models(global_model, weight_accumulator, weight_accumulator_by_client)
        elif self.helper.config["agg_method"] == 'deepsight':
            chosen = self.deepsight_aggregate_global_model(global_model,client_models,sampled_participants)
            print(f"init_ids:{sampled_participants}    choosen_ids:{chosen}")
            return self.average_chosen_models(weight_accumulator_by_client, chosen, global_model, sampled_participants)
        elif self.helper.config["agg_method"] == 'foolsgold':
            wv = self.foolsgold_aggr(weight_accumulator_by_client, weight_accumulator, sampled_participants)
            return self.average_fool_models(weight_accumulator_by_client, global_model, sampled_participants, wv)
        elif self.helper.config["agg_method"] == 'rflbat':
            chosen_ids = self.rflbat_aggr(weight_accumulator_by_client, weight_accumulator, sampled_participants)
            return self.average_chosen_models(weight_accumulator_by_client, chosen_ids, global_model, sampled_participants)
        elif self.helper.config["agg_method"] == 'second':
            chosen_ids = self.second_filter(sampled_participants)
            return self.average_chosen_models(weight_accumulator_by_client, chosen_ids, global_model, sampled_participants)
        elif self.helper.config["agg_method"] == 'indicator':
            chosen_ids = self.ood_indicator(sampled_participants, client_models)
            # chosen_ids = self.rflbat_aggr(weight_accumulator_by_client, weight_accumulator, sampled_participants)
            return self.average_chosen_models(weight_accumulator_by_client, chosen_ids, global_model, sampled_participants)
        else:
            raise NotImplementedError

    def average_shrink_models(self, global_model, weight_accumulator,  weight_accumulator_by_client):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config['num_sampled_participants']) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
            # data.add_(update_per_layer)

        return True

    def average_models(self, global_model, weight_accumulator,  weight_accumulator_by_client, sampled_participants):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)
        chosen_id = sampled_participants
        print(f"avg_ids:{sampled_participants}    choosen_ids:{chosen_id}")
        for i in chosen_id:
            index = sampled_participants.index(i)
            client_weight = weight_accumulator_by_client[index]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                averaged_weights[name] += client_weight[name]
        lr = 1
        # lr = self.helper.config["lr"]
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * \
                               (1 / len(chosen_id)) * lr
            update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True

    def average_deep_models(self, global_model, weight_accumulator, choose_num):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/choose_num) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
        return True

    def average_chosen_models(self, weight_accumulator_by_client, chosen_id, global_model, sampled_participants):

        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)
        if len(chosen_id) == 0:
            return
        for i in chosen_id:
            index = sampled_participants.index(i)
            client_weight = weight_accumulator_by_client[index]
            # client_weight = weight_accumulator_by_client[i]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                averaged_weights[name] += client_weight[name]
        lr = 1
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * \
                               (1/len(chosen_id)) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
        return True


    def average_fool_models(self, weight_accumulator_by_client, global_model, sampled_participants, wv):
        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)
        for i in sampled_participants:
            index = sampled_participants.index(i)
            w = int(wv[index])
            client_weight = weight_accumulator_by_client[index]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                averaged_weights[name] += client_weight[name] * w

        lr = 0.5
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * \
                               (1/len(sampled_participants)) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
        return True

    def clip_updates(self, weight_accumulator, weight_accumulator_by_client, sampled_participants):
        """
        限制每个客户端对全局模型更新的“贡献强度”，防止某个客户端（尤其是攻击者）通过发送极大的权重更新来恶意偏移全局模型\n
        weight_accumulator 构造\n
        for name, data in self.helper.global_model.state_dict().items():\n
            .....\n
            weight_accumulator[name] = torch.zeros_like(data) \n       
        """
        for key in weight_accumulator:
            # num_batches_tracked 是 Batch Normalization 层的一个统计量，它记录训练了多少个 batch
            # 这个值是一个整数，不需要进行梯度裁剪或 L2 范数计算，所以代码通过 if 语句将其跳过
            if 'num_batches_tracked' not in key:
                update = weight_accumulator[key]
                # 计算更新量 update 的 L2 范数（即向量的长度）
                l2_update = torch.norm(update, p=2) 

                print(f"[*] klog: Layer {key} L2 Norm = {l2_update.item():.4f}") 
                               
                # FOCUS clip_factor: 1
                update.div_(max(1, l2_update/self.helper.config['clip_factor']))
        return

    def save_history(self, index, weight_accumulator, userID=0):
        folderpath = '{0}/foolsgold'.format(self.helper.config["folder_path"])
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        history_name = '{0}/history_{1}.pth'.format(folderpath, userID)
        update_name = '{0}/saved_updates/update_{1}.pth'.format(self.helper.config["folder_path"], userID)
        model = torch.load(update_name)
        if os.path.exists(history_name):
            loaded_params = torch.load(history_name)
            history = dict()
            for name, data in loaded_params.items():
                history[name] = data + model[name]
            torch.save(history, history_name)
        else:
            torch.save(model, history_name)

    def foolsgold_aggr(self, weight_accumulator_by_client, weight_accumulator, sampled_participants):
        num = self.helper.config["num_sampled_participants"]
        for i in sampled_participants:
            index = sampled_participants.index(i)
            self.save_history(index, weight_accumulator_by_client, userID=i)
        if self.helper.config["dataset"] == 'cifar10' or self.helper.config["dataset"] == 'cifar100':
            layer_name = 'linear'
        elif self.helper.config['dataset'] == 'tiny-imagenet-200':
            layer_name = 'fc'
        else:
            layer_name = 'fc2'
        epsilon = 1e-5
        folderpath = '{0}/foolsgold'.format(self.helper.config["folder_path"])
        # Load params
        his = []
        for i in sampled_participants:
        # for i in range(num):
            history_name = '{0}/history_{1}.pth'.format(folderpath, i)
            his_i_params = torch.load(history_name)
            for name, data in his_i_params.items():
                if layer_name in name:
                    # print((data.cpu().numpy()).flatten())
                    his = np.append(his, (data.cpu().numpy()).flatten())

        his = np.reshape(his, (num, -1))
        # print(f"his shape :{his.shape}")
        # print(his)
        print("FoolsGold: Finish loading history updates")
        cs = smp.cosine_similarity(his) - np.eye(num)
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        # Pardoning
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # Federated SGD iteration
        print(f"FoolsGold: Accumulation with lr {wv}")
        return wv

    def rflbat_aggr(self, weight_accumulator_by_client, weight_accumulator, sampled_participants):
        print(f"into rflbat")
        eps1 = 10
        eps2 = 6
        dataAll = []
        folder_path = self.helper.config["folder_path"]
        for i in sampled_participants:
            file_name = '{0}/saved_updates/update_{1}.pth'.format(folder_path, i)
            index = sampled_participants.index(i)
            dataList = []
            if os.path.exists(file_name):
                # loaded_params = torch.load(file_name)
                loaded_params = weight_accumulator_by_client[index]
                for name, data in loaded_params.items():
                    if 'mnist' in self.helper.config["dataset"] or 'linear' in name or 'layer4.1.conv' in name or 'fc' in name:
                        dataList.extend(((data.cpu().numpy()).flatten()).tolist())
                dataAll.append(dataList)
        pca = PCA(n_components=2)  # instantiate
        pca = pca.fit(dataAll)
        X_dr = pca.transform(dataAll)
        # Compute sum eu distance
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i == j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i] - X_dr[j])
            eu_list.append(eu_sum)
        accept = []
        x1 = []
        for i in range(len(eu_list)):
            if eu_list[i] < eps1 * np.median(eu_list):
                accept.append(i)
                x1 = np.append(x1, X_dr[i])
            else:
                print("RFLBAT: discard update {0}".format(i))

        print("RFLBAT: the first clients accepted are {0}".format(accept))

        x1 = np.reshape(x1, (-1, X_dr.shape[1]))
        num_clusters = gap_statistics(x1, num_sampling=5, K_max=10, n=len(x1))
        print("RFLBAT: the number of clusters is {0}".format(num_clusters))
        k_means = KMeans(n_clusters=num_clusters, init='k-means++').fit(x1)
        predicts = k_means.labels_

        # select the most suitable cluster
        v_med = []
        for i in range(num_clusters):
            temp = []
            for j in range(len(predicts)):
                if predicts[j] == i:
                    temp.append(dataAll[accept[j]])
            if len(temp) <= 1:
                v_med.append(1)
                continue
            v_med.append(np.median(np.average(smp.cosine_similarity(temp), axis=1)))
        temp = []
        for i in range(len(accept)):
            if predicts[i] == v_med.index(min(v_med)):
                temp.append(accept[i])
        accept = temp
        print("RFLBAT: the second clients accepted are {0}".format(accept))
        # compute eu list again to exclude outliers
        temp = []
        for i in accept:
            temp.append(X_dr[i])
        X_dr = temp
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i == j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i] - X_dr[j])
            eu_list.append(eu_sum)
        temp = []
        for i in range(len(eu_list)):
            if eu_list[i] < eps2 * np.median(eu_list):
                temp.append(accept[i])
            else:
                print("RFLBAT: discard update {0}".format(i))
        accept = temp
        chosen_id = []
        for i in sampled_participants:
            index = sampled_participants.index(i)
            if index in accept:
                chosen_id.append(i)
        # aggregate
        print("RFLBAT: the final clients accepted are {0}".format(chosen_id))
        return chosen_id

    def deepsight_aggregate_global_model(self, global_model, clients, chosen_ids):
        def ensemble_cluster(neups, ddifs, biases):

            def sanitize(data, name):
                if not np.isfinite(data).all():
                    print(f"[*] klog Warning: {name} contains NaN/Inf. Sanitizing...")
                    # 替换为 0，防止报错
                    return np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                return data           
             
            biases = np.array([bias.cpu().numpy() for bias in biases])
            N = len(neups)
            # use bias to conduct DBSCAM
            # FOCUS 修改于 1/30 由于 训练 mnist_140_Jan.30_11.27.34_deepsight_True_a3fl 发生了错误
            """
            Traceback (most recent call last):
            File "clean_mnist_a3fl.py", line 81, in <module>
                main(helper)
            File "clean_mnist_a3fl.py", line 55, in main
                fler.train()
            File "../fl_utils/fler.py", line 193, in train
                self.aggregator.agg(self.helper.global_model, weight_accumulator, weight_accumulator_by_client, self.helper.client_models, sampled_participants, epoch)
            File "../fl_utils/aggregator.py", line 64, in agg
                chosen = self.deepsight_aggregate_global_model(global_model,client_models,sampled_participants)
            File "../fl_utils/aggregator.py", line 430, in deepsight_aggregate_global_model
                clusters = ensemble_cluster(neups, client_ddifs, biases)
            File "../fl_utils/aggregator.py", line 366, in ensemble_cluster
                cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases).labels_
            File "/usr/local/lib/python3.7/site-packages/sklearn/cluster/_dbscan.py", line 346, in fit
                X = self._validate_data(X, accept_sparse="csr")
            File "/usr/local/lib/python3.7/site-packages/sklearn/base.py", line 566, in _validate_data
                X = check_array(X, **check_params)
            File "/usr/local/lib/python3.7/site-packages/sklearn/utils/validation.py", line 800, in check_array
                _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
            File "/usr/local/lib/python3.7/site-packages/sklearn/utils/validation.py", line 116, in _assert_all_finite
                type_err, msg_dtype if msg_dtype is not None else X.dtype
            ValueError: Input contains NaN, infinity or a value too large for dtype('float32').            
            """
            biases = sanitize(biases, "biases")
            cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases).labels_
            print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neups = sanitize(neups, "neups")
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            print("neup_cluster:{}".format(neup_labels))
            ddifs = sanitize(ddifs, "ddifs")
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            print("ddif_cluster:{}".format(ddif_labels))

            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j])) / 3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]

            ensembled_labels = DBSCAN(min_samples=3, metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels

        global_weight = list(global_model.state_dict().values())[-2]
        global_bias = list(global_model.state_dict().values())[-1]

        biases = [(list(clients[i].state_dict().values())[-1] - global_bias) for i in
                  chosen_ids]  # 与全局模型的偏置差异
        weights = [list(clients[i].state_dict().values())[-2] for i in chosen_ids]

        n_client = len(chosen_ids)
        cosine_similarity_dists = np.array((n_client, n_client))
        neups = list()
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in chosen_ids:
            id = chosen_ids.index(i)
            C_nn = torch.sum(weights[id] - global_weight, dim=[1]) + biases[id] - global_bias
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2

            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)
        # normalize
        neups = np.array([(neup / sC_nn2).cpu().numpy() for neup in neups])
        print("n_exceeds:{}".format(n_exceeds))

        rand_input = torch.randn((256, 3, 32, 32)).to(self.helper.device)
        if self.helper.config["dataset"] == 'mnist':
            rand_input = torch.randn((20, 1, 28, 28)).to(self.helper.device)
        elif self.helper.config["dataset"] == 'tiny-imagenet-200':
            rand_input = torch.randn((64, 3, 224, 224)).to(self.helper.device)

        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        client_ddifs = [torch.mean(torch.softmax(clients[i](rand_input), dim=1), dim=0) / global_ddif
                        for i in chosen_ids]
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])

        # use n_exceed to label
        classification_boundary = np.median(np.array(n_exceeds)) / 2

        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            print("cluser size:{} n_mal:{}".format(cluster_size, n_mal))
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)

        temp_chosen_ids = copy.deepcopy(chosen_ids)
        final_chosen = copy.deepcopy(chosen_ids)
        for i in range(len(chosen_ids) - 1, -1, -1):
            if clusters[i] in deleted_cluster_ids:
                del final_chosen[i]

        if len(final_chosen) == 0:
            final_chosen = temp_chosen_ids

        return final_chosen

    def _global_watermarking_test_sub(self, test_data, model=None):
        """
        应该是测量当前模型对水印任务的执行效果\n
        TODO 具体写还没看
        """
        if model == None:
            model = self.helper.global_model
        model.eval()
        total_loss = 0
        dataset_size = 0
        correct = 0
        data_iterator = test_data
        wm_label_sum_list = [0 for i in range(self.helper.num_classes)]
        wm_label_correct_list = [0 for i in range(self.helper.num_classes)]
        wm_label_acc_list = [0 for i in range(self.helper.num_classes)]
        wm_label_dict = dict()
        for i in range(self.helper.num_classes):
            wm_label_dict[i] = 0
        ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        # client_features = []
        for batch_id, batch in enumerate(data_iterator):
            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)
            output = model(data)
            # features = model.features(data).detach().cpu()
            # client_features.extend(((features.cpu().numpy()).flatten()).tolist())
            # client_features.append(features)
            total_loss += ce_loss(output, targets).item()
            pred = output.data.max(1)[1]

            for pred_item in pred:
                wm_label_dict[pred_item.item()] += 1
            # 统计每个类别, 本来有多少数据wm_label_sum_list, 预测正确有多少数据wm_label_correct_list
            for target_label in range(self.helper.num_classes):
                wm_label_targets = torch.ones_like(targets) * target_label
                wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))

                wm_label_sum_list[target_label] += wm_label_index.cpu().sum().item()
                wm_label_correct_list[target_label] += pred.eq(targets.data.view_as(pred))[
                    wm_label_index.bool()].cpu().sum().item()

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            dataset_size += len(targets)

        watermark_acc = 100.0 * (float(correct) / float(dataset_size))
        for i in range(self.helper.num_classes):
            wm_label_dict[i] = round(wm_label_dict[i] / dataset_size, 2)
        for target_label in range(self.helper.num_classes):
            wm_label_acc_list[target_label] = round(
                100.0 * (float(wm_label_correct_list[target_label]) / float(wm_label_sum_list[target_label])), 2)

        wm_label_acc = max(wm_label_acc_list)
        wm_index_label = wm_label_acc_list.index(wm_label_acc)
        total_l = total_loss / dataset_size
        total_l = np.nan_to_num(total_l, nan=0.2)
        # if

        model.train()
        return (total_l, watermark_acc, wm_label_acc, wm_index_label, wm_label_acc_list, wm_label_dict)

    # 预处理-----先做水印ood数据的训练
    def pre_process(self, test_data, round):
        """
        在正式聚合客户端更新之前, 由服务器主动在全局模型中植入水印\n
        round就是epoch\n
        wm_data 和 test_data 不一样
        
        wm_data 是水印数据, 来源于预定义的 OOD 数据集
        """
        print(f"[*] enter pre_process  ")
        if round in self.watermarking_rounds:
            # TOANSWER 这里似乎没进入过?
            print(f"[*] enter pre_process - if | round is {round} ")
            # 备份全局模型
            target_params_variables = dict()
            for name, param in self.helper.global_model.state_dict().items():
                target_params_variables[name] = param.clone()
            # 水印任务之前的bn值
            before_wm_injection_bn_stats_dict = dict()
            for key, value in self.helper.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    before_wm_injection_bn_stats_dict[key] = value.clone().detach()
                    
            print(f"[***] klog before_wm_injection_bn_stats_dict -> {before_wm_injection_bn_stats_dict}")
            # 开始水印任务
            print(f"benign inserting new watermarking")
            
            wm_data = copy.deepcopy(self.wm_data)
            self._global_watermark_injection(watermark_data=wm_data,
                            test_data=test_data,
                            target_params_variables=target_params_variables,
                            model=self.helper.global_model,
                            round=round)

            watermarking_update_norm = self._model_dist_norm(self.helper.global_model, target_params_variables)
            print(f"watermarking update norm is :{watermarking_update_norm}")
            # 测试指标任务
            wm_data = copy.deepcopy(self.wm_data)
            loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.helper.global_model)
            print(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

            # 记录训练后的bn参数
            for key, value in self.helper.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.after_wm_injection_bn_stats_dict[key] = value.clone().detach()

            self.check_model.copy_params(self.helper.global_model.state_dict())
            for key, value in self.check_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.check_model.state_dict()[key].copy_(before_wm_injection_bn_stats_dict[key])
                    # 恢复bn参数( 直接把括号内的数据覆盖到目标张量上 )
                    self.helper.global_model.state_dict()[key].copy_(before_wm_injection_bn_stats_dict[key])
            print(f"after replace wm bn with original bn:")

    # 计算模型和原始模型的差距(避免偏差过大)
    # TOANSWER 两个函数的区别在哪里呢? 为什么使用两个呢?
    def _model_dist_norm_var(self, model, target_params_variables, norm=2):
        """
        将整个神经网络所有层的参数差异合并成一个巨大的向量, 并计算其范数(距离)
        """
        size = 0 # 统计模型总共有多少个参数
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0) # 在 GPU (CUDA) 上申请一块连续的、大小为 size 的浮点数内存空间
        size = 0
        # layer - target_params_variables[name] -> 计算当前层权重与原始备份之间的差值矩阵(和下面不同, 这里没有使用 .data)
        # 把模型各层分布式的差异点, 合并成一个长向量
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]
        # 计算张量的 L_norm 范数
        return torch.norm(sum_var, norm)

    def _projection(self, target_params_variables):
        """
        target_params_variables : 植入水印前全局模型的备份
        强制性的约束操作, 确保模型在水印训练后, 依然保持在原始模型周围的一个极小的半径范围内
        """
        model_norm = self._model_dist_norm(self.helper.global_model, target_params_variables)
        #  TOANSWER 按理说self.helper.config["global_is_projection_grad"]应该表示决定了是否开启约束的开关, 
        # 但是cifar100.yaml line 116 -> global_is_projection_grad: False ? 默认不约束?还是说会跑很多次?
        if model_norm > 0.8 and self.helper.config["global_is_projection_grad"]:
            print(f" klog enter _projection - if ")
            # 计算缩放比例?
            norm_scale = 0.8 / model_norm
            for name, param in self.helper.global_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                # 实际的缩放操作, 等比例缩放每一层的位移, 将模型拉回到半径为 0.8 的边界上
                param.data.copy_(target_params_variables[name]+clipped_difference)
        return True

    def _model_dist_norm(self, model, target_params):
        """ 
        简单计算 计算的是两个模型参数之间的 欧几里得距离 即 L2 范数
        Return : Float math.sqrt(squared_sum)
        """
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    # 全局水印任务-----在下发模型之前完成
    def _global_watermark_injection(self, watermark_data, test_data, target_params_variables, round=None, model=None):
        """
        self._global_watermark_injection(watermark_data=wm_data,\n
                test_data=test_data, 干净的测试数据集\n
                target_params_variables=target_params_variables(植入水印前全局模型的备份),\n
                model=self.helper.global_model,\n
                round=round)\n
        在保证模型不偏离原始状态太远的前提下, 强行让模型学会识别水印数据(OOD数据)
        """
        if model == None:
            model = self.helper.global_model
        model.train()

        total_loss = 0
        # self._loss_function()
        lr = 0.0008  # 0.001  0.005(cifar100,mnist)
        momentum = 0.9 # 动量 
        weight_decay = 0.0005 # 权重衰减

        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=self.helper.config['global_milestones'],
                                                              gamma=self.helper.config['global_lr_gamma'])

        print(f"wm_mu:{self.wm_mu}")

        retrain_no_times = 200  # 50 200 
        # 外层循环 (internal_round): 执行 200 次完整的微调
        for internal_round in range(retrain_no_times):

            # if internal_round % 50 == 0:
            #     print(f"global watermarking injection round:{internal_round}")
            data_iterator = copy.deepcopy(watermark_data)

            # 内层循环 (enumerate(data_iterator)): 遍历水印数据集的每一个 Batch
            for batch_id, watermark_batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                wm_data, wm_targets = watermark_batch
                wm_data = wm_data.cuda().detach().requires_grad_(False)
                wm_targets = wm_targets.cuda().detach().requires_grad_(False)

                data = wm_data
                targets = wm_targets

                output = model(data)

                class_loss = nn.functional.cross_entropy(output, targets)
                distance_loss = self._model_dist_norm_var(model, target_params_variables)
                # FOCUS 惩罚因子. 控制两个 Loss 的权重. mu越大, 模型更新越保守, 水印植入越浅, 但主任务精度保得越好
                # TOANSWER 为什么是 self.wm_mu / 2, 为了在求导时抵消系数 2 ? 唉, 优化理论没学, 还是数学没学好
                # 如何通过 Loss 自动引导梯度走向那些与主任务无关的方向? 因为如果修改"敏感神经元", 主任务的 class_loss 会大幅波动, 因此, 梯度会自动滑向那些对 class_loss 贡献小、但能满足识别水印需求的参数位移?
                loss = class_loss + (self.wm_mu / 2) * distance_loss

                loss.backward()
                self.optimizer.step()
                # 矫正模型(如果太偏了的话)
                self._projection(target_params_variables)
                total_loss += loss.data

                if internal_round == retrain_no_times - 1 and batch_id == 0:

                    wm_data = copy.deepcopy(self.wm_data)
                    loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data,model=model)
                    print(
                        f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

                    print(f" ")

            self.scheduler.step()

        return True

    def _indicator(self, local_model_state_dict, wm_data):
        benign_client = []
        label_inds = []
        label_acc_ws = []
        total_loss = []

        for ind, model_state_dict in enumerate(local_model_state_dict):
            self.check_model.copy_params(self.helper.global_model.state_dict())
            for name, data in model_state_dict.items():
                if "num_batches_tracked" in name:
                    continue
                # 替换bn层
                if "running" in name:
                    new_value = self.after_wm_injection_bn_stats_dict[name]
                else:
                    new_value = data.clone().detach()
                self.check_model.state_dict()[name].copy_(new_value)
            wm_copy_data = copy.deepcopy(wm_data)
            c_loss, _, label_acc_w, label_ind, _, _\
                = self._global_watermarking_test_sub(test_data=wm_copy_data, model=self.check_model)

            label_inds.append(label_ind)
            label_acc_ws.append(label_acc_w)
            total_loss.append(c_loss)
            if label_acc_w < self.helper.config["thred"]:
                benign_client.append(ind)
        print(f"loss: {total_loss}")
        print(f"label ind:{label_inds}")
        print(f"label acc wm:{label_acc_ws}")
        return benign_client, label_inds, label_acc_ws, total_loss

    def ood_indicator(self, sampled_participants, client_models):

        wm_data = copy.deepcopy(self.wm_data)
        local_model_state_dict = []
        for i in sampled_participants:
            local_model_state_dict_sub = dict()
            for name, param in client_models[i].state_dict().items():
                local_model_state_dict_sub[name] = param.clone().detach()
            local_model_state_dict.append(local_model_state_dict_sub)
        chosen_id, label_ind, label_acc_wm, total_loss= self._indicator(local_model_state_dict, wm_data)

        final_id = []
        for i in chosen_id:
            final_id.append(sampled_participants[i])

        print(f"Indicator final choose id: {final_id}")
        return final_id

    def second_filter(self, sampled_participants):
        wm_data = copy.deepcopy(self.wm_data)
        local_model_state_dict = []
        for i in sampled_participants:
            local_model_state_dict_sub = dict()
            for name, param in self.helper.client_models[i].state_dict().items():
                local_model_state_dict_sub[name] = param.clone().detach()
            local_model_state_dict.append(local_model_state_dict_sub)

        chosen_id, label_ind, label_acc_wm, total_loss= self._indicator(local_model_state_dict, wm_data)
        print("suspicious first clients accepted are {0}".format(chosen_id))

        max_loss = max(total_loss)
        max_index = total_loss.index(max_loss)
        avg_loss = sum(total_loss)/len(total_loss)
        t_loss = np.array(total_loss).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(t_loss)

        labels = []
        for i in range(len(total_loss)):
            if kmeans.labels_[i] == np.argmin(kmeans.cluster_centers_):
                labels.append(0)
            else:
                labels.append(1)
        print(f"kmeans cluster centers: {kmeans.cluster_centers_}")
        print(f"kmeans class result: {labels}")

        index = 0
        for i in total_loss:
            # if i>0.1 and index in chosen_id:
            #     chosen_id.remove(index)
            # else:
            #     if i >= avg_loss and index in chosen_id:
            #         chosen_id.remove(index)
            #     elif i<avg_loss and index not in chosen_id:
            #         chosen_id.append(index)
            if i >= avg_loss and index in chosen_id:
                chosen_id.remove(index)
            elif i < avg_loss and index not in chosen_id:
                chosen_id.append(index)
            index += 1

        # 可疑对象
        suspicious_id = []
        for i in range(len(sampled_participants)):
            if i not in chosen_id:
                suspicious_id.append(i)

        for i in range(len(suspicious_id)):
            index = suspicious_id[i]
            if label_ind[index] != label_ind[max_index] and labels[index] == 0 and total_loss[suspicious_id[i]]<0.1:
                chosen_id.append(index)

        print("suspicious second clients accepted are {0}".format(chosen_id))

        new_loss = []
        new_id = []
        for i in chosen_id:
            new_loss.append(total_loss[i])
            new_id.append(i)

        new_loss = np.array(new_loss).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(new_loss)
        labels = []
        for i in range(len(new_loss)):
            if kmeans.labels_[i] == np.argmin(kmeans.cluster_centers_):
                labels.append(0)
            else:
                labels.append(1)
        print(f"kmeans cluster centers: {kmeans.cluster_centers_}")
        print(f"kmeans class result: {labels}")
        lens = len(chosen_id)
        tmp_id = copy.deepcopy(chosen_id)
        print(f"len: {lens}")
        print(f"tmp_id: {tmp_id}")

        for i in range(lens):
            if labels[i]==1:
                # print(i)
                id = tmp_id[i]
                chosen_id.remove(id)

        final_id = []
        for i in chosen_id:
            final_id.append(sampled_participants[i])
        print(f"suspicious accept: {final_id}")

        return final_id


