import torch
import numpy as np
from collections import OrderedDict
import copy
import csv
import json
import os

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster

from fl_utils.utils import flatten_update, get_layer_name_for_dataset

class FedAvgAggregator:
    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        lr = 1
        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        chosen_id = sampled_participants
        for i in chosen_id:
            index = sampled_participants.index(i)
            client_weight = weight_accumulator_by_client[index]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                averaged_weights[name] += client_weight[name]

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * (1 / len(chosen_id)) * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(self.helper.device))

        return True


class APRAAggregator:
    """
    APRA: Adaptive Progressive Robust Aggregation 自适应渐进鲁棒聚合

    Four stages:
      1. 双重指标提取（扁平权重 + NBD (Neuron Bias Deviation) / NDIF (Noise Differences) ）
      2. 基于 Epoch 衰减 MAD 阈值的自适应统计预过滤
      3. 基于轮廓系数自动选择 K 值的层次聚类
      4. 基于置信度门控自适应裁剪的信任加权聚合
    """

    def __init__(self, helper):
        self.helper = helper
        folder_path = self.helper.config.get("folder_path", ".")
        self.apra_client_trace_path = os.path.join(folder_path, "apra_client_trace.csv")
        self.apra_round_summary_path = os.path.join(folder_path, "apra_round_summary.csv")

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        """

        Args:
            global_model (_type_): _description_
            weight_accumulator (_type_): _description_
            weight_accumulator_by_client (_type_): _description_
            client_models (List): self.helper.client_models[participant_id].load_state_dict(model.state_dict())
            sampled_participants (_type_): _description_
            epoch (_type_): _description_

        Returns:
            _type_: _description_
        """
        config = self.helper.config
        device = self.helper.device

        n_clients = len(sampled_participants)
        if n_clients < 2:
            update_norms = np.array([
                self._update_l2_norm(weight_accumulator_by_client[i])
                for i in range(len(weight_accumulator_by_client))
            ])
            self._record_apra_trace(
                epoch=epoch,
                sampled_participants=sampled_participants,
                update_norms=update_norms,
                features=np.zeros((n_clients, 0)),
                mad_mask=np.ones(n_clients, dtype=bool),
                mad_effective_mask=np.ones(n_clients, dtype=bool),
                mad_info={"fallback_used": True, "reason": "n_clients < 2"},
                cluster_labels=np.full(n_clients, -1),
                selected_cluster=None,
                cluster_raw_mask=np.ones(n_clients, dtype=bool),
                cluster_effective_mask=np.ones(n_clients, dtype=bool),
                cluster_info={"fallback_used": True, "reason": "n_clients < 2"},
                chosen_ids=sampled_participants,
                trust_weights=np.ones(n_clients) / max(n_clients, 1),
                clip_info={},
            )
            return self._simple_average(global_model, weight_accumulator_by_client, sampled_participants)

        # Stage 1: Multi-dimensional feature extraction
        features, update_norms = self._extract_features(
            global_model, client_models, weight_accumulator_by_client,
            sampled_participants
        )

        # Stage 2: Adaptive statistical pre-filtering
        mask, mad_info = self._adaptive_mad_filter(features, update_norms, epoch)
        n_accepted = mask.sum()
        print(f"APRA Stage1: {n_accepted}/{n_clients} clients pass MAD filter")

        if n_accepted < 2:
            filtered_ids = sampled_participants.copy()
            filtered_features = features.copy()
            filtered_indices = list(range(n_clients))
            mad_info["fallback_used"] = True
        else:
            filtered_indices = np.where(mask)[0].tolist()
            filtered_ids = [sampled_participants[i] for i in filtered_indices]
            filtered_features = features[mask]
            mad_info["fallback_used"] = False
        mad_effective_mask = np.zeros(n_clients, dtype=bool)
        mad_effective_mask[filtered_indices] = True

        # Stage 3: Hierarchical clustering + auto cluster selection
        cluster_labels_by_global_index = np.full(n_clients, -1)
        cluster_raw_mask_by_global_index = np.zeros(n_clients, dtype=bool)
        cluster_info = {"fallback_used": False, "cluster_scores": {}, "best_k": None, "best_score": None}
        selected_cluster = None
        if len(filtered_indices) >= 3:
            cluster_labels, selected_cluster, cluster_info = self._hierarchical_cluster(filtered_features)
            for local_i, global_i in enumerate(filtered_indices):
                cluster_labels_by_global_index[global_i] = int(cluster_labels[local_i])
            print(f"APRA Stage2: {len(set(cluster_labels))} clusters, selected cluster {selected_cluster} "
                  f"({sum(cluster_labels == selected_cluster)} clients)")

            cluster_mask = cluster_labels == selected_cluster
            for local_i, global_i in enumerate(filtered_indices):
                cluster_raw_mask_by_global_index[global_i] = bool(cluster_mask[local_i])
            trusted_global_indices = [filtered_indices[i] for i in range(len(filtered_indices)) if cluster_mask[i]]
            if len(trusted_global_indices) < 2:
                trusted_global_indices = filtered_indices
                trust_features = filtered_features
                cluster_info["fallback_used"] = True
            else:
                trust_features = filtered_features[cluster_mask]
        else:
            trusted_global_indices = filtered_indices
            trust_features = filtered_features
            cluster_raw_mask_by_global_index[filtered_indices] = True
            cluster_info = {
                "fallback_used": True,
                "reason": "fewer than 3 clients after MAD",
                "cluster_scores": {},
                "best_k": None,
                "best_score": None,
            }

        cluster_effective_mask = np.zeros(n_clients, dtype=bool)
        cluster_effective_mask[trusted_global_indices] = True

        chosen_ids = [sampled_participants[i] for i in trusted_global_indices]
        print(f"APRA Final: {len(chosen_ids)} clients selected for aggregation")

        # Stage 4: Trust-weighted aggregation with adaptive clipping
        trust_weights = self._compute_trust_weights(trust_features)

        clip_info = self._weighted_average_with_clip(
            global_model,
            weight_accumulator_by_client,
            chosen_ids,
            sampled_participants,
            trust_weights,
            epoch,
        )

        self._record_apra_trace(
            epoch=epoch,
            sampled_participants=sampled_participants,
            update_norms=update_norms,
            features=features,
            mad_mask=mask,
            mad_effective_mask=mad_effective_mask,
            mad_info=mad_info,
            cluster_labels=cluster_labels_by_global_index,
            selected_cluster=selected_cluster,
            cluster_raw_mask=cluster_raw_mask_by_global_index,
            cluster_effective_mask=cluster_effective_mask,
            cluster_info=cluster_info,
            chosen_ids=chosen_ids,
            trust_weights=trust_weights,
            clip_info=clip_info,
        )

        return True

    # ===================== Stage 1: Feature Extraction =====================

    def _extract_features(self, global_model, client_models, weight_accumulator_by_client,
                          sampled_participants):
        config = self.helper.config
        dataset = config['dataset']
        layer_name = get_layer_name_for_dataset(dataset)
        device = self.helper.device
        
        # 将所有客户端的压平权重组合成一个二维矩阵 , (客户端数量, 输出层参数数量)
        flat_updates = []
        # 所有客户端整个模型更新的 L2 范数
        update_norms = []

        for idx, client_id in enumerate(sampled_participants):
            update = weight_accumulator_by_client[idx]
            flat = flatten_update(update, layer_names=[layer_name])
            if len(flat) == 0:
                flat = np.zeros(1)
            flat_updates.append(flat)

            # Compute L2 norm
            l2 = 0.0
            for name, data in update.items():
                if 'num_batches_tracked' in name:
                    continue
                l2 += torch.norm(data, p=2).item() ** 2
            update_norms.append(np.sqrt(l2))

        flat_features = np.array(flat_updates)

        # 双重指标拼接（NBD/NDIF）
        if config['apra_use_nbd_ndif'] and client_models is not None:
            nbd_ndif = self._extract_nbd_ndif(global_model, client_models, sampled_participants)
            if nbd_ndif is not None and len(nbd_ndif) == len(sampled_participants):
                flat_features = np.hstack([flat_features, nbd_ndif])

        # PCA dimensionality reduction
        n_components = min(config['apra_pca_components'], *flat_features.shape)
        if flat_features.shape[0] >= 2 and n_components >= 2:
            pca = PCA(n_components=n_components)
            if np.isnan(flat_features).any() or np.isinf(flat_features).any():
                print("Warning: flat_features contains NaN or Inf. Cleaning data before PCA...")
                # 将 NaN 替换为 0
                flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=1.0, neginf=-1.0)
            features = pca.fit_transform(flat_features)
        else:
            features = flat_features

        return features, np.array(update_norms)

    def _extract_nbd_ndif(self, global_model, client_models, sampled_participants):
        device = self.helper.device
        config = self.helper.config
        dataset = config['dataset']

        try:
            global_state = list(global_model.state_dict().values())
            if len(global_state) < 2:
                return None
            global_weight = global_state[-2]
            global_bias = global_state[-1]

            # 遍历客户端计算偏置差值
            # 用于衡量模型的输出神经元（对应分类任务中的每一个类别）在本地训练过程中被改变的剧烈程度（即能量分布）
            nbd_features = []
            for client_id in sampled_participants:
                if client_id >= len(client_models):
                    nbd_features.append([0, 0])
                    continue
                client_state = list(client_models[client_id].state_dict().values())
                if len(client_state) < 2:
                    nbd_features.append([0, 0])
                    continue
                client_bias = client_state[-1]
                nbd_features.append(
                    np.array(client_bias.cpu().numpy() - global_bias.cpu().numpy()).flatten()
                )

            nbd_features = np.array(nbd_features)

            # NDIF: 构造随机噪声输入作为探针输入到模型中，用以强行触发隐藏的后门
            if dataset == 'mnist':
                rand_input = torch.randn((8, 1, 28, 28)).to(device)
            elif dataset == 'tiny-imagenet-200':
                rand_input = torch.randn((16, 3, 224, 224)).to(device)
            else:
                rand_input = torch.randn((32, 3, 32, 32)).to(device)
            # 通过让全局模型预测一组“随机噪声”，计算出干净模型对无意义输入时的“盲测输出概率分布”，作为后续揪出后门攻击者的基准底线
            """
            对于一个干净的、正常的全局模型来说，由于输入的 rand_input 是毫无意义的纯随机噪声，模型不应该在任何特定类别上表现出强烈的偏好。
            因此，global_ndif 算出来的概率分布理论上应该非常接近均匀分布
            如果是诚实客户端：它的模型也是干净的，对噪声的预测也是均匀的。
            """
            global_output = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0) + 1e-8
            ndif_features = []
            for client_id in sampled_participants:
                if client_id >= len(client_models):
                    ndif_features.append([0])
                    continue
                """
                衡量本地模型在哪些类别上产生了异常的“兴奋”。如果某个客户端是后门攻击者，
                即使输入的是噪声，其模型也可能强烈地将输出指向“后门目标类”，导致该比值在某个维度上激增
                """
                client_output = torch.mean(torch.softmax(client_models[client_id](rand_input), dim=1), dim=0)
                ndif = (client_output / global_output).cpu().detach().numpy()
                ndif_features.append(ndif)

            ndif_features = np.array(ndif_features)

            # Concatenate
            combined = np.hstack([nbd_features, ndif_features])
            return combined
        except Exception:
            return None

    # ===================== Stage 2: Adaptive MAD Filter =====================

    def _adaptive_mad_filter(self, features, update_norms, epoch):
        """
        用客户端模型更新的 L2 范数做 MAD 异常值预过滤
        不是直接判断“谁是攻击者”，而是先根据每个客户端更新范数是否异常，做一次统计预过滤
        Args:
            features (numpy.ndarray): 形状通常是 (n_clients, d)。
每一行对应一个被采样客户端的特征向量，由 _extract_features 生成，可能包含输出层 flatten 后的 PCA 特征、NBD/NDIF 特征等
            update_norms (numpy.ndarray): 形状是 (n_clients,)。
每个元素是一个客户端本轮模型更新的整体 L2 范数。顺序与 sampled_participants 完全一致
            epoch (_type_): _description_

        Returns:
            (mask, mad_info): 
            mask 是一个布尔数组，形状是 (n_clients,)，表示哪些客户端通过了 MAD 过滤。mad_info 是一个字典，包含 MAD 过滤的统计信息和是否使用了安全保底机制等
            
        """
        config = self.helper.config
        n_clients = len(update_norms)

        if n_clients < 3:
            """
            median_norm：本轮更新范数的中位数。如果没有客户端，则为 0.0。
            mad：Median Absolute Deviation，中位绝对偏差。这里因为客户端太少，没有实际计算，直接记为 0.0。
            k：MAD 阈值。这里没有使用，所以是 None。
            modified_z_scores：每个客户端的修正 z-score。客户端太少时全置为 0.0。
            safety_keep_used：是否触发安全保留机制。这里没有触发。
            fallback_used：是否走 fallback。这里是 True。
            """
            return np.ones(n_clients, dtype=bool), {
                "median_norm": float(np.median(update_norms)) if n_clients else 0.0,
                "mad": 0.0,
                "k": None,
                "modified_z_scores": [0.0 for _ in range(n_clients)],
                "safety_keep_used": False,
                "fallback_used": True,
            }

        # 初始 MAD 阈值。通常较大，表示训练早期更宽松
        k_init = config['apra_k_init']
        # 阈值衰减速度
        k_decay = config['apra_k_decay']
        # 当前轮实际使用的 MAD 阈值
        # 利用指数衰减公式计算本轮的动态阈值倍数 k
        # 训练早期 k 大，过滤宽松；随着 epoch 增大，k 逐渐下降，过滤变严格；最低不会低于 2.0
        k = max(2.0, k_init * np.exp(-k_decay * epoch))

        # 所有客户端更新范数的中位数
        # 相比于平均值，中位数极不容易受到极少数恶意大攻击者的拉扯
        median_norm = np.median(update_norms)
        # 中位绝对偏差
        # 它衡量客户端更新范数围绕中位数的离散程度。相比均值和标准差，MAD 对异常值更鲁棒
        mad = np.median(np.abs(update_norms - median_norm))

        # 计算修正 z-score
        # modified_z_scores: numpy.ndarray，形状 (n_clients,)
        # 第 i 个客户端的更新范数距离中位数有多异常
        # 公式里的 0.6745 是 MAD 转换到类标准差尺度的常用系数。如果数据近似正态分布，这个修正 z-score 可以类似普通 z-score 使用
        modified_z_scores = 0.6745 * (update_norms - median_norm) / mad
        # (n_clients,)
        # 异常大和异常小的更新范数都会被筛掉
        mask = np.abs(modified_z_scores) <= k

        # 如果 MAD 筛完后剩下的客户端太少，就触发安全保留, 至少保留 2 个客户端，或者至少保留一半客户端
        safety_keep_used = False
        if mask.sum() < max(2, n_clients // 2):
            sorted_indices = np.argsort(np.abs(modified_z_scores))
            keep_count = max(2, n_clients // 2)
            mask[:] = False
            mask[sorted_indices[:keep_count]] = True
            safety_keep_used = True

        print(f"APRA MAD: median={median_norm:.4f} mad={mad:.4f} k(epoch={epoch})={k:.2f} "
              f"kept={mask.sum()}/{n_clients}")

        return mask, {
            "median_norm": float(median_norm),
            "mad": float(mad),
            "k": float(k),
            "modified_z_scores": modified_z_scores.tolist(),
            "safety_keep_used": safety_keep_used,
        }

    # ===================== Stage 3: Hierarchical Clustering =====================

    def _hierarchical_cluster(self, features):
        """
        接收经过第一阶段清洗和降维后的多维特征矩阵 features，自动评估将客户端分为几个群体（簇）最合理，
        然后利用凝聚层次聚类算法将客户端归类。
        最后，在一堆划分好的群体中，通过综合“群体人数”和“内部方向一致性”，锁定那个代表全场“最干净、最正常”的客户端大部队群体
        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        n = len(features)
        if n < 3:
            return np.zeros(n, dtype=int), 0, {
                "best_k": 1,
                "best_score": None,
                "cluster_scores": {0: float(n)},
                "fallback_used": True,
            }

        # 聚类至少需要将数据分为 2 个群体
        #  n // 2 表示最多认为有一半的客户端可能是恶意/异常的，从而聚成独立的类
        max_k = min(n - 1, max(2, n // 2))
        best_k = 2
        best_score = -1

        # 自适应 K 值搜索
        # 利用轮廓系数自动寻找最佳 K 值
        for k in range(2, max_k + 1):
            # 凝聚层次聚类（Agglomerative Clustering）
            # 使用欧几里得距离来衡量客户端特征之间的疏密程度
            # 采用沃德最小方差定向链接法，最小化所有簇内数据点与其簇中心之间的平方距离和
            # 使每次合并簇时引起的总平方误差和增量最小，这能让聚类结果非常紧凑、均匀
            clustering = AgglomerativeClustering(
                n_clusters=k, metric='euclidean', linkage='ward'
            )
            labels = clustering.fit_predict(features)
            if len(set(labels)) < 2:
                continue
            try:
                # 它是评估聚类质量的核心。它综合考虑了簇内紧凑度（客户端离自己群体的其他人有多近）和簇间分离度（客户端离其他群体的组织有多远）
                # 轮廓系数越接近 1，说明分类越完美；接近 0，说明簇重叠较多；接近 -1，说明分类错误
                score = silhouette_score(features, labels)
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best_k = k

        # 执行最终聚类
        final_clustering = AgglomerativeClustering(
            n_clusters=best_k, metric='euclidean', linkage='ward'
        )
        labels = final_clustering.fit_predict(features)

        # 各个簇打分
        # 初始化一个用于存储各个群体得分的字典 cluster_scores
        cluster_scores = {}
        for c in np.unique(labels):
            indices = np.where(labels == c)[0]
            cluster_size = len(indices)
            # 如果一个群体里只有 1 个客户端（或者空了），说明它是一个极其不合群的“孤儿点”（通常是发动了剧烈后门攻击或参数投毒的恶意客户端）。直接将其得分设为 -1
            if cluster_size <= 1:
                cluster_scores[c] = -1
                continue
            cluster_data = features[indices]
            # 计算群体内部所有人两两之间的余弦相似度（Cosine Similarity），然后求平均值 internal_sim
            """
            在联邦学习防御中，恶意攻击者的梯度方向往往跟大众相反或者偏离。
            正常客户端由于都在向着收敛方向优化，它们更新的方向（余弦角度）应当高度一致
            """
            internal_sim = np.mean(sk_cosine_similarity(cluster_data))
            # 群体最终得分 = 群体人数 * 内部方向相似度
            cluster_scores[c] = cluster_size * internal_sim

        selected_cluster = max(cluster_scores, key=cluster_scores.get)

        return labels, selected_cluster, {
            "best_k": int(best_k),
            "best_score": float(best_score),
            "cluster_scores": {int(k): float(v) for k, v in cluster_scores.items()},
            "fallback_used": False,
        }

    # ===================== Stage 4: Trust-Weighted + Clip =====================

    def _compute_trust_weights(self, features):
        """
        在 MAD 和聚类筛选之后，对最终保留下来的客户端再计算一个“信任权重”，用于后续加权聚合和裁剪
        理论上，越可疑的客户端权重越低，越可信的客户端权重越高
        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        n = len(features)
        if n <= 1:
            return np.ones(n) / n

        # 客户端之间的两两余弦相似度矩阵
        cs = sk_cosine_similarity(features)
        # 对角线处理：每个客户端与自己的相似度是 1，但我们不希望这个值干扰后续的最大相似度计算，所以先把对角线置为 -1
        np.fill_diagonal(cs, -1.0)
        epsilon = 1e-5
        # 每个客户端与其他客户端的最大相似度
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    """
                    如果客户端 j 整体上比客户端 i 更容易和别人相似，
                    那么在计算 i 对 j 的相似度时，对这个相似度做一定缩放，避免误伤某些正常客户端
                    """
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # 孤立度
        wv = 1 - np.max(cs, axis=1)
        wv = np.clip(wv, 0, 1)

        # 归一化
        max_wv = np.max(wv)
        if max_wv > 0:
            wv = wv / max_wv
        wv[wv == 1] = 0.99

        # Logit transform
        wv = np.log(wv / (1 - wv) + epsilon) + 0.5
        wv = np.clip(wv, 0, 1)
        
        wv = 1.0 - wv

        # 再归一化
        if np.sum(wv) > 0:
            wv = wv / np.sum(wv)
        else:
            wv = np.ones(n) / n

        return wv

    def _weighted_average_with_clip(self, global_model, weight_accumulator_by_client,
                                     chosen_ids, sampled_participants, trust_weights, epoch):
        config = self.helper.config
        device = self.helper.device
        lr = 1.0
        base_clip = config['apra_base_clip']

        n = len(chosen_ids)
        if n == 0:
            return {}

        # Compute adaptive clip factors from trust weights
        min_trust = np.min(trust_weights) if len(trust_weights) > 0 else 1.0 / n
        max_trust = np.max(trust_weights) if len(trust_weights) > 0 else 1.0 / n
        clip_factors = np.zeros(n)
        for i in range(n):
            # Low trust → tighter clip
            w_i = trust_weights[i]
            ratio = (w_i + 1e-8) / (max_trust + 1e-8)
            # 攻击者的缩放值小
            clip_factors[i] = base_clip * ratio

        # Apply clipping to each selected client's update
        clip_info = {}
        for idx_in_list, global_idx in enumerate(chosen_ids):
            local_idx = sampled_participants.index(global_idx)
            update = weight_accumulator_by_client[local_idx]
            cf = clip_factors[idx_in_list]
            norm_before = self._update_l2_norm(update)

            for key in update:
                if 'num_batches_tracked' in key:
                    continue
                data = update[key]
                l2 = torch.norm(data, p=2)
                data.div_(max(1.0, l2.item() / cf))
            clip_info[int(global_idx)] = {
                "trust_weight": float(trust_weights[idx_in_list]),
                "clip_factor": float(cf),
                "update_norm_before_clip": float(norm_before),
                "update_norm_after_clip": float(self._update_l2_norm(update)),
            }

        # Weighted aggregation
        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        for idx_in_list, global_idx in enumerate(chosen_ids):
            local_idx = sampled_participants.index(global_idx)
            if local_idx >= len(weight_accumulator_by_client):
                continue
            client_weight = weight_accumulator_by_client[local_idx]
            w = trust_weights[idx_in_list]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                update = client_weight[name].float() * w
            
                if not torch.is_floating_point(averaged_weights[name]):
                    temp_float = averaged_weights[name].float() + update
                    averaged_weights[name].copy_(temp_float.long())
                else:
                    averaged_weights[name] += update

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(device))

        return clip_info

    def _update_l2_norm(self, update):
        l2 = 0.0
        for name, data in update.items():
            if 'num_batches_tracked' in name:
                continue
            l2 += torch.norm(data, p=2).item() ** 2
        return np.sqrt(l2)

    # ===================== Monitoring =====================

    def _record_apra_trace(self, epoch, sampled_participants, update_norms, features,
                           mad_mask, mad_effective_mask, mad_info,
                           cluster_labels, selected_cluster, cluster_raw_mask,
                           cluster_effective_mask, cluster_info,
                           chosen_ids, trust_weights, clip_info):
        folder_path = self.helper.config.get("folder_path", ".")
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        sampled_ids = [int(x) for x in sampled_participants]
        chosen_set = set(int(x) for x in chosen_ids)
        adversary_ids = self._get_adversary_ids()
        z_scores = mad_info.get("modified_z_scores", [])
        feature_array = np.asarray(features)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(len(sampled_ids), -1)

        trust_by_client = {}
        for idx, client_id in enumerate(chosen_ids):
            if idx < len(trust_weights):
                trust_by_client[int(client_id)] = float(trust_weights[idx])

        rows = []
        for idx, client_id in enumerate(sampled_ids):
            role = "malicious" if client_id in adversary_ids else "benign"
            client_clip_info = clip_info.get(client_id, {})
            feature_vec = feature_array[idx].tolist() if idx < len(feature_array) else []
            rows.append({
                "epoch": int(epoch),
                "client_id": client_id,
                "role": role,
                "is_adversary": int(client_id in adversary_ids),
                "update_norm": self._safe_float(update_norms[idx]) if idx < len(update_norms) else "",
                "mad_z_score": self._safe_float(z_scores[idx]) if idx < len(z_scores) else "",
                "mad_pass": int(bool(mad_mask[idx])) if idx < len(mad_mask) else "",
                "mad_effective_pass": int(bool(mad_effective_mask[idx])) if idx < len(mad_effective_mask) else "",
                "cluster_label": int(cluster_labels[idx]) if idx < len(cluster_labels) else "",
                "selected_cluster": "" if selected_cluster is None else int(selected_cluster),
                "cluster_pass": int(bool(cluster_raw_mask[idx])) if idx < len(cluster_raw_mask) else "",
                "cluster_effective_pass": int(bool(cluster_effective_mask[idx])) if idx < len(cluster_effective_mask) else "",
                "final_selected": int(client_id in chosen_set),
                "trust_weight": self._safe_float(trust_by_client.get(client_id, "")),
                "clip_factor": self._safe_float(client_clip_info.get("clip_factor", "")),
                "update_norm_before_clip": self._safe_float(client_clip_info.get("update_norm_before_clip", "")),
                "update_norm_after_clip": self._safe_float(client_clip_info.get("update_norm_after_clip", "")),
                "feature_norm": self._safe_float(np.linalg.norm(feature_vec)) if len(feature_vec) else "",
                "feature_0": self._safe_float(feature_vec[0]) if len(feature_vec) > 0 else "",
                "feature_1": self._safe_float(feature_vec[1]) if len(feature_vec) > 1 else "",
                "feature_2": self._safe_float(feature_vec[2]) if len(feature_vec) > 2 else "",
            })

        self._append_csv(self.apra_client_trace_path, rows)

        mad_pass_ids = [sampled_ids[i] for i in range(len(sampled_ids)) if mad_mask[i]]
        mad_effective_pass_ids = [sampled_ids[i] for i in range(len(sampled_ids)) if mad_effective_mask[i]]
        cluster_pass_ids = [sampled_ids[i] for i in range(len(sampled_ids)) if cluster_raw_mask[i]]
        cluster_effective_pass_ids = [sampled_ids[i] for i in range(len(sampled_ids)) if cluster_effective_mask[i]]
        final_selected_ids = [client_id for client_id in sampled_ids if client_id in chosen_set]
        final_rejected_ids = [client_id for client_id in sampled_ids if client_id not in chosen_set]

        summary = {
            "epoch": int(epoch),
            "num_sampled": len(sampled_ids),
            "num_adversaries": int(self.helper.config.get("num_adversaries", 0)),
            "sampled_ids": self._json_list(sampled_ids),
            "sampled_benign_ids": self._json_list([x for x in sampled_ids if x not in adversary_ids]),
            "sampled_malicious_ids": self._json_list([x for x in sampled_ids if x in adversary_ids]),
            "mad_median_norm": self._safe_float(mad_info.get("median_norm", "")),
            "mad_mad": self._safe_float(mad_info.get("mad", "")),
            "mad_k": self._safe_float(mad_info.get("k", "")),
            "mad_safety_keep_used": int(bool(mad_info.get("safety_keep_used", False))),
            "mad_fallback_used": int(bool(mad_info.get("fallback_used", False))),
            "mad_pass_ids": self._json_list(mad_pass_ids),
            "mad_reject_ids": self._json_list([x for x in sampled_ids if x not in mad_pass_ids]),
            "mad_effective_pass_ids": self._json_list(mad_effective_pass_ids),
            "mad_effective_reject_ids": self._json_list([x for x in sampled_ids if x not in mad_effective_pass_ids]),
            "mad_pass_benign_ids": self._json_list([x for x in mad_pass_ids if x not in adversary_ids]),
            "mad_pass_malicious_ids": self._json_list([x for x in mad_pass_ids if x in adversary_ids]),
            "mad_reject_benign_ids": self._json_list([x for x in sampled_ids if x not in mad_pass_ids and x not in adversary_ids]),
            "mad_reject_malicious_ids": self._json_list([x for x in sampled_ids if x not in mad_pass_ids and x in adversary_ids]),
            "cluster_best_k": "" if cluster_info.get("best_k") is None else int(cluster_info.get("best_k")),
            "cluster_best_score": self._safe_float(cluster_info.get("best_score", "")),
            "cluster_scores": json.dumps(cluster_info.get("cluster_scores", {}), sort_keys=True),
            "cluster_selected_cluster": "" if selected_cluster is None else int(selected_cluster),
            "cluster_fallback_used": int(bool(cluster_info.get("fallback_used", False))),
            "cluster_pass_ids": self._json_list(cluster_pass_ids),
            "cluster_reject_ids": self._json_list([x for x in sampled_ids if x not in cluster_pass_ids]),
            "cluster_effective_pass_ids": self._json_list(cluster_effective_pass_ids),
            "cluster_effective_reject_ids": self._json_list([x for x in sampled_ids if x not in cluster_effective_pass_ids]),
            "cluster_pass_benign_ids": self._json_list([x for x in cluster_pass_ids if x not in adversary_ids]),
            "cluster_pass_malicious_ids": self._json_list([x for x in cluster_pass_ids if x in adversary_ids]),
            "cluster_reject_benign_ids": self._json_list([x for x in sampled_ids if x not in cluster_pass_ids and x not in adversary_ids]),
            "cluster_reject_malicious_ids": self._json_list([x for x in sampled_ids if x not in cluster_pass_ids and x in adversary_ids]),
            "final_selected_ids": self._json_list(final_selected_ids),
            "final_rejected_ids": self._json_list(final_rejected_ids),
            "final_selected_benign_ids": self._json_list([x for x in final_selected_ids if x not in adversary_ids]),
            "final_selected_malicious_ids": self._json_list([x for x in final_selected_ids if x in adversary_ids]),
            "final_rejected_benign_ids": self._json_list([x for x in final_rejected_ids if x not in adversary_ids]),
            "final_rejected_malicious_ids": self._json_list([x for x in final_rejected_ids if x in adversary_ids]),
            "final_selected_benign_count": len([x for x in final_selected_ids if x not in adversary_ids]),
            "final_selected_malicious_count": len([x for x in final_selected_ids if x in adversary_ids]),
            "final_rejected_benign_count": len([x for x in final_rejected_ids if x not in adversary_ids]),
            "final_rejected_malicious_count": len([x for x in final_rejected_ids if x in adversary_ids]),
        }
        self._append_csv(self.apra_round_summary_path, [summary])

        print(
            "APRA Monitor: "
            f"selected benign={summary['final_selected_benign_ids']} "
            f"selected malicious={summary['final_selected_malicious_ids']} "
            f"rejected benign={summary['final_rejected_benign_ids']} "
            f"rejected malicious={summary['final_rejected_malicious_ids']}"
        )

    def _get_adversary_ids(self):
        if hasattr(self.helper, "adversary_list"):
            return set(int(x) for x in self.helper.adversary_list)
        if self.helper.config.get("is_poison"):
            return set(range(int(self.helper.config.get("num_adversaries", 0))))
        return set()

    def _append_csv(self, path, rows):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _json_list(self, values):
        return json.dumps([int(x) for x in values])

    def _safe_float(self, value):
        if value == "" or value is None:
            return ""
        return float(value)

    def _simple_average(self, global_model, weight_accumulator_by_client, sampled_participants):
        avg = FedAvgAggregator(self.helper)
        return avg.aggregate(global_model, None, weight_accumulator_by_client,
                            None, sampled_participants, 0)
