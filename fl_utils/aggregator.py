import math
import sys
import csv
import json
import pickle

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


class AggRecorder:
    """
    Record human-readable aggregation decisions and machine-readable defense data.

    Config knobs:
      agg_record_enabled: bool, default False
      agg_record_dir: optional output directory, default <folder_path>/agg_records
      agg_record_tensor_mode: "stats" or "full", default "stats"
      agg_record_update_summary: bool, default True
    """

    def __init__(self, helper):
        self.helper = helper
        self.config = getattr(helper, "config", {})
        self.enabled = self._config_bool("agg_record_enabled", False)
        folder_path = self.config.get("folder_path", ".")
        configured_dir = self.config.get("agg_record_dir")
        self.base_dir = configured_dir or os.path.join(folder_path, "agg_records")
        self.pkl_dir = os.path.join(self.base_dir, "pkl")
        self.round_csv = os.path.join(self.base_dir, "agg_rounds.csv")
        self.stage_csv = os.path.join(self.base_dir, "agg_stages.csv")
        self.tensor_mode = str(self.config.get("agg_record_tensor_mode", "stats")).lower()
        if self.tensor_mode not in {"stats", "full"}:
            self.tensor_mode = "stats"
        self.current = None

    def _config_bool(self, key, default=False):
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def begin_round(
        self,
        epoch,
        method,
        sampled_participants,
        weight_accumulator_by_client=None,
    ):
        if not self.enabled:
            return

        sampled_participants = self._to_plain_list(sampled_participants)
        self.current = {
            "epoch": epoch,
            "method": method,
            "status": "running",
            "initial_participants": sampled_participants,
            "final_selected": [],
            "metadata": {
                "dataset": self.config.get("dataset"),
                "attacker_method": self.config.get("attacker_method"),
                "is_poison": self.config.get("is_poison"),
                "num_sampled_participants": self.config.get("num_sampled_participants"),
            },
            "common": {
                "update_summary": self._summarize_updates(
                    sampled_participants,
                    weight_accumulator_by_client,
                ),
            },
            "stages": [],
            "extra": {},
        }

    def record_stage(self, name, selected=None, rejected=None, human=None, data=None):
        if not self.enabled or self.current is None:
            return

        stage = {
            "name": name,
            "selected": self._to_plain_list(selected),
            "rejected": self._to_plain_list(rejected),
            "human": self._to_common_types(human or {}, force_full=True),
            "data": self._to_common_types(data or {}),
        }
        self.current["stages"].append(stage)

    def finish(self, final_selected=None, status="success", extra=None):
        if not self.enabled or self.current is None:
            return

        if final_selected is not None:
            self.current["final_selected"] = self._to_plain_list(final_selected)
        self.current["status"] = status
        if extra:
            self.current["extra"].update(self._to_common_types(extra))

        self._persist(self.current)
        self.current = None

    def fail(self, error):
        if not self.enabled or self.current is None:
            return

        self.record_stage(
            "error",
            human={"error": repr(error)},
            data={"error_type": type(error).__name__, "error": repr(error)},
        )
        self.finish(status="failed")

    def _persist(self, record):
        os.makedirs(self.pkl_dir, exist_ok=True)

        method = str(record["method"]).replace(os.sep, "_")
        pkl_path = os.path.join(
            self.pkl_dir,
            "round_{0:04d}_{1}.pkl".format(int(record["epoch"]), method),
        )
        record["pkl_path"] = pkl_path
        with open(pkl_path, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

        rejected = [
            participant
            for participant in record["initial_participants"]
            if participant not in record["final_selected"]
        ]
        round_row = {
            "epoch": record["epoch"],
            "method": record["method"],
            "status": record["status"],
            "initial_participants": self._json_dumps(record["initial_participants"]),
            "final_selected": self._json_dumps(record["final_selected"]),
            "rejected": self._json_dumps(rejected),
            "num_initial": len(record["initial_participants"]),
            "num_final": len(record["final_selected"]),
            "stage_names": self._json_dumps([stage["name"] for stage in record["stages"]]),
            "pkl_path": pkl_path,
        }
        self._append_csv(self.round_csv, round_row)

        for stage in record["stages"]:
            stage_row = {
                "epoch": record["epoch"],
                "method": record["method"],
                "stage": stage["name"],
                "selected": self._json_dumps(stage["selected"]),
                "rejected": self._json_dumps(stage["rejected"]),
                "human": self._json_dumps(stage["human"]),
            }
            self._append_csv(self.stage_csv, stage_row)

    def _append_csv(self, path, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _summarize_updates(self, participants, weight_accumulator_by_client):
        if not self._config_bool("agg_record_update_summary", True):
            return {}
        if weight_accumulator_by_client is None:
            return {}

        summaries = []
        for participant_id, single_wa in zip(participants, weight_accumulator_by_client):
            layer_norms = {}
            total_sq_norm = 0.0
            for name, value in single_wa.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                if not torch.is_floating_point(value):
                    continue
                norm = float(torch.norm(value.detach().float().cpu(), p=2).item())
                layer_norms[name] = norm
                total_sq_norm += norm * norm
            summaries.append({
                "participant_id": participant_id,
                "total_l2_norm": math.sqrt(total_sq_norm),
                "layer_l2_norms": layer_norms,
            })
        return summaries

    def _to_common_types(self, value, force_full=False):
        if isinstance(value, torch.Tensor):
            return self._tensor_to_common(value, force_full=force_full)
        if isinstance(value, np.ndarray):
            return self._ndarray_to_common(value, force_full=force_full)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._to_common_types(v, force_full=force_full) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_common_types(v, force_full=force_full) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _tensor_to_common(self, tensor, force_full=False):
        cpu_tensor = tensor.detach().cpu()
        result = {
            "type": "tensor",
            "dtype": str(cpu_tensor.dtype),
            "shape": list(cpu_tensor.shape),
        }
        if force_full or self.tensor_mode == "full":
            result["data"] = cpu_tensor.tolist()
        else:
            result["stats"] = self._numeric_stats(cpu_tensor)
        return result

    def _ndarray_to_common(self, array, force_full=False):
        array = np.asarray(array)
        result = {
            "type": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        if force_full or self.tensor_mode == "full":
            result["data"] = array.tolist()
        else:
            result["stats"] = self._numeric_stats(array)
        return result

    def _numeric_stats(self, value):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if array.size == 0 or not np.issubdtype(array.dtype, np.number):
            return {}
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return {"finite_count": 0, "nan_count": int(np.isnan(array).sum())}
        return {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "l2_norm": float(np.linalg.norm(finite.reshape(-1))),
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(array).sum()),
        }

    def _to_plain_list(self, value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (set, tuple)):
            value = list(value)
        elif not isinstance(value, list):
            value = [value]
        return [self._to_common_types(item, force_full=True) for item in value]

    def _json_dumps(self, value):
        return json.dumps(self._to_common_types(value, force_full=True), ensure_ascii=False)


class Aggregator:
    def __init__(self, helper, ood_set):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []
        self.agg_recorder = AggRecorder(helper)
        # 水印数据, 也是 OOD 数据集
        self.wm_data = ood_set # Out-of-Distribution Dataset (分布外数据集), 指的是与模型正常训练任务(In-Distribution, ID)完全无关的数据
        # 添加水印的轮数
        self.watermarking_rounds = [round for round in range(1,100,10)]
        # 水印之后的bn层
        self.after_wm_injection_bn_stats_dict = dict()
        self.wm_mu = self.helper.config["watermarking_mu"] # \mu ? 一个正则化超参数, 限制了模型修改的幅度, 防止植入水印导致模型在正常任务上彻底失效
        if self.helper.config["dataset"] == 'mnist':
            check_model = SimpleMnist()
        elif self.helper.config['dataset'] == 'tiny-imagenet-200':
            check_model = pt_resnet18(num_classes=self.helper.num_classes)
        else:
            channels = getattr(self.helper, 'channel', 3)
            check_model = ResNet18(num_classes=self.helper.num_classes, channels=channels)
            # check_model = ResNet34(num_classes=self.helper.num_classes)
            # check_model = Mobilenet()

        self.check_model = check_model.cuda()
        self.target_label = -1


    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants, epoch):
        """
        TODO 没看
        """
        agg_method = self.helper.config['agg_method']
        self.agg_recorder.begin_round(
            epoch,
            agg_method,
            sampled_participants,
            weight_accumulator_by_client,
        )
        final_selected = sampled_participants
        result = None

        try:
            if agg_method == 'avg':
                result = self.average_models(global_model, weight_accumulator,  weight_accumulator_by_client, sampled_participants)
            elif agg_method == 'apra':
                apraAggregator = APRAAggregator(self.helper, self.agg_recorder)
                final_selected = apraAggregator.aggregate(global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants, epoch)
                result = True
            elif agg_method == 'clip':
                self.clip_updates(weight_accumulator, weight_accumulator_by_client, sampled_participants)
                result = self.average_shrink_models(global_model, weight_accumulator, weight_accumulator_by_client)
            elif agg_method == 'deepsight':
                final_selected = self.deepsight_aggregate_global_model(global_model,client_models,sampled_participants)
                """
                init_ids:[3, 17, 0, 2, 7, 1, 14, 5, 4, 12, 16, 18, 6, 11, 19, 15, 13, 9, 8, 10]
                choosen_ids:[3, 17, 0, 2, 7, 1, 14, 5, 4, 12, 16, 18, 6, 11, 19, 15, 13, 9, 8, 10]
                """
                print(f"init_ids:{sampled_participants}    choosen_ids:{final_selected}")
                result = self.average_chosen_models(weight_accumulator_by_client, final_selected, global_model, sampled_participants)
            elif agg_method == 'foolsgold':
                wv = self.foolsgold_aggr(weight_accumulator_by_client, weight_accumulator, sampled_participants)
                final_selected = [participant_id for participant_id, weight in zip(sampled_participants, wv) if weight > 0]
                result = self.average_fool_models(weight_accumulator_by_client, global_model, sampled_participants, wv)
            elif agg_method == 'rflbat':
                final_selected = self.rflbat_aggr(weight_accumulator_by_client, weight_accumulator, sampled_participants)
                result = self.average_chosen_models(weight_accumulator_by_client, final_selected, global_model, sampled_participants)
            elif agg_method == 'second':
                final_selected = self.second_filter(sampled_participants)
                result = self.average_chosen_models(weight_accumulator_by_client, final_selected, global_model, sampled_participants)
            elif agg_method == 'indicator':
                final_selected = self.ood_indicator(sampled_participants, client_models)
                # chosen_ids = self.rflbat_aggr(weight_accumulator_by_client, weight_accumulator, sampled_participants)
                result = self.average_chosen_models(weight_accumulator_by_client, final_selected, global_model, sampled_participants)
            elif agg_method == 'rfa':
                result = self.rfa_models(
                    global_model,
                    weight_accumulator_by_client,
                    sampled_participants,
                )
            else:
                raise NotImplementedError
        except Exception as exc:
            self.agg_recorder.fail(exc)
            raise

        self.agg_recorder.finish(final_selected=final_selected)
        return result

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
        self.agg_recorder.record_stage(
            "avg",
            selected=chosen_id,
            human={
                "strategy": "fedavg",
                "num_participants": len(chosen_id),
            },
        )
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


    def clip_updates(self, weight_accumulator, weight_accumulator_by_client, sampled_participants):
        """
        限制每个客户端对全局模型更新的“贡献强度”，防止某个客户端（尤其是攻击者）通过发送极大的权重更新来恶意偏移全局模型\n
        weight_accumulator 构造\n
        for name, data in self.helper.global_model.state_dict().items():\n
            .....\n
            weight_accumulator[name] = torch.zeros_like(data) \n       
        """
        clip_records = []
        for key in weight_accumulator:
            # num_batches_tracked 是 Batch Normalization 层的一个统计量，它记录训练了多少个 batch
            # 这个值是一个整数，不需要进行梯度裁剪或 L2 范数计算，所以代码通过 if 语句将其跳过
            if 'num_batches_tracked' not in key:
                update = weight_accumulator[key]
                # 计算更新量 update 的 L2 范数（即向量的长度）
                l2_update = torch.norm(update, p=2) 

                # print(f"[*] klog: Layer {key} L2 Norm = {l2_update.item():.4f}") 
                               
                # FOCUS clip_factor: 1
                divisor = max(1, l2_update/self.helper.config['clip_factor'])
                update.div_(divisor)
                divisor_value = float(divisor.item()) if isinstance(divisor, torch.Tensor) else float(divisor)
                clip_records.append({
                    "layer": key,
                    "l2_norm_before_clip": float(l2_update.item()),
                    "divisor": divisor_value,
                    "clip_scale": 1.0 / divisor_value,
                })
        self.agg_recorder.record_stage(
            "clip",
            selected=sampled_participants,
            human={
                "clip_factor": self.helper.config['clip_factor'],
                "num_layers_clipped": len(clip_records),
            },
            data={"layers": clip_records},
        )
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
        if self.helper.config["dataset"] == 'cifar10' or self.helper.config["dataset"] == 'cifar100' or self.helper.config["dataset"] == 'gtsrb':
            layer_name = 'linear'
        elif 'mnist' in self.helper.config["dataset"]:
            layer_name = 'linear'
        elif self.helper.config['dataset'] == 'tiny-imagenet-200':
            layer_name = 'fc'
        else:
            layer_name = 'fc2'
        epsilon = 1e-5
        folderpath = '{0}/foolsgold'.format(self.helper.config["folder_path"])
        # Load params
        his = []
        # 选取全连接层的参数并拉长为长向量 his 
        for i in sampled_participants:
        # for i in range(num):
            history_name = '{0}/history_{1}.pth'.format(folderpath, i)
            his_i_params = torch.load(history_name)
            # print(f"klog his_i_params.keys() ==> ", his_i_params.keys()) 'linear.weight', 'linear.bias'
            for name, data in his_i_params.items():
                if layer_name in name:
                    # print((data.cpu().numpy()).flatten())
                    # 强制拉平 : 如果没有指定轴（axis 参数），np.append(arr, values) 会把 arr 和 values 全部摊平，然后整合成一个一维数组。
                    his = np.append(his, (data.cpu().numpy()).flatten())
        # (参与者数量, 参数维度)
        his = np.reshape(his, (num, -1))
        # print(f"his shape :{his.shape}")
        # print(his)
        print("FoolsGold: Finish loading history updates")
        if np.isnan(his).any():
            print("⚠️ 警告: 检测到历史更新矩阵中包含 NaN！正在将其替换为 0...")
            his = np.nan_to_num(his, nan=0.0)
        # 计算所有参与者之间的两两相似度, 减去单位矩阵 np.eye 是为了排除“自己与自己”的 100% 相似度
        # cs : (num, num) 它存储了客户端 i 的更新向量与客户端 j 的更新向量之间的余弦相似度
        cs = smp.cosine_similarity(his) - np.eye(num)
        # maxcs 计算了每个客户端与其他客户端相似度的最大值
        maxcs = np.max(cs, axis=1) + epsilon
        # 如果客户端 j 的最大相似度大于客户端 i 的最大相似度，说明 j 更可疑。
        # 此时按比例缩小 i 和 j 之间的相似度评估，以防止偶尔与 Sybil 相似的诚实客户端被过度惩罚
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        # Pardoning
        # 用 1 减去最大相似度。相似度越高（接近 1），得到的 wv 就越低（接近 0）
        wv = 1 - (np.max(cs, axis=1))
        # > 1 || < 0 的异常值处理
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        # 归一化与边界处理
        # 保证了在当前这批客户端中，最诚实(干净节点)的那个人，权重会被拉高到 1.0
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        # Logit 函数
        # 当 wv 接近 1 时，\frac{wv}{1-wv} 的值会爆炸式增长，经过 log 后得到一个较大的正数
        #  wv 接近 0 时，\frac{wv}{1-wv} 接近 0，经过 log 后得到一个很小的负数
        # 经验性的偏置 0.5 ?
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # Federated SGD iteration
        positive_weight_clients = [
            participant_id
            for participant_id, weight in zip(sampled_participants, wv)
            if weight > 0
        ]
        zero_weight_clients = [
            participant_id
            for participant_id, weight in zip(sampled_participants, wv)
            if weight <= 0
        ]
        self.agg_recorder.record_stage(
            "foolsgold",
            selected=positive_weight_clients,
            rejected=zero_weight_clients,
            human={
                "layer_name": layer_name,
                "weights_by_client": {
                    participant_id: float(weight)
                    for participant_id, weight in zip(sampled_participants, wv)
                },
            },
            data={
                "history_matrix": his,
                "cosine_similarity": cs,
                "max_cosine_similarity": maxcs,
                "weights": wv,
            },
        )
        print(f"FoolsGold: Accumulation with lr {wv}")
        return wv


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
                # 将该客户端的更新乘以其权重 w，然后累加到容器中
                averaged_weights[name] += client_weight[name] * w
        # 聚合学习率, 控制全局模型向聚合更新方向迈进的步长
        lr = 0.5
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * \
                               (1/len(sampled_participants)) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
        return True



    def rflbat_aggr(self, weight_accumulator_by_client, weight_accumulator, sampled_participants):
        print(f"into rflbat")
        # eps1 = 10, eps2 = 6: 设定两个中位数检测的阈值，用于剔除距离过远的离群点
        eps1 = 10
        eps2 = 6
        # 存放所有客户端参数更新的容器
        # [客户端数量, 总参数量]
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
        # 初始化 PCA，目标是将成千上万维的参数降到 2维 
        pca = PCA(n_components=2)  # instantiate
        # FOCUS 5/2 修改 cifar10_700_May.02_01.00.15_rflbat_True_mia_no-noise_reba 梯度爆炸
        if np.isnan(dataAll).any() or np.isinf(dataAll).any():
            print("Warning: RFLBAT detected NaN/Inf in dataAll. Cleaning before PCA...")
            # 用 0 填充 NaN，用大数填充 Inf
            dataAll = np.nan_to_num(dataAll, nan=0.0, posinf=1e6, neginf=-1e6)
        pca = pca.fit(dataAll)
        # X_dr 形状 (num, 2)
        X_dr = pca.transform(dataAll)
        # Compute sum eu distance
        # 计算每个点到其他所有点的欧几里得距离之和
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i == j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i] - X_dr[j])
            eu_list.append(eu_sum)
        accept = []
        # 这是一个由通过初次筛选的客户端坐标组成的数组
        x1 = []
        for i in range(len(eu_list)):
            # 如果某人的距离总和超过中位数的 10 倍，说明他在 2 维空间里离大家极远，直接视为恶意攻击者丢弃
            if eu_list[i] < eps1 * np.median(eu_list):
                accept.append(i)
                # x1 = np.append(x1, X_dr[i])
                x1.append(X_dr[i])
            else:
                print("RFLBAT: discard update {0}".format(i))
                
            if len(accept) <= 1:
                print("Warning: RFLBAT filtered out too many clients! Accepting all clients for clustering.")
                accept = list(range(len(eu_list)))
                x1 = [X_dr[i] for i in accept]

        print("RFLBAT: the first clients accepted are {0}".format(accept))
        first_selected = [sampled_participants[idx] for idx in accept]
        first_rejected = [
            participant_id
            for idx, participant_id in enumerate(sampled_participants)
            if idx not in accept
        ]
        self.agg_recorder.record_stage(
            "rflbat_first_distance_filter",
            selected=first_selected,
            rejected=first_rejected,
            human={
                "accepted_indices": accept,
                "eps1": eps1,
                "distance_median": float(np.median(eu_list)),
            },
            data={
                "pca_points": X_dr,
                "distance_sums": np.asarray(eu_list),
                "high_dim_updates": np.asarray(dataAll),
            },
        )
        
        x1 = np.array(x1)
        # 这是一个由通过初次筛选的客户端坐标组成的数组
        x1 = np.reshape(x1, (-1, X_dr.shape[1]))
        # FOCUS 修改于 5/2 cifar10_700_May.02_08.05.34_rflbat_True_mia_no-noise_reba 梯度爆炸遗留问题
        if len(x1) <= 2:
            print("Warning: Too few samples for clustering. Setting num_clusters to 1 directly.")
            num_clusters = 1
        else:
            # K_max 不能大于或等于当前的样本数量 len(x1)
            k_max_val = min(10, len(x1) - 1)
            k_max_val = max(2, k_max_val)  # 强制下限为 2，防止 gap_statistics 内部越界
            # if k_max_val <= 1:
            #     num_clusters = 1
            # else:
            #     num_clusters = gap_statistics(x1, num_sampling=5, K_max=k_max_val, n=len(x1))
        # 原始代码
        # num_clusters = gap_statistics(x1, num_sampling=5, K_max=10, n=len(x1))
            try:
                num_clusters = gap_statistics(x1, num_sampling=5, K_max=k_max_val, n=len(x1))
            except (IndexError, ValueError) as e:
                print(f"Warning: gap_statistics failed with error: {e}. Defaulting num_clusters = 1")
                num_clusters = 1
        
        print("RFLBAT: the number of clusters is {0}".format(num_clusters))
        if num_clusters <= 1:
            # 如果聚类数是 1，代表所有人同属一类，不需要调用 KMeans
            predicts = np.zeros(len(x1), dtype=int)
        else:
            """
            n_clusters: 指定要划分的簇（类别）的数量 其数值由前面的 gap_statistics 函数计算得出
            init='k-means++': 它不会随机选择初始中心点，而是通过概率分布让初始中心点彼此尽可能远，从而加速收敛并避免陷入局部最优解。
            n_init='auto': 指定算法以不同的随机初始中心运行多少次，最终选择效果最好的一次。'auto' 会根据 init 参数自动选择次数
            """
            k_means = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto').fit(x1)
            # 从训练好的模型中提取出每个样本的类别标签
            predicts = k_means.labels_
        # k_means = KMeans(n_clusters=num_clusters, init='k-means++').fit(x1)
        # predicts = k_means.labels_

        # select the most suitable cluster
        # 存储每个簇内部的“相似度得分”
        v_med = []
        for i in range(num_clusters):
            # temp 在第一个循环中，它收集属于当前簇 i 的所有客户端的原始高维参数更新
            temp = []
            for j in range(len(predicts)):
                if predicts[j] == i:
                    temp.append(dataAll[accept[j]])
            if len(temp) <= 1:
                v_med.append(1)
                continue
            # 计算 temp 中两两之间的余弦相似度 ->
            # 对相似度矩阵按行求平均。得到每个点与簇内其他点的平均相似度 ->
            # 取上述平均值的中位数，作为该簇的最终得分
            v_med.append(np.median(np.average(smp.cosine_similarity(temp), axis=1)))
        temp = []
        for i in range(len(accept)):
            if predicts[i] == v_med.index(min(v_med)):
                temp.append(accept[i])
        # 只保留这个得分最低的簇里的客户端
        accept = temp
        print("RFLBAT: the second clients accepted are {0}".format(accept))
        second_selected = [sampled_participants[idx] for idx in accept]
        second_rejected = [
            participant_id
            for idx, participant_id in enumerate(sampled_participants)
            if idx not in accept
        ]
        self.agg_recorder.record_stage(
            "rflbat_cluster_filter",
            selected=second_selected,
            rejected=second_rejected,
            human={
                "accepted_indices": accept,
                "num_clusters": int(num_clusters),
                "cluster_scores": v_med,
            },
            data={
                "cluster_labels": np.asarray(predicts),
                "cluster_scores": np.asarray(v_med),
            },
        )
        # compute eu list again to exclude outliers
        temp = []
        for i in accept:
            temp.append(X_dr[i])
        # 仅包含被选中簇成员的 2 维 PCA 坐标
        X_dr = temp
        # 记录选中簇内每个成员到簇内其他成员的欧氏距离之和
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i == j:
                    continue
                # 计算两个 2 维坐标点之间的直线距离
                eu_sum += np.linalg.norm(X_dr[i] - X_dr[j])
            eu_list.append(eu_sum)
        temp = []
        for i in range(len(eu_list)):
            # 超参数（代码前面定义为 6），过滤阈值的倍数
            if eu_list[i] < eps2 * np.median(eu_list):
                temp.append(accept[i])
            else:
                print("RFLBAT: discard update {0}".format(i))
        accept = temp
        chosen_id = []
        # ID 映射与返回
        for i in sampled_participants:
            index = sampled_participants.index(i)
            if index in accept:
                chosen_id.append(i)
        # aggregate
        print("RFLBAT: the final clients accepted are {0}".format(chosen_id))
        self.agg_recorder.record_stage(
            "rflbat_final_distance_filter",
            selected=chosen_id,
            rejected=[participant_id for participant_id in sampled_participants if participant_id not in chosen_id],
            human={
                "accepted_indices": accept,
                "eps2": eps2,
                "distance_median": float(np.median(eu_list)) if len(eu_list) > 0 else None,
            },
            data={"distance_sums": np.asarray(eu_list)},
        )
        return chosen_id

    def deepsight_aggregate_global_model(self, global_model, clients, chosen_ids):
        """

        Args:
            global_model (_type_): _description_
            clients (_type_): _description_
            chosen_ids (_type_): choose user id: [3, 19, 12, 14, 11, 4, 0, 17, 13, 2, 16, 8, 6, 18, 5, 1, 15, 10, 7, 9]
        """
        
        def ensemble_cluster(neups, ddifs, biases):
            """集成聚类。它不是只按一个标准分堆，而是从三个维度（偏置、神经元重要性、输出分布）分别聚类，最后取“交集”
            分别从 偏置矩阵变化量（biases）、神经元重要性能级（neups）、噪声输出分布差异（ddifs） 三个独立的视角分别进行聚类，
            最后将这三个视角的结果融合在一起，重新计算出成员之间的距离矩阵，进行最终的交集裁决
            Args:
                neups (NDArray): 神经元更新强度
                ddifs (NDArray): 输出分布差异
                biases (list): 偏置项的增量
            """

            def sanitize(data, name):
                if not np.isfinite(data).all():
                    print(f"[*] klog Warning: {name} contains NaN/Inf. Sanitizing...")
                    # 替换为 0，防止报错
                    return np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                return data           
            #  biases（转换后）变成了一个二维 NumPy 数组，形状为 (n_client, 偏置的神经元个数)
            # 将存储在 GPU 上的偏置 PyTorch 张量列表统一搬运到 CPU 并打包成标准的 NumPy 矩阵
            biases = np.array([bias.cpu().numpy() for bias in biases])
            # 代表当前参与聚合的客户端总数量
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
            """
            DBSCAN(min_samples=3, metric='cosine')：实例化一个基于密度的空间聚类对象（DBSCAN）。
            min_samples=3 表示一个核心对象最少需要包含 3 个紧挨着的邻居（决定了构成一个阵营的最小人数）；
            metric='cosine' 极其关键，表示它不看更新的绝对绝对大小，而是计算夹角余弦相似度，
            这能有效防止攻击者通过刻意缩放权重大小（如缩减 10 倍）来隐藏自己
            cosine_labels：一维 NumPy 数组，长度为 N
            每个客户端获得一个阵营标签，-1 代表特立独行的噪声点
            """
            cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases).labels_
            # cosine_cluster:[0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1]
            print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neups = sanitize(neups, "neups")
            # 默认使用 metric='euclidean'（欧几里得欧氏距离），去衡量不同客户端更新的神经元绝对能量分布是否相似
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            # neup_cluster:[-1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1  0  0  0]
            print("neup_cluster:{}".format(neup_labels))
            ddifs = sanitize(ddifs, "ddifs")
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            # ddif_cluster:[ 0  0 -1 -1  1  1  1  1  1  1 -1  1  1  1 -1  0  0 -1 -1  1]
            print("ddif_cluster:{}".format(ddif_labels))

            # 用来存放任何两个客户端之间的“阵营认同距离”
            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    # 如果客户端 i 和客户端 j 在偏置聚类中属于同一个阵营，结果为 1，否则为 0
                    # 如果客户端 i 和 j 在三个维度中都被分到了同一个阵营，那么 dists_from_cluster[i, j] = 3 / 3 = 1.0
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j])) / 3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]
            # 一维 NumPy 数组，长度为 N
            # 包含了最终融合后的终审分类结果。那些在这三个维度上共同攻击者，会在这一步被极其精准地聚成同一个“恶意簇”
            ensembled_labels = DBSCAN(min_samples=3, metric='precomputed').fit(dists_from_cluster).labels_
            self.agg_recorder.record_stage(
                "deepsight_ensemble_cluster",
                selected=chosen_ids,
                human={
                    "cosine_cluster": cosine_labels.tolist(),
                    "neup_cluster": neup_labels.tolist(),
                    "ddif_cluster": ddif_labels.tolist(),
                    "ensemble_cluster": ensembled_labels.tolist(),
                },
                data={
                    "biases": biases,
                    "neups": neups,
                    "ddifs": ddifs,
                    "cluster_affinity": dists_from_cluster,
                },
            )

            return ensembled_labels

        # 全连接层的权重矩阵 W
        global_weight = list(global_model.state_dict().values())[-2]
        # 全连接层的偏置向量 $b$
        global_bias = list(global_model.state_dict().values())[-1]
        # 包含每个选定客户端的偏置增量（本地偏置 - 全局偏置）
        biases = [(list(clients[i].state_dict().values())[-1] - global_bias) for i in
                  chosen_ids] 
        # 包含每个选定客户端的本地权重矩阵
        weights = [list(clients[i].state_dict().values())[-2] for i in chosen_ids]

        n_client = len(chosen_ids)
        cosine_similarity_dists = np.array((n_client, n_client))
        # 用于衡量模型的输出神经元（对应分类任务中的每一个类别）在本地训练过程中被改变的剧烈程度（即能量分布）
        # (n_client, 类别数)
        neups = list()
        # 用来量化每个客户端模型中，究竟有多少个神经元发生了“剧烈变动”
        # 一维列表，长度等于 n_client
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in chosen_ids:
            id = chosen_ids.index(i)
            # $$C_{nn} = \sum_{j} (W_{local} - W_{global}) + (b_{local} - b_{global})$$
            # (类别数,)。如果做 MNIST 10 分类，它就是一个包含 10 个数字的向量
            # TOANSWER 这是不是有问题，还减一次 global_bias ?
            C_nn = torch.sum(weights[id] - global_weight, dim=[1]) + biases[id] - global_bias
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2
            
            # 先找出其综合更新的最大能量 C_max = max(C_nn2)
            C_max = torch.max(C_nn2).item()
            # 动态阈值 threshold
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            # 统计超过这个阈值的神经元个数
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)
        # normalize
        # 神经元能级归一化
        # 将每个客户端的神经元能量向量 neup，除以所有客户端总能量之和 sC_nn2
        neups = np.array([(neup / sC_nn2).cpu().numpy() for neup in neups])
        # n_exceeds:[7, 5, 7, 3, 4, 7, 5, 3, 4, 5, 7, 7, 3, 4, 3, 3, 2, 3, 4, 3]
        print("n_exceeds:{}".format(n_exceeds))

        # 生成随机的、无业务意义的噪声数据，作为“探测针”输入到模型中，用以强行触发隐藏的后门
        rand_input = torch.randn((256, 3, 32, 32)).to(self.helper.device)
        # if self.helper.config["dataset"] == 'mnist':
        if "mnist" in self.helper.config["dataset"]:
            rand_input = torch.randn((20, 1, 28, 28)).to(self.helper.device)
        elif self.helper.config["dataset"] == 'tiny-imagenet-200':
            rand_input = torch.randn((64, 3, 224, 224)).to(self.helper.device)
        # 通过让全局模型预测一组“随机噪声”，计算出干净模型对无意义输入时的“盲测输出概率分布”，作为后续揪出后门攻击者的基准底线
        # global_model(rand_input) / torch.softmax(global_model(rand_input), dim=1) -> (batch_size, num_class)
        # global_ddif -> (num_class, ) 它代表了全局模型对这批随机噪声输入的平均预测概率分布
        """
        对于一个干净的、正常的全局模型来说，由于输入的 rand_input 是毫无意义的纯随机噪声，模型不应该在任何特定类别上表现出强烈的偏好。
        因此，global_ddif 算出来的概率分布理论上应该非常接近均匀分布
        如果是诚实客户端：它的模型也是干净的，对噪声的预测也是均匀的。
        均匀分布除以均匀分布（global_ddif），结果接近 [1, 1, 1, 1...]，非常平稳
        """
        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        # 本地模型对噪声的预测除以全局模型的预测, 形状为 (客户端数, 类别数)
        """
        衡量本地模型在哪些类别上产生了异常的“兴奋”。如果某个客户端是后门攻击者，
        即使输入的是噪声，其模型也可能强烈地将输出指向“后门目标类”，导致该比值在某个维度上激增
        """
        
        client_ddifs = [torch.mean(torch.softmax(clients[i](rand_input), dim=1), dim=0) / global_ddif
                        for i in chosen_ids]
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])

        # use n_exceed to label
        # 将存储各客户端超阈值神经元数量的 Python 列表转为 NumPy 一维数组
        # 计算这个数组的中位数 / 2
        classification_boundary = np.median(np.array(n_exceeds)) / 2

        # 由整数 0 或 1 组成的一维列表，长度等于当前客户端数量
        # 若某位置为 1，代表该客户端在第一轮预检中被高度怀疑为后门分子；0 则代表暂时安全
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        # identified_mals:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        # ensemble clusters:[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        print("ensemble clusters:{}".format(clusters))
        # np.unique() 返回数组中所有去重后的唯一元素，并从小到大排序 
        # 盘点出本轮联邦学习中一共存在多少个不同的阵营（簇）
        cluster_ids = np.unique(clusters)

        # 用于收集所有被判定为恶意的、需要被删除的簇 ID
        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            # 计算当前正在审查的这个簇一共有多少个成员（即簇的规模）
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            # cluser size:20 n_mal:0
            print("cluser size:{} n_mal:{}".format(cluster_size, n_mal))
            # 如果该阵营内部的内鬼比例达到了 $\frac{1}{3}$ 或以上，
            # 判定该阵营沦陷，将该阵营的 ID（cluster_id）加入黑名单 deleted_cluster_ids
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        # 一维整数列表，初始化为本轮参与的全体客户端
        # 
        temp_chosen_ids = copy.deepcopy(chosen_ids)
        final_chosen = copy.deepcopy(chosen_ids)
        # 从后往前（倒序）遍历当前客户端的局部索引
        # 在下面的循环体中，我们会执行 del 物理删除操作。如果正序（从 0 开始）删除，
        # 删除前面的元素会导致后面的元素索引整体往前移，产生“索引错位”或越界崩溃
        for i in range(len(chosen_ids) - 1, -1, -1):
            if clusters[i] in deleted_cluster_ids:
                del final_chosen[i]

        if len(final_chosen) == 0:
            final_chosen = temp_chosen_ids
        self.agg_recorder.record_stage(
            "deepsight_final_filter",
            selected=final_chosen,
            rejected=[client_id for client_id in chosen_ids if client_id not in final_chosen],
            human={
                "n_exceeds": n_exceeds,
                "classification_boundary": float(classification_boundary),
                "identified_mals": identified_mals,
                "clusters": clusters.tolist(),
                "deleted_cluster_ids": deleted_cluster_ids,
            },
            data={
                "neups": neups,
                "client_ddifs": client_ddifs,
                "clusters": clusters,
                "biases": biases,
            },
        )

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
    
    # FOCUS RFA 
    
    def rfa_models(self, global_model, weight_accumulator_by_client, sampled_participants):
          if len(weight_accumulator_by_client) != len(sampled_participants):
              raise ValueError(
                  "weight_accumulator_by_client must follow sampled_participants order: "
                  f"{len(weight_accumulator_by_client)} updates vs "
                  f"{len(sampled_participants)} participants"
              )

          keys = self._rfa_update_keys(global_model, weight_accumulator_by_client[0])
          rfa_inputs = []

          for participant_id, single_wa in zip(sampled_participants, weight_accumulator_by_client):
              flat_update = self._flatten_single_wa(single_wa, keys)
              client_weight = 1.0
              rfa_inputs.append((client_weight, flat_update))

          maxiter = self.helper.config.get('rfa_maxiter', 4)

          result = aggregate_client_updates(
              rfa_inputs,
              maxiter=maxiter,
              reject_nonfinite=True,
          )

          self._apply_flat_update_to_global_model(
              global_model,
              result.point,
              keys,
          )

          self.agg_recorder.record_stage(
              "rfa",
              selected=sampled_participants,
              human={
                  "num_participants": len(sampled_participants),
                  "num_update_keys": len(keys),
                  "oracle_calls": result.num_oracle_calls,
                  "update_norm": float(np.linalg.norm(result.point)),
                  "maxiter": maxiter,
              },
              data={
                  "keys": keys,
                  "geometric_median_update": result.point,
                  "logs": result.logs,
              },
          )

          print(
              f"RFA aggregation: participants={len(sampled_participants)}, "
              f"oracle_calls={result.num_oracle_calls}, "
              f"update_norm={np.linalg.norm(result.point):.6f}"
          )

          return global_model

    def _rfa_update_keys(self, global_model, single_wa):
          global_state = global_model.state_dict()
          keys = []

          for name, tensor in global_state.items():
              if name not in single_wa:
                  continue

              update_tensor = single_wa[name]

              if not isinstance(update_tensor, torch.Tensor):
                  update_tensor = torch.as_tensor(update_tensor)

              if not torch.is_floating_point(tensor):
                  continue

              if tensor.shape != update_tensor.shape:
                  raise ValueError(
                      f"Shape mismatch for {name}: "
                      f"global shape={tuple(tensor.shape)}, "
                      f"update shape={tuple(update_tensor.shape)}"
                  )

              keys.append(name)

          if not keys:
              raise ValueError("No valid floating-point update keys found for RFA aggregation")

          return keys

    def _flatten_single_wa(self, single_wa, keys):
          chunks = []

          for name in keys:
              if name not in single_wa:
                  raise KeyError(f"single_wa missing key: {name}")

              update_tensor = single_wa[name]

              if not isinstance(update_tensor, torch.Tensor):
                  update_tensor = torch.as_tensor(update_tensor)

              chunks.append(
                  update_tensor.detach()
                  .cpu()
                  .reshape(-1)
                  .numpy()
                  .astype(np.float64, copy=False)
              )

          return np.concatenate(chunks)

    def _apply_flat_update_to_global_model(self, global_model, flat_update, keys):
          global_state = global_model.state_dict()
          offset = 0

          with torch.no_grad():
              for name in keys:
                  tensor = global_state[name]
                  numel = tensor.numel()

                  update_slice = flat_update[offset: offset + numel]

                  if update_slice.shape[0] != numel:
                      raise ValueError(
                          f"RFA flat update size mismatch at {name}: "
                          f"expected {numel}, got {update_slice.shape[0]}"
                      )

                  update_tensor = torch.from_numpy(update_slice).to(
                      device=tensor.device,
                      dtype=tensor.dtype,
                  ).view_as(tensor)

                  tensor.add_(update_tensor)

                  offset += numel

          if offset != len(flat_update):
              raise ValueError(
                  f"Unused values in flat_update: used {offset}, total {len(flat_update)}"
              )

          global_model.load_state_dict(global_state)
          return global_model