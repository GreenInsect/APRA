import torch
import numpy as np
from collections import OrderedDict
import copy

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
    APRA: Adaptive Progressive Robust Aggregation

    Combines and improves upon four defense strategies:
      - DeepSight: Multi-view feature extraction (NEUP/DDIF)
      - RFLBAT: PCA dimensionality reduction + clustering
      - FoolsGold: Cosine-similarity based trust weighting
      - Clip: Gradient norm clipping

    Four stages:
      1. Multi-dimensional feature extraction (flat weights + NEUP/DDIF)
      2. Adaptive statistical pre-filtering with epoch-decaying MAD threshold
      3. Hierarchical clustering with silhouette-score auto-k selection
      4. Trust-weighted aggregation with confidence-gated adaptive clipping
    """

    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        config = self.helper.config
        device = self.helper.device

        n_clients = len(sampled_participants)
        if n_clients < 2:
            return self._simple_average(global_model, weight_accumulator_by_client, sampled_participants)

        # Stage 1: Multi-dimensional feature extraction
        features, update_norms = self._extract_features(
            global_model, client_models, weight_accumulator_by_client,
            sampled_participants
        )

        # Stage 2: Adaptive statistical pre-filtering
        mask = self._adaptive_mad_filter(features, update_norms, epoch)
        n_accepted = mask.sum()
        print(f"APRA Stage1: {n_accepted}/{n_clients} clients pass MAD filter")

        if n_accepted < 2:
            filtered_ids = sampled_participants.copy()
            filtered_features = features.copy()
            filtered_indices = list(range(n_clients))
        else:
            filtered_indices = np.where(mask)[0].tolist()
            filtered_ids = [sampled_participants[i] for i in filtered_indices]
            filtered_features = features[mask]

        # Stage 3: Hierarchical clustering + auto cluster selection
        if len(filtered_indices) >= 3:
            cluster_labels, selected_cluster = self._hierarchical_cluster(filtered_features)
            print(f"APRA Stage2: {len(set(cluster_labels))} clusters, selected cluster {selected_cluster} "
                  f"({sum(cluster_labels == selected_cluster)} clients)")

            cluster_mask = cluster_labels == selected_cluster
            trusted_global_indices = [filtered_indices[i] for i in range(len(filtered_indices)) if cluster_mask[i]]
            if len(trusted_global_indices) < 2:
                trusted_global_indices = filtered_indices
                trust_features = filtered_features
            else:
                trust_features = filtered_features[cluster_mask]
        else:
            trusted_global_indices = filtered_indices
            trust_features = filtered_features

        chosen_ids = [sampled_participants[i] for i in trusted_global_indices]
        print(f"APRA Final: {len(chosen_ids)} clients selected for aggregation")

        # Stage 4: Trust-weighted aggregation with adaptive clipping
        trust_weights = self._compute_trust_weights(trust_features)

        self._weighted_average_with_clip(
            global_model,
            weight_accumulator_by_client,
            chosen_ids,
            sampled_participants,
            trust_weights,
            epoch,
        )

        return True

    # ===================== Stage 1: Feature Extraction =====================

    def _extract_features(self, global_model, client_models, weight_accumulator_by_client,
                          sampled_participants):
        config = self.helper.config
        dataset = config['dataset']
        layer_name = get_layer_name_for_dataset(dataset)
        device = self.helper.device

        flat_updates = []
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

        # Optionally add NEUP/DDIF features (DeepSight-style)
        if config['apra_use_neup_ddif'] and client_models is not None:
            neup_ddif = self._extract_neup_ddif(global_model, client_models, sampled_participants)
            if neup_ddif is not None and len(neup_ddif) == len(sampled_participants):
                flat_features = np.hstack([flat_features, neup_ddif])

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

    def _extract_neup_ddif(self, global_model, client_models, sampled_participants):
        device = self.helper.device
        config = self.helper.config
        dataset = config['dataset']

        try:
            global_state = list(global_model.state_dict().values())
            if len(global_state) < 2:
                return None
            global_weight = global_state[-2]
            global_bias = global_state[-1]

            neup_features = []
            for client_id in sampled_participants:
                if client_id >= len(client_models):
                    neup_features.append([0, 0])
                    continue
                client_state = list(client_models[client_id].state_dict().values())
                if len(client_state) < 2:
                    neup_features.append([0, 0])
                    continue
                client_bias = client_state[-1]
                neup_features.append(
                    np.array(client_bias.cpu().numpy() - global_bias.cpu().numpy()).flatten()
                )

            neup_features = np.array(neup_features)

            # DDIF
            if dataset == 'mnist':
                rand_input = torch.randn((8, 1, 28, 28)).to(device)
            elif dataset == 'tiny-imagenet-200':
                rand_input = torch.randn((16, 3, 224, 224)).to(device)
            else:
                rand_input = torch.randn((32, 3, 32, 32)).to(device)

            global_output = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0) + 1e-8
            ddif_features = []
            for client_id in sampled_participants:
                if client_id >= len(client_models):
                    ddif_features.append([0])
                    continue
                client_output = torch.mean(torch.softmax(client_models[client_id](rand_input), dim=1), dim=0)
                ddif = (client_output / global_output).cpu().detach().numpy()
                ddif_features.append(ddif)

            ddif_features = np.array(ddif_features)

            # Concatenate
            combined = np.hstack([neup_features, ddif_features])
            return combined
        except Exception:
            return None

    # ===================== Stage 2: Adaptive MAD Filter =====================

    def _adaptive_mad_filter(self, features, update_norms, epoch):
        config = self.helper.config
        n_clients = len(update_norms)

        if n_clients < 3:
            return np.ones(n_clients, dtype=bool)

        # Adaptive k: decays from k_init to 2.0 over epochs
        k_init = config['apra_k_init']
        k_decay = config['apra_k_decay']
        k = max(2.0, k_init * np.exp(-k_decay * epoch))

        # MAD-based outlier detection on L2 norms
        median_norm = np.median(update_norms)
        mad = np.median(np.abs(update_norms - median_norm))
        mad = max(mad, 1e-8)

        modified_z_scores = 0.6745 * (update_norms - median_norm) / mad
        mask = np.abs(modified_z_scores) <= k

        # Safety: at least keep half
        if mask.sum() < max(2, n_clients // 2):
            sorted_indices = np.argsort(np.abs(modified_z_scores))
            keep_count = max(2, n_clients // 2)
            mask[:] = False
            mask[sorted_indices[:keep_count]] = True

        print(f"APRA MAD: median={median_norm:.4f} mad={mad:.4f} k(epoch={epoch})={k:.2f} "
              f"kept={mask.sum()}/{n_clients}")

        return mask

    # ===================== Stage 3: Hierarchical Clustering =====================

    def _hierarchical_cluster(self, features):
        n = len(features)
        if n < 3:
            return np.zeros(n, dtype=int), 0

        # Try k from 2 to n//2, pick best silhouette score
        max_k = min(n - 1, max(2, n // 2))
        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            clustering = AgglomerativeClustering(
                n_clusters=k, metric='euclidean', linkage='ward'
            )
            labels = clustering.fit_predict(features)
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(features, labels)
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best_k = k

        # Final clustering with best_k
        final_clustering = AgglomerativeClustering(
            n_clusters=best_k, metric='euclidean', linkage='ward'
        )
        labels = final_clustering.fit_predict(features)

        # Select best cluster: largest + highest internal cosine similarity
        cluster_scores = {}
        for c in np.unique(labels):
            indices = np.where(labels == c)[0]
            cluster_size = len(indices)
            if cluster_size <= 1:
                cluster_scores[c] = -1
                continue
            cluster_data = features[indices]
            internal_sim = np.mean(sk_cosine_similarity(cluster_data))
            cluster_scores[c] = cluster_size * internal_sim

        selected_cluster = max(cluster_scores, key=cluster_scores.get)

        return labels, selected_cluster

    # ===================== Stage 4: Trust-Weighted + Clip =====================

    def _compute_trust_weights(self, features):
        n = len(features)
        if n <= 1:
            return np.ones(n) / n

        # Pairwise cosine similarity
        cs = sk_cosine_similarity(features)
        epsilon = 1e-5

        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # Pardoning
        wv = 1 - np.max(cs, axis=1)
        wv = np.clip(wv, 0, 1)

        # Rescale
        max_wv = np.max(wv)
        if max_wv > 0:
            wv = wv / max_wv
        wv[wv == 1] = 0.99

        # Logit transform
        wv = np.log(wv / (1 - wv) + epsilon) + 0.5
        wv = np.clip(wv, 0, 1)

        # Normalize
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
            return

        # Compute adaptive clip factors from trust weights
        min_trust = np.min(trust_weights) if len(trust_weights) > 0 else 1.0 / n
        max_trust = np.max(trust_weights) if len(trust_weights) > 0 else 1.0 / n
        clip_factors = np.zeros(n)
        for i in range(n):
            # Low trust → tighter clip
            w_i = trust_weights[i]
            clip_factors[i] = base_clip * (1.0 + (max_trust - w_i) / (max_trust + 1e-8))

        # Apply clipping to each selected client's update
        for idx_in_list, global_idx in enumerate(chosen_ids):
            local_idx = sampled_participants.index(global_idx)
            update = weight_accumulator_by_client[local_idx]
            cf = clip_factors[idx_in_list]

            for key in update:
                if 'num_batches_tracked' in key:
                    continue
                data = update[key]
                l2 = torch.norm(data, p=2)
                data.div_(max(1.0, l2.item() / cf))

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

    def _simple_average(self, global_model, weight_accumulator_by_client, sampled_participants):
        avg = FedAvgAggregator(self.helper)
        return avg.aggregate(global_model, None, weight_accumulator_by_client,
                            None, sampled_participants, 0)
