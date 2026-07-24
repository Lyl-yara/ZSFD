import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from TF_attributes_fs import compute_attributes

def get_feature_extractor(fs):
    return compute_attributes(fs)


def extract_features(patch, ATTR_FN, SELECTED_FEATURES):
    """
    Calculate features specified in SELECTED_FEATURES using predefined ATTR_FN
    Supports dynamic SELECTED_FEATURES configuration
    """
    if len(patch) == 0:
        return np.zeros(len(SELECTED_FEATURES), dtype=np.float32)

    feat = []
    for key in SELECTED_FEATURES:
        try:
            if key in ATTR_FN and ATTR_FN[key] is not None:
                value = ATTR_FN[key](patch)
                if isinstance(value, (np.ndarray, list, tuple)):
                    value = float(np.mean(value))
                else:
                    value = float(value)
                feat.append(value)
            else:
                print(f"Warning: Feature '{key}' not found in ATTR_FN. Using 0.0")
                feat.append(0.0)
        except Exception as e:
            print(f"Error computing feature '{key}': {e}")
            feat.append(0.0)

    feat_array = np.array(feat, dtype=np.float32)
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1e8, neginf=-1e8)
    return feat_array


def generate_semantic_vector(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    oc = parts[0]
    sev = parts[1]
    fault = parts[2]

    oc50 = 1.0 if oc == "50" else 0.0
    oc75 = 1.0 if oc == "75" else 0.0
    oc100 = 1.0 if oc == "100" else 0.0
    f_inner = 1.0 if fault == "inner" else 0.0
    f_outer = 1.0 if fault == "outer" else 0.0
    f_pump = 1.0 if fault == "pump" else 0.0
    f_healthy = 1.0 if fault == "healthy" else 0.0
    s_light = 1.0 if sev == "light" else 0.0
    s_middle = 1.0 if sev == "middle" else 0.0
    s_serious = 1.0 if sev == "serious" else 0.0

    return np.array([
        oc50, oc75, oc100,
        f_inner, f_outer, f_pump, f_healthy,
        s_light, s_middle, s_serious
    ], dtype=np.float32)


def split_semantic_groups(sem_vec):
    g1 = sem_vec[..., 0:3]
    g2 = sem_vec[..., 3:7]
    g3 = sem_vec[..., 7:10]
    return [g1, g2, g3]


def build_seq_dataset(file_list, file_cls_map, config, scaler=None, is_train=True, cache_path=None, max_per_file=None, SELECTED_FEATURES=None):
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Load dataset from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (torch.tensor(data["features"], dtype=torch.float32),
                torch.tensor(data["semantics"], dtype=torch.float32),
                torch.tensor(data["labels"], dtype=torch.long),
                data["mean"], data["std"])

    print("No cache found, start feature extraction...")
    ATTR_FN = get_feature_extractor(config['fs'])

    all_feat_seqs = []
    all_cls_labels = []
    all_sem_vecs = []
    raw_sub_feature_pool = []

    N_seq = config['seq_patch_num']
    patch_len = config['patch_size']
    sub_len = config['subpatch_size']
    step = config['step']

    for file_path in file_list:
        sem_vec_np, cls_id = file_cls_map[file_path]
        df = pd.read_csv(file_path, header=None)
        sig = df.iloc[:, 0].values.astype(np.float32)

        single_patch_feats = []
        start = 0
        count = 0

        while start + patch_len <= len(sig) and count < max_per_file:
            patch = sig[start: start + patch_len]
            sub_feat_seq = []
            for s in range(0, patch_len, sub_len):
                sub_seg = patch[s:s + sub_len]
                if len(sub_seg) < sub_len:
                    sub_seg = np.pad(sub_seg, (0, sub_len - len(sub_seg)), 'constant')
                feat = extract_features(sub_seg, ATTR_FN, SELECTED_FEATURES)
                sub_feat_seq.append(feat)
                if is_train:
                    raw_sub_feature_pool.append(feat)
            single_patch_feats.append(np.stack(sub_feat_seq, axis=0))
            count += 1
            start += step

        single_patch_feats = np.array(single_patch_feats)
        total_patch_num = len(single_patch_feats)

        for s_idx in range(total_patch_num - N_seq + 1):
            seq_data = single_patch_feats[s_idx: s_idx + N_seq]
            all_feat_seqs.append(seq_data)
            all_cls_labels.append(cls_id)
            all_sem_vecs.append(sem_vec_np)

    all_feat_seqs = np.array(all_feat_seqs, dtype=np.float32)
    all_sem_vecs = np.array(all_sem_vecs)

    if len(all_feat_seqs) == 0:
        mean = np.zeros(len(SELECTED_FEATURES))
        std = np.ones(len(SELECTED_FEATURES))
    else:
        if is_train:
            raw_sub_feature_pool = np.array(raw_sub_feature_pool)
            mean = raw_sub_feature_pool.mean(axis=0)
            std = raw_sub_feature_pool.std(axis=0) + 1e-8
        else:
            mean, std = scaler
        all_feat_seqs = (all_feat_seqs - mean) / std

    feat_tensor = torch.tensor(all_feat_seqs, dtype=torch.float32)
    label_tensor = torch.tensor(all_cls_labels, dtype=torch.long)
    sem_tensor = torch.tensor(all_sem_vecs, dtype=torch.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=all_feat_seqs,
            semantics=all_sem_vecs,
            labels=np.array(all_cls_labels),
            mean=mean,
            std=std
        )
        print(f"Dataset cache saved to: {cache_path}")

    return feat_tensor, sem_tensor, label_tensor, mean, std


def split_sample(features, semantics, labels, train_num=300, val_num=60):
    cls_index_dict = defaultdict(list)
    for idx, lab in enumerate(labels.numpy()):
        cls_index_dict[lab].append(idx)

    train_idx_list = []
    val_idx_list = []
    for cls_id, all_idx in cls_index_dict.items():
        if len(all_idx) >= train_num + val_num:
            tr_idx = all_idx[:train_num]
            va_idx = all_idx[train_num:train_num + val_num]
        else:
            raise ValueError(f"Class {cls_id} does not have enough samples! Required: {train_num+val_num}, available: {len(all_idx)}")
        train_idx_list.extend(tr_idx)
        val_idx_list.extend(va_idx)

    train_feats = features[train_idx_list]
    train_sem = semantics[train_idx_list]
    train_labels = labels[train_idx_list]

    val_feats = features[val_idx_list]
    val_sem = semantics[val_idx_list]
    val_labels = labels[val_idx_list]

    print(f"Seen fixed split: train samples={len(train_idx_list)}, val samples={len(val_idx_list)}")
    return (train_feats, train_sem, train_labels), (val_feats, val_sem, val_labels)
