import os
import glob
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from Data_split import build_seq_dataset, split_sample, generate_semantic_vector
from Model import ZeroShotNet
from LFSI_Qwen import get_best_feature_combination

# Hyperparameter configuration
CONFIG = {
    'fs': 20000,
    'patch_size': 2048,
    'subpatch_size': 128,
    'step': 512,
    'seq_patch_num': 8,
    'epochs': 20,
    'batch_size': 32,
    'lr': 1e-4,
    'embed_dim': 128,
    'protos_per_group': 5,
    'seen_max_per_file': 350,
    'unseen_max_per_file': 307,
    'train_num_per_class': 300,
    'val_num_per_class': 50,

    'seen_folder': r"./Data/Seen",
    'unseen_folder': r"./Data/Unseen",
    'result_folder': r"./Data/Result",
    'cache_root': r"./dataset_cache",
}

# Invoke LLM to infer optimal feature combination
SELECTED_FEATURES = get_best_feature_combination(CONFIG['seen_folder'])
print(SELECTED_FEATURES)

# If LLM weight files are not downloaded, comment out the LLM code and use the precomputed feature combination below
# SELECTED_FEATURES = ['absolute_mean', 'cwt_variance', 'spectral_spread', 'spectral_entropy',
#                      'mean', 'spectral_rolloff', 'rms', 'std_dev', 'hilbert_envelope_rms', 'peak_count']


#  Loss Functions
def cosine_loss(pred, target):
    """Semantic consistency loss """
    pred_n = F.normalize(pred, dim=1)
    target_n = F.normalize(target, dim=1)
    return 1 - torch.mean(torch.sum(pred_n * target_n, dim=1))


def reconstruction_loss(v, mu):
    """Reconstruction loss """
    v_n = F.normalize(v, dim=1)
    mu_n = F.normalize(mu, dim=1)
    return 1 - torch.mean(torch.sum(v_n * mu_n, dim=1))


def prototype_diversity_loss(prototypes):
    """Diversity loss """
    prot = F.normalize(prototypes, dim=1)
    sim = torch.mm(prot, prot.t())
    K = prot.shape[0]
    mask = 1.0 - torch.eye(K, device=prot.device)
    return torch.sum((sim * mask) ** 2) / (K * (K - 1) + 1e-8)


#  Validation & test functions
def validate_val(model, val_loader, unseen_raw_sem, device):
    """Evaluate on validation set for saving optimal model weights"""
    model.eval()
    correct = total = 0
    candidate_sem = model.get_clean_semantic(unseen_raw_sem).to(device)
    with torch.no_grad():
        for x_seq, _, labels in val_loader:
            x_seq, labels = x_seq.to(device), labels.to(device)
            v_emb = model.get_visual_embedding(x_seq)
            s_pred = model.generate_semantic(v_emb)
            sims = torch.mm(F.normalize(s_pred, dim=1), candidate_sem.t())
            preds = torch.argmax(sims, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def validate_test(model, test_loader, unseen_raw_sem, device):
    """Evaluate on test set"""
    model.eval()
    correct = total = 0
    candidate_sem = model.get_clean_semantic(unseen_raw_sem).to(device)
    with torch.no_grad():
        for x_seq, _, labels in test_loader:
            x_seq, labels = x_seq.to(device), labels.to(device)
            v_emb = model.get_visual_embedding(x_seq)
            s_pred = model.generate_semantic(v_emb)
            sims = torch.mm(F.normalize(s_pred, dim=1), candidate_sem.t())
            preds = torch.argmax(sims, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seen_folder = CONFIG['seen_folder']
    unseen_folder = CONFIG['unseen_folder']
    result_folder = CONFIG['result_folder']
    cache_root = CONFIG['cache_root']

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    seen_files = sorted(glob.glob(os.path.join(seen_folder, "*.csv")))
    unseen_files = sorted(glob.glob(os.path.join(unseen_folder, "*.csv")))

    # Build class mapping
    seen_cls_map = {fname: (torch.tensor(generate_semantic_vector(fname), dtype=torch.float32), i)
                    for i, fname in enumerate(seen_files)}

    unseen_cls_map = {fname: (torch.tensor(generate_semantic_vector(fname), dtype=torch.float32), i)
                      for i, fname in enumerate(unseen_files)}

    # Seen class semantics
    seen_sem_list = [generate_semantic_vector(f) for f in seen_files]
    seen_raw_tensor = torch.tensor(seen_sem_list, dtype=torch.float32)
    seen_raw_tensor = F.normalize(seen_raw_tensor, dim=-1).to(device)

    # Unseen class semantics
    unseen_sem_list = [generate_semantic_vector(f) for f in unseen_files]
    unseen_raw_tensor = torch.tensor(unseen_sem_list, dtype=torch.float32)
    unseen_raw_tensor = F.normalize(unseen_raw_tensor, dim=-1).to(device)

    # Construct seen dataset
    seen_full_cache = os.path.join(cache_root, "seen_full_dataset.npz")
    seen_all_feats, seen_all_sem, seen_all_labels, train_mean, train_std = build_seq_dataset(
        seen_files, seen_cls_map, CONFIG, is_train=True,
        cache_path=seen_full_cache, max_per_file=CONFIG['seen_max_per_file'],
        SELECTED_FEATURES=SELECTED_FEATURES
    )

    # Split seen samples into train / validation within each class
    (train_feats, train_sem, train_labels), (val_feats, val_sem, val_labels) = split_sample(
        seen_all_feats, seen_all_sem, seen_all_labels,
        train_num=CONFIG['train_num_per_class'],
        val_num=CONFIG['val_num_per_class']
    )

    train_dataset = TensorDataset(train_feats, train_sem, train_labels)
    val_dataset = TensorDataset(val_feats, val_sem, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Unseen test set
    unseen_full_cache = os.path.join(cache_root, "unseen_full_dataset.npz")
    test_feats, test_sem, test_labels, _, _ = build_seq_dataset(
        unseen_files, unseen_cls_map, CONFIG, scaler=(train_mean, train_std),
        is_train=False, cache_path=unseen_full_cache,
        max_per_file=CONFIG['unseen_max_per_file'], SELECTED_FEATURES=SELECTED_FEATURES
    )

    test_dataset = TensorDataset(test_feats, test_sem, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Initialize model
    feat_dim = train_feats.shape[-1]
    sem_dim = unseen_raw_tensor.shape[-1]

    model = ZeroShotNet(
        feat_dim=feat_dim,
        sem_dim=sem_dim,
        embed_dim=CONFIG['embed_dim'],
        seq_patch_num=CONFIG['seq_patch_num'],
        protos_per_group=CONFIG['protos_per_group']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    # Loss weight value
    w_cos = 1.0
    w_recon = 0.3
    w_div = 0.5
    best_val_acc = 0.0

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = total_l_sem = total_l_div = total_l_recon = 0.0

        for x_seq, y, _ in train_loader:
            x_seq, y = x_seq.to(device), y.to(device)
            optimizer.zero_grad()

            v_emb = model.get_visual_embedding(x_seq)
            s_gt = model.get_clean_semantic(y)
            s_pred = model.generate_semantic(v_emb)

            l_semantic = cosine_loss(s_pred, s_gt)
            recon_mu, prototypes = model.get_prototype_reconstruction(v_emb)
            l_recon = reconstruction_loss(v_emb, recon_mu)
            l_div = prototype_diversity_loss(prototypes)

            loss = w_cos * l_semantic + w_recon * l_recon + w_div * l_div

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l_sem += l_semantic.item()
            total_l_div += l_div.item()
            total_l_recon += l_recon.item()

        current_val_acc = validate_val(model, val_loader, seen_raw_tensor, device)
        avg_loss = total_loss / len(train_loader)

        print(f"{epoch:03d}/{CONFIG['epochs']}    Loss: {avg_loss:.4f}    Val Acc: {current_val_acc * 100:6.2f}%")

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), os.path.join(result_folder, 'model_weight.pth'))

    # Test
    best_model = ZeroShotNet(
        feat_dim, sem_dim, CONFIG['embed_dim'],
        seq_patch_num=CONFIG['seq_patch_num'],
        protos_per_group=CONFIG['protos_per_group']
    ).to(device)
    best_model.load_state_dict(torch.load(os.path.join(result_folder, 'model_weight.pth')))
    test_acc = validate_test(best_model, test_loader, unseen_raw_tensor, device)

    print(f"Final Result")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.6f}%")
    print(f"Test Accuracy: {test_acc * 100:.6f}%")


if __name__ == "__main__":
    main()
