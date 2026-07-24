import torch
from torch import nn
import torch.nn.functional as F


class DualRoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rope_rotate(self, x: torch.Tensor, pos: torch.Tensor):
        device = x.device
        theta = pos.unsqueeze(-1) * self.inv_freq
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        return torch.cat([x_rot1, x_rot2], dim=-1)

    def forward(self, single_patch_x: torch.Tensor, patch_n):
        """
        Args:
            single_patch_x: (B, I, D) sequence of subpatches within a single patch
            patch_n: (B,) global temporal index n of this patch
        Return:
            Z_tilde: (B, D) patch representation after dual-scale rotation
        """
        B, I, D = single_patch_x.shape
        device = single_patch_x.device
        # Step1: Local rotation for subpatch index i
        pos_i = torch.arange(I, device=device).float()
        X_sub_rot = self._rope_rotate(single_patch_x, pos_i)
        # Step2: Aggregate subpatches into patch representation
        Z_n = torch.mean(X_sub_rot, dim=1)
        # Step3: Global rotation for patch index n
        pos_n = patch_n.float().to(device)
        Z_tilde = self._rope_rotate(Z_n, pos_n)
        return Z_tilde


class TemporalEncoder(nn.Module):
    """ Shallow transformer encoding"""

    def __init__(
            self,
            input_dim,
            embed_dim=128,
            nhead=4,
            num_layers=2,
            ff_scale=4,
            dropout=0.1,
            seq_patch_num=8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_patch_num = seq_patch_num
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.dual_rope = DualRoPE(dim=embed_dim, base=10000.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * ff_scale,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_seq):
        """
        Args:
            x_seq: (B, N, I, feat_dim)
        Return:
            v: (B, embed_dim) globally pooled feature embedding
        """
        B, N, I, feat_dim = x_seq.shape
        patch_emb_list = []

        for n in range(N):
            single_patch = x_seq[:, n, :, :]  # (B, I, feat_dim)
            x_proj = self.input_proj(single_patch)  # (B, I, D)

            patch_n_tensor = torch.full((B,), n, device=x_seq.device, dtype=torch.float)
            z_tilde = self.dual_rope(x_proj, patch_n_tensor)  # (B, D)
            patch_emb_list.append(z_tilde)

        full_patch_seq = torch.stack(patch_emb_list, dim=1)  # (B, N, D)
        seq_out = self.transformer_encoder(full_patch_seq)  # Transformer for long-range dependency modeling
        seq_out = seq_out.transpose(1, 2)  # (B, D, N)
        final_emb = self.pool(seq_out).squeeze(-1)  # (B, D)
        return final_emb


class ZeroShotNet(nn.Module):
    def __init__(self, feat_dim, sem_dim, embed_dim=128, seq_patch_num=8,
                 num_groups=4, protos_per_group=3):
        super().__init__()
        self.encoder = TemporalEncoder(feat_dim, embed_dim=embed_dim,
                                       seq_patch_num=seq_patch_num)
        self.embed_dim = embed_dim
        self.sem_dim = sem_dim

        self.sem_embed = nn.Linear(sem_dim, embed_dim, bias=False)  # W_γ
        self.G = nn.Linear(embed_dim, embed_dim, bias=True)  # G(·)

        # Prototype Groups
        self.num_groups = num_groups
        self.grouped_prototypes = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(protos_per_group, embed_dim), dim=1))
            for _ in range(num_groups)
        ])

        # Gating function g(·)
        self.gating = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_groups)
        )

    def forward(self, x_seq):
        return self.encoder(x_seq)

    def get_visual_embedding(self, x_seq):
        return F.normalize(self.encoder(x_seq), dim=-1)

    def get_clean_semantic(self, sem_vec):
        return F.normalize(self.sem_embed(sem_vec), dim=-1)

    def get_consensus_mu(self, v):
        alpha = F.softmax(self.gating(v), dim=1)  # α = softmax(g(v))
        mu = torch.zeros((v.shape[0], self.embed_dim), device=v.device)
        for g_idx, proto_group in enumerate(self.grouped_prototypes):
            h_g = proto_group.mean(dim=0)
            mu += alpha[:, g_idx:g_idx + 1] * h_g
        return mu

    def generate_semantic(self, visual_emb):
        mu = self.get_consensus_mu(visual_emb)
        v_gamma = self.G(visual_emb)  # G(v)
        combined = v_gamma + mu
        v_tilde = F.normalize(combined, dim=-1)

        return v_tilde

    def get_prototype_reconstruction(self, emb):
        mu = self.get_consensus_mu(emb)
        prototypes_flat = torch.cat([gp for gp in self.grouped_prototypes], dim=0)
        return mu, prototypes_flat
