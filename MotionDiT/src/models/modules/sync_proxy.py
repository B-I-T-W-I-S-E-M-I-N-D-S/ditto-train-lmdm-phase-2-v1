"""
SyncProxyNet — lightweight motion-space sync proxy network.

Maps (mouth_kp_sequence, audio_feature_sequence) over sliding temporal
windows to a shared embedding space and computes a cosine-similarity loss.
Used as a differentiable audio-motion alignment signal during DiT fine-tuning,
avoiding the need to render video frames every step.

Architecture: two small MLP encoders (3 linear layers + LayerNorm + GELU)
that map flattened temporal windows of motion and audio to a common embedding.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, embed_dim: int) -> nn.Sequential:
    """3-layer MLP encoder with LayerNorm gating."""
    return nn.Sequential(
        nn.Linear(in_dim, embed_dim * 2),
        nn.LayerNorm(embed_dim * 2),
        nn.GELU(),
        nn.Linear(embed_dim * 2, embed_dim),
        nn.LayerNorm(embed_dim),
    )


class SyncProxyNet(nn.Module):
    """
    Sync proxy network for one (motion, audio/latent) modality pair.

    Args:
        motion_in_dim : number of mouth-keypoint dims (e.g. len(mouth_dims))
        audio_in_dim  : audio/latent feature dim per frame
        embed_dim     : shared embedding size (default 128)
        window        : temporal window width in frames (default 5)
    """

    def __init__(
        self,
        motion_in_dim: int,
        audio_in_dim: int,
        embed_dim: int = 128,
        window: int = 5,
    ):
        super().__init__()
        self.window = window
        self.motion_encoder = _make_mlp(motion_in_dim * window, embed_dim)
        self.audio_encoder  = _make_mlp(audio_in_dim  * window, embed_dim)

    def forward(
        self,
        motion_seq: torch.Tensor,   # [B, L, motion_in_dim]
        audio_seq:  torch.Tensor,   # [B, L, audio_in_dim]
    ) -> torch.Tensor:
        """
        Returns a scalar loss = 1 - mean cosine_similarity.
        Returns 0 tensor (no gradient) if L < window.
        """
        B, L, _ = motion_seq.shape
        w = self.window

        if L < w:
            return motion_seq.new_zeros(1).squeeze()

        # Non-overlapping windows over the sequence length
        n_windows = L // w
        usable_L  = n_windows * w

        # [B, n_windows, w, dim] → [B*n_windows, w*dim]
        m = motion_seq[:, :usable_L].reshape(B, n_windows, w, -1)
        a = audio_seq[ :, :usable_L].reshape(B, n_windows, w, -1)

        m_flat = m.reshape(B * n_windows, -1)  # [B*n_windows, w*motion_in_dim]
        a_flat = a.reshape(B * n_windows, -1)  # [B*n_windows, w*audio_in_dim]

        m_emb = self.motion_encoder(m_flat)   # [B*n_windows, embed_dim]
        a_emb = self.audio_encoder(a_flat)    # [B*n_windows, embed_dim]

        # cosine similarity ∈ [-1, 1]; loss = 1 - sim so minimum is 0 when aligned
        loss = 1.0 - F.cosine_similarity(m_emb, a_emb, dim=-1).mean()
        return loss
