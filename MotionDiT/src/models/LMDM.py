# Latent Motion Diffusion Model
import torch

from .modules.model import MotionDecoder
from .modules.diffusion import MotionDiffusion


FPS = 25
SEQ_SEC = 3.2


class LMDM:
    def __init__(
        self,
        motion_feat_dim=265,
        audio_feat_dim=1024+35,
        seq_frames=int(SEQ_SEC * FPS),
        part_w_dict=None,   # only for train
        checkpoint='',
        device='cuda',
        use_last_frame_loss=False,    # only for train
        use_reg_loss=False,    # only for train
        dim_ws=None,    # only for train
        strict_checkpoint=True,    # strict weight loading; set False for fine-tune
        # sync proxy loss params (Change 3)
        use_sync_proxy_loss=False,
        lambda_sync_proxy=0.1,
        sync_proxy_audio_dim=1024,
        sync_proxy_embed_dim=128,
        sync_proxy_ckpt='',
        mouth_dims=None,    # list[int] of absolute dim indices for mouth keypoints
        # intermediate ECS sync params (Change 4)
        use_intermediate_sync=False,
        lambda_sync_intermediate=0.05,
    ):
        self.motion_feat_dim = motion_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.seq_frames = seq_frames
        self.device = device

        model = MotionDecoder(
            nfeats=motion_feat_dim,
            seq_len=seq_frames,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=audio_feat_dim,
        )

        diffusion = MotionDiffusion(
            model,
            horizon=seq_frames,
            repr_dim=motion_feat_dim,
            n_timestep=1000,
            schedule="cosine",
            loss_type="l2",
            clip_denoised=True,
            predict_epsilon=False,
            guidance_weight=2,
            use_p2=False,
            cond_drop_prob=0.2,
            part_w_dict=part_w_dict,
            use_last_frame_loss=use_last_frame_loss,
            use_reg_loss=use_reg_loss,
            dim_ws=dim_ws,
            use_sync_proxy_loss=use_sync_proxy_loss,
            lambda_sync_proxy=lambda_sync_proxy,
            sync_proxy_audio_dim=sync_proxy_audio_dim,
            sync_proxy_embed_dim=sync_proxy_embed_dim,
            sync_proxy_ckpt=sync_proxy_ckpt,
            mouth_dims=mouth_dims,
            use_intermediate_sync=use_intermediate_sync,
            lambda_sync_intermediate=lambda_sync_intermediate,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        if checkpoint:
            print(f'[CKPT] Loading checkpoint: {checkpoint}  (strict={strict_checkpoint})')
            ckpt_data = torch.load(checkpoint, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(
                ckpt_data["model_state_dict"], strict=strict_checkpoint
            )
            if missing_keys:
                print(f'[CKPT] Missing keys  ({len(missing_keys)}): {missing_keys}')
            if unexpected_keys:
                print(f'[CKPT] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}')
            if not missing_keys and not unexpected_keys:
                print('[CKPT] All keys matched successfully.')

        diffusion = diffusion.to(device)

        self.model = model
        self.diffusion = diffusion

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def use_accelerator(self, accelerator):
        self.model = accelerator.prepare(self.model)
        self.diffusion = self.diffusion.to(accelerator.device)

    @torch.no_grad()
    def _run_diffusion_render_sample(self, kp_cond, aud_cond, noise=None):
        """
        kp_cond: [b, kp_dim], tensor
        aud_cond: [b, L, aud_dim], tensor
        pred_kp_seq: [b, L, kp_dim], tensor
        """
        device = self.device

        render_count = 1
        seq_frames = self.seq_frames
        motion_feat_dim = self.motion_feat_dim

        shape = (render_count, seq_frames, motion_feat_dim)
        cond_frame = kp_cond.to(device)
        cond = aud_cond.to(device)

        pred_kp_seq = self.diffusion.render_sample(
            shape,
            cond_frame,
            cond,
            normalizer=None,
            epoch=None,
            render_out=None,
            last_half=None,
            mode="normal",
            noise=noise,
        )
        return pred_kp_seq

