from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass, field
import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))
DEFAULT_EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments', 's2')


class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __repr__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)
    

@dataclass(repr=False)  # use repr from PrintableConfig
class TrainOptions(PrintableConfig):
    ########## experiment ##########
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR    # experiment_dir
    experiment_name: str = ""   # experiment_name

    ########## dataset ##########
    data_list_json: str = ""    # train data list: [[kps_npy, aud_npy, frame_num],]
    data_preload: bool = False  # data_preload flag
    data_preload_pkl: str = ""  # save to data_preload_pkl
    reprepare_idx_map: bool = False    # reprepare_idx_map flag for dataset
    data_cache: bool = False    # data_cache flag

    mtn_mean_var_npy: str = ""    # for mtn norm

    motion_feat_start: int = 0  # motion feat start dim
    motion_feat_offset_dim_se: tuple[int, ...] = ()    # cal offset dim range for part D-S

    use_emo: bool = False    # use_emo flag
    use_eye_open: bool = False    # use_eye_open flag
    use_eye_ball: bool = False    # use_eye_ball flag
    use_sc: bool = False    # use source canonical keypoints flag
    use_last_frame: bool = False    # use last frame as cond frame flag
    use_lmk: bool = False    # use mediapipe lmk as cond flag
    use_cond_end: bool = False    # use clip start and end frame as cond flag

    dataset_version: str = "v2"    # dataset version: [v1, v2]

    ########## model ##########
    motion_feat_dim: int = 265          # motion_feat_dim
    audio_feat_dim: int = 1103     # audio_feat_dim (1024 + 63 + 8 + 2 + 6)
    seq_frames: int = int(3.2 * 25)     # clip length

    ########## train ##########
    use_accelerate: bool = True         # use_accelerate flag for multi gpu
    epochs: int = 1000      # epochs
    batch_size: int = 512   # batch_size
    num_workers: int = 0    # num_workers
    lr: float = 1e-4        # lr
    part_w_dict_json: str = ""    # part loss weights dict json
    use_last_frame_loss: bool = False    # use_last_frame_loss flag
    use_reg_loss: bool = False    # use_reg_loss flag
    dim_ws_npy: str = ""    # dim_ws npy

    checkpoint: str = ""        # checkpoint for load

    save_ckpt_freq: int = 50    # save ckpt freq (epoch)

    ########## fine-tuning: checkpoint ##########
    strict_checkpoint: bool = True  # strict=False allows loading ckpt when new modules exist

    ########## fine-tuning: selective freezing ##########
    finetune_mode: bool = False    # enable selective parameter freezing
    # names of MotionDecoder submodules to FREEZE in finetune_mode
    # default empty = use built-in fine-tune freeze policy
    frozen_submodules: tuple[str, ...] = ()
    finetune_lr: float = 0.0    # if > 0, override lr for fine-tune; else use lr
    grad_clip_max_norm: float = 0.0    # gradient clipping max norm (0 = disabled)

    ########## fine-tuning: mouth neutralization ##########
    neutralize_mouth_ref: bool = False    # zero-out mouth dims in kp_cond (m_ref)
    mouth_dims: tuple[int, ...] = ()      # absolute dim indices for mouth keypoints
    mouth_neutral_npy: str = ""           # path to mtn_mean_var npy for neutral values (empty=zeros)

    ########## fine-tuning: mouth group weight ##########
    mouth_group_weight: float = 1.0    # dim_ws multiplier for mouth dims (>1 = upweight)

    ########## fine-tuning: sync proxy loss (Change 3) ##########
    use_sync_proxy_loss: bool = False       # enable motion-space sync proxy loss
    lambda_sync_proxy: float = 0.1         # weight for final-output sync proxy loss
    sync_proxy_audio_dim: int = 1024       # HuBERT feature dim (first N dims of aud_cond)
    sync_proxy_embed_dim: int = 128        # proxy network embedding dimension
    sync_proxy_ckpt: str = ""              # path to pretrained proxy checkpoint (optional)

    ########## fine-tuning: intermediate sync loss (Change 4) ##########
    use_intermediate_sync: bool = False     # enable ECS intermediate sync guidance
    lambda_sync_intermediate: float = 0.05 # weight for intermediate sync loss


def check_train_opt(opt: TrainOptions):
    assert opt.experiment_dir, opt.experiment_dir
    assert opt.experiment_name, opt.experiment_name
    assert opt.data_list_json, opt.data_list_json
