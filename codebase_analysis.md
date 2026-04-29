# Ditto DiT Training Codebase — Deep Analysis

## Repository Layout

```
ditto-talkinghead-train-try-to-finetune/
├── MotionDiT/
│   ├── train.py                          ← entry-point (tyro CLI → Trainer)
│   └── src/
│       ├── trainers/trainer.py           ← training loop, optimiser, checkpoint saving
│       ├── models/
│       │   ├── LMDM.py                   ← model+diffusion wrapper (MotionDecoder + MotionDiffusion)
│       │   └── modules/
│       │       ├── model.py              ← MotionDecoder (DiT architecture)
│       │       ├── diffusion.py          ← MotionDiffusion (noise schedule, loss, sampling)
│       │       ├── adan.py               ← Adan optimizer
│       │       ├── rotary_embedding_torch.py
│       │       └── utils.py              ← positional encodings, beta schedules
│       ├── datasets/s2_dataset_v2.py     ← Stage2Dataset (data loading, conditioning)
│       ├── options/option.py             ← TrainOptions dataclass (all hyperparams)
│       └── utils/utils.py                ← DictAverageMeter, json/pkl helpers
├── prepare_data/                         ← offline data preparation scripts
└── example/                              ← example data manifests
```

---

## Area 1 — Loss Computation (Training Step)

### Entry chain
```
trainer.py: _train_one_step()
  → LMDM.diffusion(x, cond_frame, cond, t_override=None)      # diffusion is MotionDiffusion
    → MotionDiffusion.forward()   [diffusion.py:380]
      → MotionDiffusion.loss()    [diffusion.py:372]
        → MotionDiffusion.p_losses()  [diffusion.py:291]
```

### `p_losses()` — the core loss function (diffusion.py L291–370)

```python
def p_losses(self, x_start, cond_frame, cond, t):
    noise = torch.randn_like(x_start)
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    # single model forward pass → x_recon shape [B, L, 265]
    x_recon = self.model(x_noisy, cond_frame, cond, t, cond_drop_prob=self.cond_drop_prob)

    # predict_epsilon=False → target = x_start (predict motion directly)
    model_out = x_recon
    target = x_start  # (or noise if predict_epsilon=True)

    # _get_pva_loss inner function: per-group P/V/A losses
    loss_dict = _get_pva_loss(model_out, target, self.part_w_dict, ...)
    total_loss = sum(loss_dict.values())
    return total_loss, loss_dict
```

### `_get_pva_loss` — per-group P/V/A loss (diffusion.py L305–363)

**`part_w_dict` (the adaptive weight dict)** — default when not overridden via JSON:
```python
{
  "scale": [0,   1,   1],   # dim 0        (1-D scale)
  "pitch": [1,   67,  1],   # dims 1-66    (22 keypoints × 3)
  "yaw":   [67,  133, 1],   # dims 67-132
  "roll":  [133, 199, 1],   # dims 133-198
  "t":     [199, 202, 1],   # dims 199-201 (translation, 3-D)
  "exp":   [202, 265, 1],   # dims 202-264 (expression deformation δ, 63-D)
}
```

**Important:** `[s, e, w]` = `[start_dim, end_dim, weight_scalar]`.  
The **mouth keypoints live inside `"exp"` (dims 202–264)** — they are **not separated out** as their own group. They receive whatever weight `"exp"` receives (default `w=1`), diluted across all 63 expression dims.

For each group, `_part_loss` computes:
- **`_p_loss`** — MSE on raw motion values (position loss, L_d)
- **`_v_loss`** — MSE on first-order differences (velocity, L_t component)
- **`_a_loss`** — MSE on second-order differences (acceleration, L_t component)
- **`_l_loss`** — MSE of first predicted frame vs. `cond_frame` (L_ini, only if `use_last_frame_loss=True`)
- **`_r_loss`** — L1 regulariser on magnitudes (only if `use_reg_loss=True`)

`dim_ws` — an optional per-dimension weight array `[dim]` loaded from a `.npy` file; multiplied element-wise with the group weight. Enables dimension-level fine-grained re-weighting within a group.

### Key observation on the adaptive weight mechanism

The **adaptive weight mechanism** is **NOT in the training loop** (no per-epoch update).  
The `part_w_dict` weight `w` is a **static scalar** per group, set once at initialization from a JSON file (`opt.part_w_dict_json`). There is no softmax update, no epoch-level dynamic adjustment in the current code.  

**The paper describes** an adaptive loss weight mechanism, but this codebase implements it as a **static, configurable per-group weight** (passed via `--part_w_dict_json`). The `dim_ws` array (via `--dim_ws_npy`) provides dimension-level static weights.

---

## Area 2 — m_ref Construction (ICS Conditioning)

### Where it is built: `s2_dataset_v2.py` — `getitem()` (L194–293)

```python
# kp_cond = m_ref (the ICS reference motion frame)
if self.use_last_frame:
    kp_cond = v_mtn[f_idx - 1]             # frame immediately before the clip
else:
    kp_cond = v_mtn[random.randint(0, len(v_mtn) - 1)]   # random frame from the same video
```

**`kp_cond` is the raw full 265-D motion vector** from the training video, including:
- `kp_cond[0]`       — scale  
- `kp_cond[1:67]`    — pitch (21 KPs × 3)  
- `kp_cond[67:133]`  — yaw  
- `kp_cond[133:199]` — roll  
- `kp_cond[199:202]` — translation  
- `kp_cond[202:265]` — expression δ including mouth keypoints ← **leakage point**

**How `kp_cond` enters the model** — `trainer.py: _train_one_step()`:
```python
x = data_dict["kp_seq"]          # [B, L, 265]
cond_frame = data_dict["kp_cond"] # [B, 265]    ← m_ref
cond = data_dict["aud_cond"]      # [B, L, aud_dim]
loss, loss_dict = self.LMDM.diffusion(x, cond_frame, cond, t_override=None)
```

**Inside `MotionDecoder.forward()` (model.py L337–386)**:
```python
# ICS concatenation — cond_frame expanded over all L frames then cat'd channel-wise
x = torch.cat([x, cond_frame.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
# nfeats * 2 = 265 * 2 = 530 → projected to 512 (latent_dim)
x = self.input_projection(x)   # Linear(530, 512)
```

The mouth dimensions of `cond_frame[202:265]` directly enter the model at the **input projection layer**, concatenated channel-wise. No masking or neutralisation happens anywhere.

---

## Area 3 — Adaptive Loss Weight Mechanism

As noted above: **the current implementation uses static weights, not dynamic per-epoch adaptation**.

The `part_w_dict` JSON format is:
```json
{
  "scale": [0, 1, 1.0],
  "pitch": [1, 67, 1.0],
  "yaw":   [67, 133, 1.0],
  "roll":  [133, 199, 1.0],
  "t":     [199, 202, 1.0],
  "exp":   [202, 265, 1.0]
}
```

The weight `w` is the third element of each tuple, used here:
```python
# diffusion.py L326-328
_p_loss = self.loss_fn(p1, p2, reduction="none") * dim_w   # dim_w = dim_ws[s:e] * w (or just w)
_v_loss = self.loss_fn(v1, v2, reduction="none") * dim_w
_a_loss = self.loss_fn(a1, a2, reduction="none") * dim_w
```

**There is no floor enforcement, no softmax normalisation, and no per-epoch update.** To raise mouth group priority, you currently would need to:
1. Create a new `part_w_dict_json` with a higher `w` for `"exp"`, OR
2. Add a dedicated `"mouth"` group with its own `[s, e, w]` slice pointing at the mouth dims within 202–264.

The `dim_ws` array (shape `[265]`) gives the finest-grained control — you can assign higher per-dimension weights to specific dims inside the expression group.

---

## Area 4 — DiT Parameter Groups (for selective fine-tuning)

### `MotionDecoder` module tree (model.py)

| Attribute | Type | Role | Notes |
|---|---|---|---|
| `rotary` | `RotaryEmbedding` | Positional encoding | Shared across ECS + decoder |
| `abs_pos_encoding` | `nn.Identity` (rotary=True) | Absolute pos enc | Identity when rotary is used |
| `time_mlp` | `Sequential(SinusoidalPosEmb, Linear, Mish)` | Timestep embedding | Diffusion timestep → embedding |
| `to_time_cond` | `Sequential(Linear)` | t → FiLM scalar | FiLM conditioning branch |
| `to_time_tokens` | `Sequential(Linear, Rearrange)` | t → 2 cross-attn tokens | Cross-attn conditioning branch |
| `null_cond_embed` | `nn.Parameter [1, L, 512]` | CFG null embedding | Guidance dropout |
| `null_cond_hidden` | `nn.Parameter [1, 512]` | CFG null hidden | Guidance dropout |
| `norm_cond` | `LayerNorm` | Norm before cross-attn | Normalises concatenated cond tokens |
| `input_projection` | `Linear(530, 512)` | ICS input projection | Projects [x ∥ cond_frame] → latent |
| `cond_encoder` | `Sequential` of 2× `TransformerEncoderLayer` | **ECS encoder blocks** | Processes audio+eye+emo+identity tokens |
| `cond_projection` | `Linear(cond_feat_dim, 512)` | Cond projection | Projects raw audio/cond features → latent |
| `non_attn_cond_projection` | `Sequential(LayerNorm, Linear, SiLU, Linear)` | FiLM hidden | Mean-pooled cond → FiLM scalar offset |
| `seqTransDecoder` | `DecoderLayerStack` of 8× `FiLMTransformerDecoderLayer` | **Main decoder blocks** | Core denoising transformer |
| `final_layer` | `Linear(512, 265)` | Output projection | Latent → motion vector |

### `FiLMTransformerDecoderLayer` internals (per block, model.py L106–222)

Each of the **8 main blocks** contains:

| Submodule | Type | Role |
|---|---|---|
| `self_attn` | `MultiheadAttention(512, 8)` | **Self-attention** → motion prior |
| `multihead_attn` | `MultiheadAttention(512, 8)` | **Cross-attention** → audio binding |
| `linear1`, `linear2` | `Linear(512, 1024)`, `Linear(1024, 512)` | **MLP / feedforward** → motion prior |
| `norm1`, `norm2`, `norm3` | `LayerNorm` | Layer norms |
| `dropout1`, `dropout2`, `dropout3` | `Dropout` | Dropout |
| `film1` | `DenseFiLM(512)` → `Sequential(Mish, Linear(512, 1024))` | **FiLM after self-attn** |
| `film2` | `DenseFiLM(512)` | **FiLM after cross-attn** |
| `film3` | `DenseFiLM(512)` | **FiLM after MLP** |

### `TransformerEncoderLayer` (ECS encoder, per block)

| Submodule | Type | Role |
|---|---|---|
| `self_attn` | `MultiheadAttention(512, 8)` | Audio/cond self-attention |
| `linear1`, `linear2` | `Linear(512, 1024)`, `Linear(1024, 512)` | MLP |
| `norm1`, `norm2` | `LayerNorm` | |

### Current optimizer setup (trainer.py L126)
```python
optim = Adan(self.LMDM.model.parameters(), lr=opt.lr, weight_decay=0.02)
```
**All parameters are trained together.** No parameter grouping, no freezing, no separate learning rates.

### Parameter groups for your proposed selective fine-tuning

| Group | Module path(s) | Proposed state |
|---|---|---|
| **Cross-attention** | `seqTransDecoder.stack[i].multihead_attn.*` (i=0..7) | **Trainable** |
| **FiLM layers** | `seqTransDecoder.stack[i].film1/2/3.*` (i=0..7) | **Trainable** |
| **ECS encoder** | `cond_encoder.*`, `cond_projection.*`, `non_attn_cond_projection.*` | **Trainable** |
| **ICS input proj** | `input_projection.*` | **Trainable** |
| **Time embedding** | `time_mlp.*`, `to_time_cond.*`, `to_time_tokens.*` | Trainable (small, important for FiLM) |
| **Self-attention** | `seqTransDecoder.stack[i].self_attn.*` (i=0..7) | **Frozen** |
| **MLP/FF** | `seqTransDecoder.stack[i].linear1/2.*` (i=0..7) | **Frozen** |
| **Layer norms** | `seqTransDecoder.stack[i].norm1/2/3.*` | Consider freezing |
| **Final layer** | `final_layer.*` | Trainable (output head) |
| **Rotary emb** | `rotary.*` | Frozen |

---

## Area 5 — Training Config Structure

**File:** `MotionDiT/src/options/option.py` — a `@dataclass` called `TrainOptions`.

**Entry point:** `MotionDiT/train.py` uses `tyro.cli(TrainOptions)` → all fields become CLI arguments automatically.

**No separate fine-tuning config exists.** The current config is the single flat dataclass. There is no YAML/JSON config file system — everything is CLI args.

### Current key hyperparameters

| Field | Default | Purpose |
|---|---|---|
| `lr` | `1e-4` | Learning rate |
| `epochs` | `1000` | Training epochs |
| `batch_size` | `512` | Batch size |
| `checkpoint` | `""` | Path to `.pt` checkpoint to load |
| `part_w_dict_json` | `""` | Per-group loss weight JSON |
| `dim_ws_npy` | `""` | Per-dimension loss weight array |
| `use_last_frame_loss` | `False` | L_ini (first-frame guidance loss) |
| `use_reg_loss` | `False` | Magnitude regulariser |
| `motion_feat_dim` | `265` | Total motion dimension |
| `audio_feat_dim` | `1103` | 1024 (HuBERT) + 63 (sc) + 8 (emo) + 2 (eye_open) + 6 (eye_ball) |
| `seq_frames` | `80` | 3.2 s × 25 fps |
| `use_emo` | `False` | Include emotion conditioning |
| `use_eye_open` | `False` | Include eye open state |
| `use_eye_ball` | `False` | Include eyeball direction |
| `use_sc` | `False` | Include source canonical keypoints (identity) |
| `use_last_frame` | `False` | Use temporally contiguous reference frame as m_ref |
| `use_cond_end` | `False` | Use clip-end frame as additional cond |

**Checkpoint loading (LMDM.py L63–66):**
```python
if checkpoint:
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint[\"model_state_dict\"])   # strict=True by default
```

No `strict=False` option currently exists.

---

## Summary Map to Your 5 Areas

| Your Area | Where in code | Key files |
|---|---|---|
| **Area 1** — Loss computation | `MotionDiffusion.p_losses()` → `_get_pva_loss()` → `_part_loss()` | `diffusion.py` L291–370 |
| **Area 2** — m_ref construction | `Stage2Dataset.getitem()` → `kp_cond` assignment | `s2_dataset_v2.py` L225–228 |
| **Area 3** — Adaptive weight mechanism | `MotionDiffusion.__init__()` → `part_w_dict` + `dim_ws` (static) | `diffusion.py` L113–124; config JSON via `--part_w_dict_json` |
| **Area 4** — DiT parameters | `MotionDecoder` full module tree | `model.py`; optimizer in `trainer.py` L126 |
| **Area 5** — Training config | `TrainOptions` dataclass + `tyro.cli` | `option.py`; `train.py` |

---

## Critical Findings & Implications

> [!IMPORTANT]
> **The "adaptive" weight mechanism described in the paper is NOT adaptive in this code.** It is a static per-group weight set at init. There is no per-epoch softmax update. The `dim_ws` array is the most granular control available.

> [!IMPORTANT]
> **m_ref (kp_cond) carries the full 265-D raw motion vector** including mouth expression dims 202–264. No neutralisation exists. The reference frame is either (a) the temporally preceding frame or (b) a random frame from the same video sequence.

> [!IMPORTANT]
> **The `exp` group (dims 202–264, 63 dims) is a single group containing ALL expression deformations** — eyebrow, cheek, jaw, lip corners, lip thin/thick, etc. Mouth-specific dims are not identified or separated. To target mouth dims specifically, you need to know which dims within 202–264 correspond to mouth keypoints and create a sub-group or use `dim_ws`.

> [!NOTE]
> **No sync loss exists anywhere.** `MotionDiffusion` has no SyncNet reference, no cosine-similarity audio-motion loss, and no render path during training. The only supervision is reconstruction + optional velocity/acceleration.

> [!NOTE]
> **Checkpoint loading uses strict=True by default.** Adding any new parameters (even loss head networks) requires changing this.

> [!NOTE]
> **The optimizer covers all parameters of `self.LMDM.model` (the MotionDecoder).** There is no selective parameter group setup, no layer-specific LR multipliers, no freeze logic.
