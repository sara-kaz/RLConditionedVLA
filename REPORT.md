# VLA Robot Learning — Project Report
**Author:** Sara Aly
**Started:** 2026-03-03
**Last Updated:** 2026-03-04

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Status](#2-current-status)
3. [Architecture — In Detail](#3-architecture--in-detail)
   - 3.1 System-Level Data Flow
   - 3.2 Model Architecture
   - 3.3 ActionRewardHistoryEncoder
   - 3.4 Token Sequence Layout
   - 3.5 Model Parameter Count
4. [How the Models Are Combined — Step by Step](#4-how-the-models-are-combined--step-by-step)
   - 4.1 The Problem: Three Incompatible Representations
   - 4.2 Step 1 — Encode Vision with CLIP
   - 4.3 Step 2 — Encode Language with CLIP
   - 4.4 Step 3 — Encode (Action, Reward) History
   - 4.5 Step 4 — Project to a Shared Space
   - 4.6 Step 5 — Assemble the Token Sequence
   - 4.7 Step 6 — Fuse with a Causal Transformer
   - 4.8 Step 7 — Read Out and Classify
   - 4.9 Why Not Simpler Fusion?
   - 4.10 Bug Found and Fixed: CLS Token Position
5. [Training Pipeline](#5-training-pipeline)
   - 5.1 Phase 1: Behavioral Cloning
   - 5.2 Phase 2: Online RL Fine-Tuning
   - 5.3 Evaluation
6. [What Has Been Built](#6-what-has-been-built)
   - 6.1 File-by-File Breakdown
   - 6.2 Key Design Decisions
7. [Environment & Data](#7-environment--data)
8. [Experiments & Results](#8-experiments--results)
9. [Issues & Fixes](#9-issues--fixes)
10. [Next Steps](#10-next-steps)
11. [References](#11-references)
12. [Change Log](#12-change-log)

---

## 1. Project Overview

### Problem Statement

Standard VLA (Vision-Language-Action) models predict robot actions from video and language alone. They ignore a critical signal: **what the robot just did and whether it worked**. This project adds that RL feedback loop explicitly.

### Goal

Build a VLA model that predicts the next **logical discrete action** for a robot by conditioning on three input streams simultaneously:

```
  VIDEO          ─┐
  LANGUAGE       ─┼──► [RLConditionedVLA] ──► next action
  (action,reward)─┘
  history
```

### Inputs & Outputs

| Input | Type | Description |
|---|---|---|
| Video frames | `(B, T, 3, H, W)` | Last T camera frames |
| Language instruction | `(B, 77)` | Natural-language task goal, CLIP-tokenized |
| Action history | `(B, H)` int64 | Indices of last H actions taken |
| Reward history | `(B, H)` float32 | Scalar reward received after each past action |
| **Output** | `(B, num_actions)` | Logits over discrete action space |

### Core Novelty

Existing VLAs (RT-2, Octo, OpenVLA) treat inference as a pure perception task. This model treats **(state, action, reward)** history as **first-class input tokens** — injecting RL memory directly into the forward pass, inspired by Decision Transformer. This lets the model reason: *"I tried action 3, got a negative reward, so I should try something different."*

### Target Deployment

```
  Simulation (MuJoCo / Gymnasium)
          │
          ▼  sim-to-real transfer
  Real robot hardware
```

---

## 2. Current Status

### What Is Complete

| Component | Status | File |
|---|---|---|
| Core model architecture | **Done** | `models/vla_model.py` |
| Action-reward history encoder | **Done** | `models/vla_model.py` |
| Trajectory dataset (sliding window) | **Done** | `data/trajectory_dataset.py` |
| Synthetic data generator | **Done** | `data/trajectory_dataset.py` |
| Phase 1: Behavioral cloning trainer | **Done** | `training/sft_trainer.py` |
| Phase 2: REINFORCE + baseline trainer | **Done** | `training/rl_trainer.py` |
| Value head for RL baseline | **Done** | `training/rl_trainer.py` |
| KL penalty vs BC checkpoint | **Done** | `training/rl_trainer.py` |
| Dummy environment (no Gym needed) | **Done** | `envs/sim_env.py` |
| Gym environment wrapper | **Done** | `envs/sim_env.py` |
| Real robot interface stub | **Done** | `envs/sim_env.py` |
| Evaluation script | **Done** | `evaluation/evaluate.py` |
| Config file | **Done** | `configs/config.yaml` |
| PDF report generator | **Done** | `generate_report.py` |

### What Is Pending

| Component | Priority | Notes |
|---|---|---|
| Real demonstration data | High | Need `.pkl` episodes from a real or scripted policy |
| Concrete Gym environment | High | Choose: MiniGrid, FetchReach, or FrankaKitchen |
| Run SFT training (Phase 1) | High | Need data first |
| Run RL training (Phase 2) | High | Need Phase 1 checkpoint |
| TensorBoard logging | Medium | Add to both trainers |
| PPO upgrade | Medium | More stable than REINFORCE for long episodes |
| Domain randomization | Medium | Required for sim-to-real |
| Real robot eval (`RealEnv`) | Low | After sim results are satisfactory |

---

## 3. Architecture — In Detail

### 3.1 System-Level Data Flow

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         INFERENCE TIME                                   │
  │                                                                          │
  │   Camera                                                                 │
  │   ┌──────┐  frame_t-2                                                    │
  │   │  🎥  │  frame_t-1   ──────────────────────────────────┐              │
  │   └──────┘  frame_t                                        │              │
  │                                                            ▼              │
  │   Human / Task                                    ┌─────────────────┐    │
  │   ┌──────────────────────┐                        │                 │    │
  │   │ "pick up the red     │ ──────────────────────►│  RLConditioned  │    │
  │   │  cube and place it   │                        │      VLA        │───►│ action_t
  │   │  on the shelf"       │                        │                 │    │
  │   └──────────────────────┘                        └─────────────────┘    │
  │                                                            ▲              │
  │   History Buffer                                           │              │
  │   ┌────────────────────────────────────────┐              │              │
  │   │ t-4: action=3, reward=-0.1             │              │              │
  │   │ t-3: action=3, reward=-0.1             │─────────────►│              │
  │   │ t-2: action=7, reward=+0.5             │              │              │
  │   │ t-1: action=1, reward=+1.0             │              │              │
  │   └────────────────────────────────────────┘              │              │
  │                                                            │              │
  │   action_t ──────────────────────► Execute on robot ──────┘              │
  │                        │                                                  │
  │                        └──► reward_t ──► append to History Buffer        │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Model Architecture

```
  VIDEO FRAMES (B, T, 3, 224, 224)
  │
  ├──► Flatten to (B*T, 3, 224, 224)
  │
  ├──► CLIP ViT-B/32 Image Encoder        [FROZEN]
  │         output: (B*T, 512)
  │
  ├──► Reshape to (B, T, 512)
  │
  └──► vis_proj  Linear(512→256)
            output: vis_tokens  (B, T, 256)
                                                    ┌─────────────────────────────┐
  LANGUAGE (B, 77) CLIP tokens                      │   CAUSAL FUSION TRANSFORMER  │
  │                                                  │                             │
  ├──► CLIP Text Encoder                [FROZEN]     │  Input sequence:            │
  │         output: (B, 512)                         │                             │
  │                                                  │  [CLS] ← learnable param    │
  └──► lang_proj  Linear(512→256)                    │   │                         │
            output: lang_token  (B, 1, 256)          │  [lang_token]               │
                                                 ───►│   │                         │
  ACTION HISTORY (B, H) int64                        │  [frame_t-T]                │
  │                                                  │   │                         │
  ├──► action_embed  Embedding(num_a+1, 256)         │  [frame_t-T+1]              │
  │         output: (B, H, 256)                      │   │  ...                    │
  │                                                  │  [frame_t-1]                │
  REWARD HISTORY (B, H) float32                      │   │                         │
  │                                                  │  [hist_t-H]                 │
  └──► reward_proj  Linear(1→256)                    │   │  ...                    │
            output: (B, H, 256)                      │  [hist_t-1]                 │
                   │                                  │                             │
                   ├──► cat + Linear(512→256)         │  Causal mask: each token   │
                   │         (fuse a+r per step)      │  attends only to earlier   │
                   │                                  │  positions (no future leak) │
                   └──► + positional embedding        │                             │
                              │                       │  6 layers, 8 heads          │
                              └──► hist_tokens        │  FFN dim: 1024              │
                                   (B, H, 256)        │  Pre-LayerNorm              │
                                              ───────►│                             │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                              CLS token output
                                                                (B, 256)
                                                                     │
                                                              ACTION HEAD
                                                         ┌───────────────────────┐
                                                         │  LayerNorm(256)        │
                                                         │  Linear(256 → 128)     │
                                                         │  GELU                  │
                                                         │  Dropout(0.1)          │
                                                         │  Linear(128 → N_act)   │
                                                         └───────────────────────┘
                                                                     │
                                                              Logits (B, N_actions)
                                                                     │
                                                         ┌───────────┴───────────┐
                                                         │                       │
                                                      argmax               softmax sample
                                                    (deterministic)        (stochastic / RL)
```

---

### 3.3 ActionRewardHistoryEncoder (Detail)

```
  For each timestep i in [t-H, ..., t-1]:

  action_i  (int64)
  │
  └──► Embedding(num_actions+1, 256)        ← index num_actions = "no action" pad
             │ a_emb_i  (256,)
             │
  reward_i  (float32)                        concat
  │                                              │
  └──► Linear(1 → 256)          ──────────► cat([a_emb_i, r_emb_i])  (512,)
             │ r_emb_i  (256,)                   │
                                            Linear(512 → 256)
                                                   │
                                            + pos_embed[i]             ← learned position
                                                   │
                                            LayerNorm
                                                   │
                                            hist_token_i  (256,)

  Stack H tokens → hist_tokens  (B, H, 256)
```

**Cold-start handling:** When the robot has taken fewer than H previous steps, the missing slots are padded with:
- `action = num_actions` (the dedicated padding index in the embedding table)
- `reward = 0.0`

---

### 3.4 Token Sequence Layout

The fusion transformer receives tokens in this fixed order:

```
  Position:  0       1        2       3     ...   T      T+1     T+2   ...  T+H
             ┌─────┬────────┬───────┬───────┬───┬───────┬───────┬───────┬─────┐
  Token:     │ CLS │  lang  │ vis_0 │ vis_1 │...│vis_T-1│hist_0 │hist_1 │ ... │
             └─────┴────────┴───────┴───────┴───┴───────┴───────┴───────┴─────┘
              aggr.  instr.   oldest         newest   oldest          newest
              token  goal    frame           frame    history         history
                                                      step            step

  Total sequence length:  1 + 1 + T + H  =  1 + 1 + 3 + 4  =  9 tokens  (default config)

  Causal mask (✓ = can attend, ✗ = blocked):

          CLS  lang vis_0 vis_1 vis_2 hist_0 hist_1 hist_2 hist_3
  CLS  [   ✓    ✗    ✗     ✗     ✗     ✗      ✗      ✗      ✗   ]
  lang [   ✓    ✓    ✗     ✗     ✗     ✗      ✗      ✗      ✗   ]
  v_0  [   ✓    ✓    ✓     ✗     ✗     ✗      ✗      ✗      ✗   ]
  v_1  [   ✓    ✓    ✓     ✓     ✗     ✗      ✗      ✗      ✗   ]
  v_2  [   ✓    ✓    ✓     ✓     ✓     ✗      ✗      ✗      ✗   ]
  h_0  [   ✓    ✓    ✓     ✓     ✓     ✓      ✗      ✗      ✗   ]
  h_1  [   ✓    ✓    ✓     ✓     ✓     ✓      ✓      ✗      ✗   ]
  h_2  [   ✓    ✓    ✓     ✓     ✓     ✓      ✓      ✓      ✗   ]
  h_3  [   ✓    ✓    ✓     ✓     ✓     ✓      ✓      ✓      ✓   ]
```

The **CLS token** is read out after all transformer layers and passed to the action head. It aggregates information from the entire sequence.

---

### 3.5 Model Parameter Count

| Component | Parameters | Trainable (freeze_clip=True) |
|---|---|---|
| CLIP ViT-B/32 image encoder | ~86.2M | No (frozen) |
| CLIP text encoder | ~37.8M | No (frozen) |
| `vis_proj` Linear(512→256) | 131,328 | Yes |
| `lang_proj` Linear(512→256) | 131,328 | Yes |
| `ActionRewardHistoryEncoder` | ~267,000 | Yes |
| CLS token (learnable) | 256 | Yes |
| Causal Transformer (6L, 8H) | ~3.2M | Yes |
| Action head | ~33,000 | Yes |
| **Total trainable** | **~3.8M** | — |
| **Total (incl. frozen CLIP)** | **~128M** | — |

> With `freeze_clip: false`, all 128M parameters become trainable. Recommended only after SFT converges.

---

## 4. How the Models Are Combined — Step by Step

This section explains exactly what happens inside `RLConditionedVLA.forward()`, why each combination choice was made, and what problem each step solves.

---

### 4.1 The Problem: Three Incompatible Representations

The three input streams live in completely different spaces:

| Input | Raw format | Problem |
|---|---|---|
| Video frames | Pixel tensors `(B, T, 3, 224, 224)` | High-dimensional, spatial, no semantic meaning on their own |
| Language | Integer token IDs `(B, 77)` | Symbolic, discrete — no overlap with pixel space |
| Action/reward history | Mixed: int indices + float scalars `(B, H)` each | Structured but heterogeneous — need to be made "token-like" |

None of these can be directly added or compared. They must each be **encoded into the same vector space** before any cross-modal reasoning can happen.

The solution is a four-step pipeline:
```
  raw inputs  →  encode each modality  →  project to shared 256-dim space
              →  assemble token sequence  →  fuse with transformer  →  action
```

---

### 4.2 Step 1 — Encode Vision with CLIP

**Code:** `RLConditionedVLA.encode_frames()` (`models/vla_model.py:117`)

```
  Input:  frames  (B, T, 3, 224, 224)
                    ↓
  Flatten:  (B*T, 3, 224, 224)          ← treat each frame independently
                    ↓
  CLIP ViT-B/32 image encoder            ← pretrained, frozen
    • Splits image into 7×7 = 49 patches of 32×32 pixels
    • Each patch → 768-dim embedding (ViT internal dim)
    • 12 self-attention layers over the 49 patch tokens
    • [CLS] token aggregates the whole image
    • Output: global image embedding
                    ↓
  CLIP output: (B*T, 512)                ← 512-dim per frame
                    ↓
  Reshape:  (B, T, 512)
                    ↓
  vis_proj  Linear(512 → 256)            ← trainable
                    ↓
  vis_tokens  (B, T, 256)
```

**Why CLIP for vision?**
- CLIP was trained on 400M (image, text) pairs — it produces embeddings where "a photo of a red cube" and an image of a red cube are close together in vector space. A randomly initialized CNN has no such property.
- We encode each frame independently (not as a video with temporal convolutions) because the fusion transformer handles temporal relationships itself.
- We encode T=3 frames, not just the latest, to give the model a short visual memory of how the scene changed.

**Why freeze CLIP?**
- CLIP's visual representations are already excellent for semantic understanding.
- Unfreezing it during early training would destroy this alignment and require much more data to recover.
- Only unfreeze after SFT converges (Phase C ablation plan).

---

### 4.3 Step 2 — Encode Language with CLIP

**Code:** `RLConditionedVLA.encode_language()` (`models/vla_model.py:129`)

```
  Input:  lang_tokens  (B, 77)           ← CLIP tokenizer output (padded to 77 tokens)
                    ↓
  CLIP text encoder (Transformer)         ← pretrained, frozen
    • 12 self-attention layers over the 77 tokens
    • Output at [EOS] position = global sentence embedding
                    ↓
  CLIP output: (B, 512)                  ← one vector per instruction
                    ↓
  lang_proj  Linear(512 → 256)           ← trainable
                    ↓
  lang_token  (B, 1, 256)                ← unsqueeze: makes it a single token
```

**Why CLIP for language (not BERT, GPT, T5)?**
- CLIP's text encoder was trained jointly with its image encoder. This means "red cube" in text is close to an image of a red cube in the same 512-dim space — this cross-modal alignment is exactly what we want for a robot following visual instructions.
- A separate text encoder (BERT, T5) would produce text embeddings in a different space from the visual embeddings, requiring much more training data to bridge them.

**Why one token for the whole instruction?**
- We want the language to act as a global "goal context" that conditions how the model interprets the visual and action tokens — a single token is the right level of abstraction.
- Word-level language tokens would make the sequence much longer and complicate the attention pattern.

**Why is CLIP's text output one vector?**
- The CLIP text encoder's output at the `[EOS]` (end-of-sequence) position aggregates the full sentence meaning — CLIP is specifically trained for this.

---

### 4.4 Step 3 — Encode (Action, Reward) History

**Code:** `ActionRewardHistoryEncoder.forward()` (`models/vla_model.py:41`)

```
  For each of the H history timesteps (i = t-H, t-H+1, ..., t-1):

  action_i  (int64)
       ↓
  action_embed  Embedding(num_actions+1, 256)
       ↓  a_emb_i  (256,)
       │
  reward_i  (float32)
       ↓
  reward_proj  Linear(1 → 256)
       ↓  r_emb_i  (256,)
       │
       ├─── concatenate ──► (512,)
       │
       ↓
  fusion  Linear(512 → 256)          ← learn how to combine action + reward signals
       ↓
  + pos_embed[i]  Embedding(H, 256)  ← tell the model which timestep this is
       ↓
  LayerNorm
       ↓
  hist_token_i  (256,)

  Stack H tokens  →  hist_tokens  (B, H, 256)
```

**Why a separate embedding for actions and rewards instead of just concatenating the raw numbers?**
- Actions are **discrete** (index 3 doesn't mean "3 times more" than index 1 — they're different actions). An embedding table assigns each action its own learned dense vector.
- Rewards are **continuous** scalars. Projecting them through `Linear(1→256)` lets the model learn what magnitude ranges matter and how to represent them in the shared space.
- Concatenating then projecting (`Linear(512→256)`) lets the model learn a joint representation where the reward **modifies** the meaning of the action — e.g., "action 3 with reward -0.1" is encoded differently from "action 3 with reward +1.0".

**Why the +1 in `Embedding(num_actions + 1, 256)`?**
- At the start of an episode, the robot has no history. The extra index (`num_actions`) is a dedicated "no previous action" padding token with its own learned embedding — not zero, which would be confused with a meaningful representation.

**Why learned positional embedding?**
- Without positional encoding, the transformer treats all history tokens as an unordered set. We need to tell it which step is oldest and which is newest.
- Learned (not sinusoidal) because the history is short (H=4) and the positions have specific semantic meaning (t-4 vs t-1 have very different relevance).

---

### 4.5 Step 4 — Project to a Shared Space

All three encoders use different native dimensions. Before they can interact in a transformer, they must live in the same vector space:

```
  CLIP vision output    512-dim  →  vis_proj  Linear(512→256)  →  256-dim
  CLIP language output  512-dim  →  lang_proj Linear(512→256)  →  256-dim
  History encoder       256-dim  (already in target dim)        →  256-dim
```

**Why 256 and not 512 (CLIP's native dim)?**
- Keeping the fusion transformer at 512-dim would double its parameter count and memory.
- 256 is large enough for the fusion transformer to reason across modalities while staying efficient.
- The projection layers (`vis_proj`, `lang_proj`) are trainable even when CLIP is frozen — they learn to adapt CLIP's representations for the robot task domain.

**Why not just average the three vectors?**
- Averaging destroys information — you cannot tell apart which part of the combined vector came from vision vs language vs history.
- A transformer learns *which* tokens to attend to based on content — it can learn "for picking tasks, pay more attention to visual tokens near the instruction tokens; for recovery after failure, pay more attention to the history tokens."

---

### 4.6 Step 5 — Assemble the Token Sequence

**Code:** `RLConditionedVLA.forward()` (`models/vla_model.py:148`)

```python
sequence = torch.cat([lang_tokens_, vis_tokens, hist_tokens, cls], dim=1)
# shape: (B, 1 + T + H + 1, 256)
#         └─ lang  └─ vis  └─hist  └─ CLS (aggregation token, goes LAST)
```

The ordering is deliberate:

```
  Position:  0         1      2      3       4      5      6      7      8
             ┌────────┬──────┬──────┬───────┬──────┬──────┬──────┬──────┬─────┐
  Token:     │  lang  │ vis₀ │ vis₁ │ vis₂  │hist₀ │hist₁ │hist₂ │hist₃ │ CLS │
             └────────┴──────┴──────┴───────┴──────┴──────┴──────┴──────┴─────┘
              (goal)   (oldest        (newest) (oldest history      newest) (reads
                        frames)                 steps)                      all)
```

**Why this ordering?**

| Choice | Reason |
|---|---|
| Language first | Sets the goal context. Vision tokens attending leftward will see the instruction and can use it to focus on task-relevant objects. |
| Vision before history | Temporal causality — observe the world first, then consider what was done in it. |
| History before CLS | The CLS token is the last to "speak" — it reads the full context and summarizes it for the action head. |
| CLS at the END | Critical: with a causal mask, only later positions can attend to earlier ones. CLS must be last to attend to everything. (See §4.10 for the bug this caused.) |

---

### 4.7 Step 6 — Fuse with a Causal Transformer

**Code:** `models/vla_model.py:159`

```python
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
out = self.transformer(sequence, mask=causal_mask)
```

**What the causal mask looks like** (True = blocked, ✓ = allowed):

```
               lang  v₀  v₁  v₂  h₀  h₁  h₂  h₃  CLS
  lang     [    ✓    ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗  ]
  vis₀     [    ✓    ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗  ]  ← oldest frame sees language
  vis₁     [    ✓    ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗  ]
  vis₂     [    ✓    ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗  ]  ← newest frame sees all frames
  hist₀    [    ✓    ✓   ✓   ✓   ✓   ✗   ✗   ✗   ✗  ]  ← oldest history sees everything before it
  hist₁    [    ✓    ✓   ✓   ✓   ✓   ✓   ✗   ✗   ✗  ]
  hist₂    [    ✓    ✓   ✓   ✓   ✓   ✓   ✓   ✗   ✗  ]
  hist₃    [    ✓    ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✗  ]  ← newest history sees all past
  CLS      [    ✓    ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓  ]  ← CLS attends to EVERYTHING
```

**What actually happens inside the transformer:**

Each of the 6 layers performs:
```
  token_i_out  =  LayerNorm(token_i_in
                   + MultiHeadAttention(query=token_i, keys/values=all_j≤i)
                   + FFN(attention_output))
```

After 6 layers, the CLS token's representation has been refined by attending to all other tokens, across all layers. The attention pattern is learned — the model figures out which (language, visual, history) combinations matter for which action decisions.

**Why 6 layers and 8 heads?**
- 6 layers: enough depth for cross-modal reasoning (3 modalities interacting) without being too slow to train.
- 8 heads: allows the model to maintain 8 parallel "questions" over the input (e.g., one head might specialize in language-visual alignment, another in history-action patterns).
- FFN dim = 1024 (= 4 × 256): standard 4× expansion ratio from the original Transformer paper.

**Why a transformer and not cross-attention?**
- Cross-attention would need to designate one modality as the "query" and the others as "keys/values" — an arbitrary choice that might not be optimal.
- Self-attention in a joint token sequence lets every modality attend to every other modality symmetrically (within the causal constraint). The model learns which cross-modal relationships matter.

---

### 4.8 Step 7 — Read Out and Classify

**Code:** `models/vla_model.py:166`

```python
cls_out = out[:, -1, :]          # (B, 256) — last token = CLS
return self.action_head(cls_out) # (B, num_actions)
```

The CLS token at the last position now contains a 256-dim representation that has attended to all 8 preceding tokens. It is passed through:

```
  cls_out  (B, 256)
       ↓
  LayerNorm(256)           ← stabilize scale before the linear layers
       ↓
  Linear(256 → 128)
       ↓
  GELU activation          ← smooth nonlinearity (better gradient flow than ReLU)
       ↓
  Dropout(0.1)             ← regularization during training
       ↓
  Linear(128 → num_actions)
       ↓
  logits  (B, num_actions) ← raw scores, not probabilities

  At inference:    argmax(logits)          → deterministic action
  During RL:       softmax(logits) → sample → stochastic action
```

**Why a two-layer head instead of one linear layer?**
- The CLS token contains a mixed representation from all modalities. A single linear layer would have to simultaneously disentangle the modalities and learn the action boundaries. A two-layer MLP with a nonlinearity gives it capacity to do this in two steps.
- The 128-dim bottleneck forces the model to compress into a more action-relevant representation before the final prediction.

---

### 4.9 Why Not Simpler Fusion?

The following alternatives were considered and rejected:

| Alternative | Why rejected |
|---|---|
| **Concatenate all features → single MLP** | Fixed fusion, no cross-modal attention. The model can't learn to "look at the red cube when the instruction mentions red" — it mixes all features equally. |
| **Late fusion: separate MLP per modality, sum outputs** | Each modality is processed independently — no cross-modal interaction before the decision. Can't correlate "the reward was bad" with "what the scene looked like when that happened." |
| **Cross-attention with history as query** | Forces history to be the "primary" modality querying vision+language. Arbitrary choice; a joint transformer is more flexible. |
| **RNN over the history** | Implicit memory — harder to inspect, gradients vanish over long sequences. Explicit token-based history lets you inspect attention weights to understand which past steps the model is using. |
| **Bidirectional (no causal mask) transformer** | Simpler. The causal mask enforces temporal ordering (old frames → new frames → history steps → CLS), which is a useful inductive bias. If needed, this is a simple ablation (A6). |

---

### 4.10 Bug Found and Fixed: CLS Token Position

**Discovered:** 2026-03-04

**The bug:**
The original implementation placed the CLS token at **position 0** (first in sequence):
```python
# WRONG — CLS is at position 0
sequence = torch.cat([cls, lang_tokens_, vis_tokens, hist_tokens], dim=1)
cls_out  = out[:, 0, :]
```

**Why this is wrong:**

In a TransformerEncoder with a causal mask, position `i` can only attend to positions `j ≤ i`. With CLS at position 0:
- Layer 1: CLS can only attend to itself → output = f(CLS_init)
- Layer 2: CLS can only attend to itself (its own Layer 1 output) → output = f(f(CLS_init))
- After 6 layers: CLS output = f⁶(CLS_init) — only a function of its own initial value

**Consequence:** the action head was reading from a token that had seen **zero information** from the vision, language, or history tokens. The entire CLIP+history encoding was effectively ignored.

**The fix:**

Move CLS to the **last position**:
```python
# CORRECT — CLS is at the last position
sequence = torch.cat([lang_tokens_, vis_tokens, hist_tokens, cls], dim=1)
cls_out  = out[:, -1, :]
```

With CLS at the last position, the causal mask allows it to attend to **all** previous tokens. After 6 layers it aggregates information from language, every visual frame, and every history step — which is the intended behavior.

**Files fixed:** `models/vla_model.py:157,166` and `training/rl_trainer.py:217,223`

---

## 5. Training Pipeline

### 5.1 Phase 1: Behavioral Cloning (SFT)

```
  Expert demonstrations (episodes.pkl)
            │
            ▼
  TrajectoryDataset
  (sliding window: for each timestep t,
   extract frames[t-T:t], action_hist[t-H:t],
   reward_hist[t-H:t], target=action[t])
            │
            ├─── 90% train split
            └─── 10% val split
                      │
                      ▼
            DataLoader (batch_size=32, shuffled)
                      │
                      ▼
            RLConditionedVLA.forward(...)
                      │
                      ▼
            CrossEntropyLoss + label smoothing (0.05)
                      │
                      ▼
            AdamW (lr=3e-4) + CosineAnnealingLR
                      │
                      ▼
            Save best_sft.pt  (highest val accuracy)
```

**Hyperparameters:**

| Setting | Value | Rationale |
|---|---|---|
| Loss | Cross-entropy + label_smoothing=0.05 | Prevents overconfident predictions on noisy demo data |
| LR | 3e-4 → 1e-6 (cosine) | Standard for transformer fine-tuning |
| Grad clip | 1.0 | Stabilizes early training |
| Label smoothing | 0.05 | Regularizes; robot actions are often ambiguous |
| Epochs | 50 | Adjust based on dataset size |
| Batch size | 32 | Balance between stability and speed |

**Run:**
```bash
python -m training.sft_trainer --config configs/config.yaml
```

---

### 5.2 Phase 2: Online RL Fine-Tuning

```
  Load best_sft.pt
          │
          ├──► Policy model  (trainable)
          └──► Reference BC model  (frozen — for KL penalty)
                    │
  For each RL epoch:
                    │
                    ▼
          Roll out 4 episodes in environment
          Collect buffer: [(frames, lang, act_hist, rew_hist, action, reward, done), ...]
                    │
                    ▼
          Compute discounted returns G_t  (γ=0.99)
          Normalize returns (mean/std)
                    │
                    ▼
          Forward pass through policy:
            logits   = model(frames, lang, act_hist, rew_hist)
            values   = ValueHead(cls_features)
            advantage = G_t - values.detach()
                    │
                    ▼
          Loss = policy_loss + value_loss + entropy_bonus + KL_penalty
                    │
                 ┌──┴──────────────────────────────────────────┐
                 │                                              │
          policy_loss                                   KL_penalty
          = -log π(a|s) × advantage            = KL(π_current || π_BC)
          (REINFORCE)                           (prevents forgetting BC)
                 │
                 └──► AdamW (lr=1e-5) update
                    │
                    ▼
          Save best_rl.pt  (highest mean episode return)
```

**Loss breakdown:**

```
  total_loss = policy_loss
             + 0.5  × value_loss        (MSE of value baseline vs returns)
             - 0.01 × entropy           (exploration bonus)
             + 0.1  × KL_divergence     (stay close to BC policy)
```

**Why each term:**

| Term | Purpose | Default coef |
|---|---|---|
| Policy loss | Core REINFORCE signal — reward good actions | 1.0 |
| Value loss | Train the baseline to reduce variance | 0.5 |
| Entropy bonus | Encourage exploration, prevent premature collapse | 0.01 |
| KL penalty | Prevent catastrophic forgetting of BC behavior | 0.1 |

**Run:**
```bash
python -m training.rl_trainer --config configs/config.yaml
```

---

### 5.3 Evaluation

```
  Load checkpoint (SFT or RL)
          │
          ▼
  Roll out N episodes (default: 50)
  At each step:
    - Build frame queue (last T frames)
    - Build action/reward history queue
    - Forward pass → logits → argmax (deterministic) or sample (stochastic)
    - Execute action, receive reward, update queues
          │
          ▼
  Report:
    - Mean episode return ± std
    - Mean episode length
    - Success rate  (return ≥ threshold)
    - Action distribution histogram
```

```bash
# Evaluate RL checkpoint
python -m evaluation.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/rl/best_rl.pt \
    --episodes 50

# Stochastic (sample from softmax rather than argmax)
python -m evaluation.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/rl/best_rl.pt \
    --stochastic
```

---

## 6. What Has Been Built

### 6.1 File-by-File Breakdown

#### `models/vla_model.py`

| Class / Function | Description |
|---|---|
| `ActionRewardHistoryEncoder` | Encodes (action_idx, reward) pairs into token embeddings with positional encoding |
| `RLConditionedVLA` | Full model: CLIP encoders + history encoder + causal transformer + action head |
| `RLConditionedVLA.encode_frames()` | Runs CLIP ViT-B/32 on a batch of frame sequences |
| `RLConditionedVLA.encode_language()` | Runs CLIP text encoder on tokenized instructions |
| `RLConditionedVLA.forward()` | Full forward pass → action logits |
| `RLConditionedVLA.predict()` | Single-sample greedy action selection (no grad) |
| `RLConditionedVLA.num_trainable_params()` | Returns count of trainable parameters |

#### `data/trajectory_dataset.py`

| Class / Function | Description |
|---|---|
| `TrajectoryDataset` | PyTorch Dataset — sliding window over episode list |
| `make_random_episodes()` | Generates synthetic random episodes for pipeline testing |
| `load_episodes()` | Load episode list from `.pkl` file |
| `save_episodes()` | Save episode list to `.pkl` file |

**Episode format:**
```python
{
    "frames":      np.ndarray,  # (T_ep, H, W, 3) uint8
    "instruction": str,          # natural-language task goal
    "actions":     np.ndarray,  # (T_ep,) int64 — discrete action indices
    "rewards":     np.ndarray,  # (T_ep,) float32
}
```

#### `training/sft_trainer.py`

| Function | Description |
|---|---|
| `build_dataloaders()` | Loads episodes, creates train/val split, returns DataLoaders |
| `build_model()` | Instantiates `RLConditionedVLA` from config |
| `run_epoch()` | One train or val epoch — returns loss and accuracy |
| `train()` | Full SFT training loop with checkpointing and JSON logging |

#### `training/rl_trainer.py`

| Class / Function | Description |
|---|---|
| `ValueHead` | Small MLP: maps CLS token features to scalar state-value estimate |
| `RolloutBuffer` | Accumulates one batch of (frames, lang, a_hist, r_hist, action, reward, done) |
| `RolloutBuffer.compute_returns()` | Discounted return computation with episode boundary reset |
| `collect_rollout()` | Runs policy for one episode, fills RolloutBuffer |
| `rl_update()` | REINFORCE + value baseline + KL penalty update step |
| `rl_train()` | Full RL training loop |

#### `envs/sim_env.py`

| Class | Description |
|---|---|
| `BaseEnv` | Abstract interface: `reset()` → obs, `step(action)` → (obs, r, done, info) |
| `RandomDummyEnv` | Zero-dependency test environment — random frames, reward +1 if action matches hidden target |
| `SimEnv` | Gym/Gymnasium wrapper with domain randomization support |
| `RealEnv` | Stub for real robot — fill in camera capture and robot SDK calls |

#### `evaluation/evaluate.py`

| Function | Description |
|---|---|
| `evaluate()` | Rolls out N episodes, computes mean return, success rate, action distribution |

#### `configs/config.yaml`

Single source of truth for all hyperparameters. Sections: `model`, `data`, `training` (SFT), `rl`, `env`, `eval`.

#### `generate_report.py`

| Function | Description |
|---|---|
| `update_last_updated()` | Rewrites the "Last Updated" date in REPORT.md |
| `update_changelog()` | Appends a dated entry to the Change Log table |
| `build_pdf()` | Runs pandoc + xelatex to produce REPORT.pdf |
| `generate()` | Updates date then builds PDF |
| `watch()` | Polls REPORT.md every 2s, rebuilds PDF on change |

---

### 6.2 Key Design Decisions

| Decision | Choice | Why | Alternative Considered |
|---|---|---|---|
| Vision-language backbone | CLIP ViT-B/32 | Strong zero-shot V-L alignment, fits on one GPU | BLIP-2 (overkill), R3M (no language) |
| Action space | Discrete (N classes) | Simpler to train/eval, natural for navigation tasks | Continuous (joint angles), RT-2 tokenized bins |
| RL history encoding | Dedicated token encoder | Explicit, interpretable, inspectable via attention | RNN hidden state (implicit, harder to debug) |
| RL algorithm | REINFORCE + baseline | Simple, correct, easy to debug as prototype | PPO (more stable but complex) |
| Catastrophic forgetting | KL penalty vs BC checkpoint | Standard RLHF practice, proven to work | Elastic weight consolidation (EWC) |
| CLIP frozen by default | Yes | Preserve V-L alignment during early training | Unfreeze after SFT convergence |
| Causal mask | Upper-triangular boolean | Enforces temporal ordering, compatible with autoregressive extension | Bidirectional (allows future leakage) |

---

## 7. Environment & Data

### Installation

```bash
# Core dependencies
pip install torch torchvision Pillow PyYAML numpy matplotlib

# CLIP backbone
pip install git+https://github.com/openai/CLIP.git

# Simulation environments (optional)
pip install gymnasium
pip install gym-robotics    # FrankaKitchen, FetchReach, FetchPush
pip install minigrid        # MiniGrid navigation tasks (simplest to start)
```

### Using Your Own Data

```python
from data.trajectory_dataset import save_episodes

episodes = [
    {
        "frames":      np.ndarray,  # (T, H, W, 3) uint8
        "instruction": "pick up the red cube",
        "actions":     np.ndarray,  # (T,) int64
        "rewards":     np.ndarray,  # (T,) float32
    },
    # ... more episodes
]

save_episodes(episodes, "data/my_episodes.pkl")
# Then set data.episodes_path in config.yaml
```

### Generating Scripted Data (placeholder)

Until real or learned demonstrations are available, `make_random_episodes()` generates random synthetic data so the entire pipeline can be tested end-to-end without any real environment.

### Connecting a Real Robot

1. Implement `envs/sim_env.py → RealEnv._capture_frame()` — call your camera SDK
2. Implement `RealEnv.step(action)` — send command to robot arm, return reward
3. Change the import in `training/rl_trainer.py`:
   ```python
   from envs.sim_env import RealEnv as SimEnv
   ```

---

## 8. Experiments & Results

> No experiments have been run yet. This section will be filled in as results are collected.

### 8.1 Experiment Log

| Date | Phase | Environment | Epochs | Val Acc / Mean Return | Notes |
|---|---|---|---|---|---|
| — | — | — | — | — | Pending |

### 8.2 Planned Ablations

These ablations will isolate the contribution of each design choice:

| # | Ablation | What changes | Expected outcome |
|---|---|---|---|
| A1 | No history | `history_len=0` | Lower performance — model cannot recover from mistakes |
| A2 | Action-only history | Remove reward from history encoder | Slightly worse — model can't tell if actions worked |
| A3 | Unfreeze CLIP | `freeze_clip: false` | Better if data is large, worse if small (overfitting) |
| A4 | PPO vs REINFORCE | Replace rl_trainer algorithm | More stable training, better final performance |
| A5 | History length sweep | H ∈ {1, 2, 4, 8} | Find optimal memory window |
| A6 | Bidirectional attention | Remove causal mask | Similar or slightly better (at cost of temporal coherence) |

---

## 9. Issues & Fixes

| Date | Issue | Fix | File |
|---|---|---|---|
| 2026-03-03 | `--highlight-style` deprecated in pandoc 3.9; LaTeX error from complex YAML header-includes | Removed header-includes from YAML, passed all styling as pandoc `-V` CLI variables | `generate_report.py` |
| 2026-03-04 | CLS token at position 0 with causal mask is isolated — never attends to vision, language, or history tokens across any layer | Moved CLS to last position in the sequence; read out `out[:, -1, :]` instead of `out[:, 0, :]` | `models/vla_model.py`, `training/rl_trainer.py` |

---

## 10. Next Steps

Steps are ordered by dependency — each row should be completed before the ones below it that depend on it.

### Phase A — Get the Pipeline Running (Immediate)

| # | Task | Description | Depends on |
|---|---|---|---|
| A1 | Choose simulation environment | Pick one: **MiniGrid-Empty** (simplest), **FetchReach** (realistic), or **FrankaKitchen** (hardest) | — |
| A2 | Define action space | List the exact N discrete actions for chosen env, update `config.yaml → model.num_actions` | A1 |
| A3 | Collect demonstration data | Run a scripted/random policy or use env's built-in demo mode. Save as `data/episodes.pkl` | A1, A2 |
| A4 | Run Phase 1 SFT | `python -m training.sft_trainer --config configs/config.yaml`. Monitor val accuracy. | A3 |
| A5 | Verify Phase 1 checkpoint | Load `checkpoints/best_sft.pt`, run evaluator, check action distribution is non-trivial | A4 |

### Phase B — RL Fine-Tuning

| # | Task | Description | Depends on |
|---|---|---|---|
| B1 | Run Phase 2 RL | `python -m training.rl_trainer --config configs/config.yaml`. Monitor mean return per epoch. | A5 |
| B2 | Add TensorBoard logging | Add `SummaryWriter` calls to both trainers. Log loss curves, accuracy, return. | B1 |
| B3 | Compare SFT vs RL | Evaluate both `best_sft.pt` and `best_rl.pt` with 50 episodes. Record in §7.1. | B1 |

### Phase C — Improve & Ablate

| # | Task | Description | Depends on |
|---|---|---|---|
| C1 | Run ablations A1–A6 | Each ablation = change one config param, retrain, evaluate | B3 |
| C2 | Upgrade to PPO | Replace `rl_trainer.py` REINFORCE loop with PPO actor-critic | B3 |
| C3 | Unfreeze CLIP layers | Set `freeze_clip: false`, reduce LR, fine-tune entire model | B3 |

### Phase D — Sim-to-Real

| # | Task | Description | Depends on |
|---|---|---|---|
| D1 | Enable domain randomization | Set `env.domain_randomization: true` in config. Tune `obs_noise_std`. | B3 |
| D2 | Implement `RealEnv` | Wire real camera and robot arm SDK into `envs/sim_env.py → RealEnv` | D1 |
| D3 | Evaluate on real robot | Run `evaluate.py` with `RealEnv`. Record success rate. | D2 |

### Phase E — Write-Up

| # | Task | Description |
|---|---|---|
| E1 | Update §8 with all experiment results | Fill in the experiment log and ablation results table |
| E2 | Write thesis chapter | Describe architecture, training, results, ablations |
| E3 | Generate final PDF report | `python generate_report.py` |

---

## 11. References

| Paper | Authors | Venue | Relevance |
|---|---|---|---|
| "Learning Transferable Visual Models from Natural Language Supervision" | Radford et al. | ICML 2021 | CLIP backbone — vision + language encoders |
| "Decision Transformer: Reinforcement Learning via Sequence Modeling" | Chen et al. | NeurIPS 2021 | Core idea: (s,a,r) history as sequence tokens |
| "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" | Brohan et al. | CoRL 2023 | VLA architecture reference; action tokenization |
| "Octo: An Open-Source Generalist Robot Policy" | Octo Team | RSS 2024 | Fine-tunable VLA; multi-task robot learning |
| "Training Language Models to Follow Instructions with Human Feedback" | Ouyang et al. | NeurIPS 2022 | KL penalty strategy against reference model (RLHF) |
| "Proximal Policy Optimization Algorithms" | Schulman et al. | arXiv 2017 | PPO — planned upgrade from REINFORCE |
| "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" | Dosovitskiy et al. | ICLR 2021 | Vision Transformer (ViT) used inside CLIP |

---

## 12. Change Log

| Date | Change |
|---|---|
| 2026-03-03 | Project initialized. Full codebase scaffolded: model, SFT trainer, RL trainer, dataset, sim env, evaluation, config, PDF report generator. |
| 2026-03-04 | REPORT.md restructured: added detailed architecture diagrams, token sequence layout, parameter count table, full file-by-file breakdown, prioritized next steps roadmap. |
| 2026-03-04 | Added Section 4: full step-by-step model combination explanation with code-level detail and rationale for every fusion decision. |
| 2026-03-04 | Bug fixed: CLS token was at position 0 with causal mask (could never attend to any other token). Moved to last position in sequence. Fixed in `models/vla_model.py` and `training/rl_trainer.py`. |
