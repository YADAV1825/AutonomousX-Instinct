#!/usr/bin/env python3

import os
import math
import time
import gc
from functools import partial
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn    
from flax.training import train_state, checkpoints
from jax.tree_util import tree_map
import optax


# ==========================================================
# =========================== CONFIG =======================
# ==========================================================

DATA_PATH = "/mnt/diskx/data/dolma_tokens.bin"
CKPT_DIR = os.path.abspath("./checkpoints")
VAL_SPLIT = 0.01          # 1% of data for validation
VAL_TOKENS = 5_000_000    # tokens per validation run
VAL_INTERVAL = 500_000_000  # validate every 1B tokens
LOG_FILE = "training_log.txt"
PPL_FILE = "val_perplexity.txt"

SEQ_LEN = 1024
VOCAB_SIZE = 50304

#650m ~700m
# N_LAYERS = 32
# D_MODEL = 1024
# N_HEADS = 16
# D_HEAD = 64
# D_FF = 4096

#1.3B or 1.4B something
N_LAYERS = 24
D_MODEL  = 2048
N_HEADS  = 16
D_HEAD   = 128        # MUST be D_MODEL // N_HEADS
D_FF     = 8192
ROTARY_PCT = 0.25

TOTAL_TOKENS_TARGET = 70_000_000_000
WARMUP_FRAC = 0.01

SANITY_TOKENS = 10_000_000
CKPT_INTERVAL = 26_000_000_000
#hmm after many tunings attempt this size perfectly fits at 1.5B oom and so at 1B I am already making on PILE so why make two so on DOLMA I was trying to make a larger one that fits on tpu with pmap and jax
LR_MAX = 2.5e-4  # I know i have defined LR's them here haahha but I am not gonna use them later hahaha🤣🤣🤣
LR_MIN = 2e-5

WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0

PER_DEVICE_BATCH = 6 # even 7 fit but I want it to be safer side cause at 8th it gave oom I don't want it to oom at middle so 6 is safer for a v4-8 128Gb HBM (vram)
GRAD_ACCUM = 1


# ==========================================================
# ======================== DEVICE SETUP ====================
# ==========================================================

DEVICE_COUNT = jax.device_count()
GLOBAL_BATCH = PER_DEVICE_BATCH * DEVICE_COUNT
TOKENS_PER_STEP = GLOBAL_BATCH * SEQ_LEN * GRAD_ACCUM

os.makedirs(CKPT_DIR, exist_ok=True)

print("Devices:", DEVICE_COUNT)
print("Tokens per step:", TOKENS_PER_STEP)


# ==========================================================
# ========================= MODEL ==========================
# ==========================================================
# ==========================================================
# ======================= RoPE CACHE =======================
# ==========================================================

def build_rope_cache(seq_len, head_dim, rotary_pct):
    dim = int(head_dim * rotary_pct)
    freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    pos = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", pos, freqs)
    sin = jnp.sin(angles)
    cos = jnp.cos(angles)
    return sin, cos

ROPE_SIN, ROPE_COS = build_rope_cache(SEQ_LEN, D_HEAD, ROTARY_PCT)


def apply_rope(q, k):
    # q,k shape: [B, T, H, D_HEAD]
    dim = int(D_HEAD * ROTARY_PCT)

    sin = ROPE_SIN[None, :, None, :]
    cos = ROPE_COS[None, :, None, :]

    q_rot, q_pass = q[..., :dim], q[..., dim:]
    k_rot, k_pass = k[..., :dim], k[..., dim:]

    q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
    k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]

    q_rot = jnp.concatenate(
        [q1 * cos - q2 * sin,
         q1 * sin + q2 * cos],
        axis=-1
    )

    k_rot = jnp.concatenate(
        [k1 * cos - k2 * sin,
         k1 * sin + k2 * cos],
        axis=-1
    )

    return (
        jnp.concatenate([q_rot, q_pass], axis=-1),
        jnp.concatenate([k_rot, k_pass], axis=-1),
    )

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * (scale / norm)


class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):

        h = RMSNorm(D_MODEL)(x)
        h = Attention()(h, mask)
        x = x + h

        h = RMSNorm(D_MODEL)(x)
        h = nn.Dense(D_FF, dtype=jnp.bfloat16)(h)
        h = nn.gelu(h)
        h = nn.Dense(D_MODEL, dtype=jnp.bfloat16)(h)

        return x + h
class Attention(nn.Module):
    @nn.compact
    def __call__(self, x, mask):

        B, T, C = x.shape

        qkv = nn.Dense(3 * C, use_bias=False, dtype=jnp.bfloat16)(x)
        qkv = qkv.reshape(B, T, 3, N_HEADS, D_HEAD)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # Apply RoPE
        q, k = apply_rope(q, k)

        att = jnp.einsum("bthd,bshd->bhts", q, k)
        att = att / math.sqrt(D_HEAD)
        
        mask = mask.astype(jnp.float32)
        mask = (1.0 - mask) * -1e10
        
        att = att + mask
        
        att = nn.softmax(att.astype(jnp.float32), axis=-1)
        att = att.astype(jnp.bfloat16)
        
        out = jnp.einsum("bhts,bshd->bthd", att, v)

        out = out.reshape(B, T, C)

        return nn.Dense(C, use_bias=False, dtype=jnp.bfloat16)(out)



class GPT(nn.Module):
    @nn.compact
    def __call__(self, input_ids):

        batch, seq_len = input_ids.shape

        mask = nn.attention.make_causal_mask(
            jnp.ones((batch, seq_len), dtype=jnp.bool_)
        )

        x = nn.Embed(
            VOCAB_SIZE,
            D_MODEL,
            embedding_init=nn.initializers.normal(0.02),
            dtype=jnp.bfloat16,
        )(input_ids)

        RematBlock = nn.remat(Block)

        for _ in range(N_LAYERS):
            x = RematBlock()(x, mask)

        x = RMSNorm(D_MODEL)(x)

        logits = nn.Dense(
            VOCAB_SIZE,
            use_bias=False,
            dtype=jnp.bfloat16
        )(x)

        return logits


# ==========================================================
# ====================== TRAIN STATE =======================
# ==========================================================
#attempt adamW : 23 (4hours) ( got it that AdamW won't fit more than 1B no matter what memory tricks you do for SEQ lenght 1024
#attempt adafactor: 136 (hours 68)
#😭😭😭 giving up on Adafactor can't tune it well the loss is stuck at 10.86 
#
#
# def cosine_schedule(step):
#     total_steps = TOTAL_TOKENS_TARGET // TOKENS_PER_STEP
#     warmup_steps = int(0.03 * total_steps)

#     step = jnp.minimum(step, total_steps)

#     lr = jnp.where(
#         step < warmup_steps,
#         1e-3 * step / warmup_steps,
#         1e-3 * 0.5 * (1 + jnp.cos(
#             jnp.pi * (step - warmup_steps) / (total_steps - warmup_steps)
#         ))
#     )
#     return lr
# # def cosine_with_plateau(step):
# #     total_steps = TOTAL_TOKENS_TARGET // TOKENS_PER_STEP

# #     warmup_tokens = 750_000_000
# #     plateau_tokens = 1_500_000_000

# #     warmup_steps = warmup_tokens // TOKENS_PER_STEP
# #     plateau_steps = plateau_tokens // TOKENS_PER_STEP

# #     step = jnp.minimum(step, total_steps)

# #     def warmup():
# #         return 3e-4 * step / warmup_steps

# #     def plateau():
# #         return 3e-4

# #     def decay():
# #         decay_progress = (
# #             step - warmup_steps - plateau_steps
# #         ) / (total_steps - warmup_steps - plateau_steps)

# #         cosine = 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
# #         return 3e-4 * cosine

# #     return jax.lax.cond(
# #         step < warmup_steps,
# #         lambda _: warmup(),
# #         lambda _: jax.lax.cond(
# #             step < warmup_steps + plateau_steps,
# #             lambda _: plateau(),
# #             lambda _: decay(),
# #             operand=None
# #         ),
# #         operand=None
# #     )
# learning_rate=cosine_schedule
# # learning_rate = cosine_with_plateau
# # def two_phase_schedule(step):
# #     total_steps = TOTAL_TOKENS_TARGET // TOKENS_PER_STEP

# #     phase1_tokens = 1_000_000_000
# #     phase1_steps = phase1_tokens // TOKENS_PER_STEP

# #     step = jnp.minimum(step, total_steps)

# #     def phase1():
# #         return 8e-5

# #     def phase2():
# #         decay_progress = (step - phase1_steps) / (total_steps - phase1_steps)
# #         cosine = 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
# #         return 8e-5 * cosine

# #     return jax.lax.cond(
# #         step < phase1_steps,
# #         lambda _: phase1(),
# #         lambda _: phase2(),
# #         operand=None
# #     )
# def create_state():
#     model = GPT()
#     rng = jax.random.PRNGKey(0)

#     params = model.init(
#         rng,
#         jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
#     )
#     tx = optax.chain(
#         optax.clip_by_global_norm(1.0),
#         optax.adafactor(
#             learning_rate=cosine_schedule,
#             multiply_by_parameter_scale=False,
#             decay_rate=0.999,
#             clipping_threshold=1.0,
#             weight_decay_rate=0.01,
#         )
#     )

#     return train_state.TrainState.create(
#         apply_fn=model.apply,
#         params=params,
#         tx=tx,
#     )
# # def create_state():
# #     model = GPT()
# #     rng = jax.random.PRNGKey(0)

# #     params = model.init(
# #         rng,
# #         jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
# #     )

# #     tx = optax.chain(
# #         optax.clip_by_global_norm(1.0),
# #         optax.adafactor(
# #             learning_rate=two_phase_schedule,  # ← HERE
# #             multiply_by_parameter_scale=False,
# #             decay_rate=0.999,
# #             clipping_threshold=1.0,
# #             weight_decay_rate=0.01,
# #         )
# #     )

# #     return train_state.TrainState.create(
# #         apply_fn=model.apply,
# #         params=params,
# #         tx=tx,
# #     )

# def create_state():
#     model = GPT()
#     rng = jax.random.PRNGKey(0)

#     params = model.init(
#         rng,
#         jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
#     )

#     tx = optax.chain(
#         optax.clip_by_global_norm(CLIP_NORM),
#         optax.adamw(
#             learning_rate=lr_schedule,
#             b1=0.9,
#             b2=0.95,
#             eps=1e-8,
#             weight_decay=WEIGHT_DECAY,
#         )
#     )

#     return train_state.TrainState.create(
#         apply_fn=model.apply,
#         params=params,
#         tx=tx,
#     )



#day 3rd after failing on Adafactor i decided to try on Lion if that also fails I will try with SGD+ momentum but not gonna give up , cause it was never an option!! Power comes in response to a need not a desire you need to create that need !!!



# attempt Lion: 4th (2 hours)
def create_state():
    model = GPT()
    rng = jax.random.PRNGKey(0)

    params = model.init(
        rng,
        jnp.ones((1, SEQ_LEN), dtype=jnp.int32),
    )

    total_steps = TOTAL_TOKENS_TARGET // TOKENS_PER_STEP
    warmup_steps = int(0.03 * total_steps)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=2e-4,        # <-- Lion sweet spot
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=2e-5,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.lion(
            learning_rate=schedule,
            b1=0.9,
            b2=0.99,
            weight_decay=0.1,
        ),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
# ==========================================================
# ======================== LR SCHEDULE =====================
# ==========================================================

def lr_schedule(step):

    step = jnp.asarray(step, dtype=jnp.float32)
    tokens_seen = step * jnp.asarray(TOKENS_PER_STEP, dtype=jnp.float32)

    total_tokens = jnp.asarray(TOTAL_TOKENS_TARGET, dtype=jnp.float32)
    warmup_tokens = total_tokens * WARMUP_FRAC

    def warmup():
        return LR_MAX * tokens_seen / warmup_tokens

    def cosine():
        progress = (tokens_seen - warmup_tokens) / (total_tokens - warmup_tokens)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return LR_MIN + (LR_MAX - LR_MIN) * cosine

    return jax.lax.cond(
        tokens_seen < warmup_tokens,
        lambda _: warmup(),
        lambda _: cosine(),
        operand=None
    )


# ==========================================================
# ======================== TRAIN STEP ======================
# ==========================================================

@partial(jax.pmap, axis_name="data")
def train_step(state, batch):

    def loss_fn(params):
        logits = state.apply_fn(params, batch["input_ids"])
        logits = jnp.clip(logits, -30.0, 30.0)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32),
            batch["labels"]
        )

        return loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")

    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, state.params
    )

    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )

    return new_state, loss
    
@partial(jax.pmap, axis_name="data")
def eval_step(state, batch):

    def loss_fn(params):
        logits = state.apply_fn(params, batch["input_ids"])
        logits = jnp.clip(logits, -30.0, 30.0)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32),
            batch["labels"]
        )
        return loss.mean()

    loss = loss_fn(state.params)
    loss = jax.lax.pmean(loss, axis_name="data")
    return loss

# ==========================================================
# ================= SAFE CHECKPOINT ========================
# ==========================================================
# bro i completely forgot about checkpointing after running this script for 3 hours and realizing it !!!😂🤡

def save_checkpoint_safe(state, tokens_seen):

    if jax.process_index() != 0:
        return

    print(f"Saving checkpoint at {tokens_seen} tokens")

    ckpt_dir = os.path.join(
        CKPT_DIR,
        f"ckpt_{tokens_seen}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Take one replica (because of pmap)
    host_state = tree_map(lambda x: x[0], state)

    # Move to CPU
    cpu_state = jax.device_get(host_state)
    jax.block_until_ready(cpu_state)

    checkpoints.save_checkpoint(
        ckpt_dir,
        cpu_state,
        step=0,
        overwrite=True,
        keep=1,
    )

    del cpu_state
    del host_state
    gc.collect()

# ==========================================================
# ============================ MAIN ========================
# ==========================================================
def main():

    state = create_state()
    state = jax.device_put_replicated(state, jax.devices())

    train_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")

    # ---- Split dataset ----
    split_idx = int(len(train_data) * (1 - VAL_SPLIT))
    train_tokens = train_data[:split_idx]
    val_tokens   = train_data[split_idx:]

    tokens_seen = 0
    train_ptr = 0
    val_ptr = 0

    next_ckpt = CKPT_INTERVAL
    next_val  = VAL_INTERVAL
    sanity_done = False

    pbar = tqdm(total=TOTAL_TOKENS_TARGET, unit="tok", unit_scale=True)

    while tokens_seen < TOTAL_TOKENS_TARGET:

        # ================= TRAIN =================
        seqs = []
        for _ in range(GLOBAL_BATCH):
            if train_ptr + SEQ_LEN + 1 >= len(train_tokens):
                train_ptr = 0
            seq = train_tokens[train_ptr:train_ptr + SEQ_LEN + 1]
            train_ptr += SEQ_LEN + 1
            seqs.append(seq)

        batch_np = np.stack(seqs).reshape(
            DEVICE_COUNT,
            PER_DEVICE_BATCH,
            SEQ_LEN + 1
        )

        batch = {
            "input_ids": batch_np[:, :, :-1].astype(np.int32),
            "labels":    batch_np[:, :, 1:].astype(np.int32),
        }

        state, loss = train_step(state, batch)
        jax.block_until_ready(loss)

        tokens_seen += TOKENS_PER_STEP
        train_loss = float(loss[0])

        pbar.set_postfix({"loss": f"{train_loss:.3f}"})
        pbar.update(TOKENS_PER_STEP)

        # ================= SANITY CKPT =================
        if not sanity_done and tokens_seen >= SANITY_TOKENS:
            save_checkpoint_safe(state, tokens_seen)
            sanity_done = True

        # ================= VALIDATION =================
        if tokens_seen >= next_val:

            total_val_loss = 0.0
            val_steps = max(1, VAL_TOKENS // TOKENS_PER_STEP)


            for _ in range(val_steps):

                seqs = []
                for _ in range(GLOBAL_BATCH):
                    if val_ptr + SEQ_LEN + 1 >= len(val_tokens):
                        val_ptr = 0
                    seq = val_tokens[val_ptr:val_ptr + SEQ_LEN + 1]
                    val_ptr += SEQ_LEN + 1
                    seqs.append(seq)

                batch_np = np.stack(seqs).reshape(
                    DEVICE_COUNT,
                    PER_DEVICE_BATCH,
                    SEQ_LEN + 1
                )

                batch = {
                    "input_ids": batch_np[:, :, :-1].astype(np.int32),
                    "labels":    batch_np[:, :, 1:].astype(np.int32),
                }

                val_loss = eval_step(state, batch)
                jax.block_until_ready(val_loss)
                total_val_loss += float(val_loss[0])

            avg_val_loss = total_val_loss / val_steps
            val_perplexity = math.exp(avg_val_loss)

            log_line = (
                f"{tokens_seen} tokens | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f}\n"
            )

            print(log_line.strip())

            with open(LOG_FILE, "a") as f:
                f.write(log_line)
            with open(PPL_FILE, "a") as f:
                f.write(
                    f"{tokens_seen} tokens | "
                    f"val_perplexity={val_perplexity:.6f}\n"
                )

            next_val += VAL_INTERVAL

        # ================= REGULAR CKPT =================
        if tokens_seen >= next_ckpt:
            save_checkpoint_safe(state, tokens_seen)
            next_ckpt += CKPT_INTERVAL


    # ==================================================
    # ================= FINAL SAVE =====================
    # ==================================================

    print("Saving FINAL checkpoint...")
    save_checkpoint_safe(state, tokens_seen)

    print("Training complete.")



if __name__ == "__main__":
    main()

#hope this works!!!