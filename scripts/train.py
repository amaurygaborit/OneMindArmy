import os
import sys
import yaml
import json
import argparse
import subprocess
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

try:
    from dataset import OneMindArmyDataset
except ImportError:
    print("[Fatal] dataset.py not found. Ensure it is in the same directory.")
    sys.exit(1)


# ==============================================================================
# --- 1. NEURAL NETWORK ARCHITECTURE (TRANSFORMER) ---
# ==============================================================================

class OneMindArmyNet(nn.Module):
    def __init__(self, config, meta):
        super().__init__()

        self.action_space  = meta["actionSpace"]
        self.num_players   = meta["numPlayers"]
        self.num_pos       = meta["numPos"]
        self.nn_input_size = meta["nnInputSize"]

        # Token geometry
        self.kTokenDim = 4 + self.num_pos
        self.seq_len   = self.nn_input_size // self.kTokenDim

        if self.nn_input_size % self.kTokenDim != 0:
            print(f"[Warning] nnInputSize ({self.nn_input_size}) is not divisible by "
                  f"kTokenDim ({self.kTokenDim}). Check your meta.json.")

        # Architecture hyperparameters
        self.d_model = config["network"].get("dModel", 256)
        self.n_heads = config["network"].get("nHeads", 16)
        self.n_layers = config["network"].get("nLayers", 8)
        self.dim_ff  = config["network"].get("dimFeedforward", 512)

        # Embedding & positional encoding
        self.embedding   = nn.Linear(self.kTokenDim, self.d_model)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model) * 0.02
        )

        # Transformer trunk (dropout locked at 0 for RL — no regularisation needed here)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_ff,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Policy head — returns raw logits (Softmax applied at inference / in ONNX wrapper)
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.action_space),
        )

        # Value head — one scalar per player, range [-1, 1]
        # Game-agnostic: works for 2-player zero-sum, 2-player non-zero-sum, N-player, etc.
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_players),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(-1, self.seq_len, self.kTokenDim)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        global_repr  = x.mean(dim=1)                  # mean pooling over sequence
        policy_logits = self.policy_head(global_repr)
        value_pred    = self.value_head(global_repr)
        return policy_logits, value_pred


# ==============================================================================
# --- 2. EXPORT & COMPILATION PIPELINE ---
# ==============================================================================

class ONNXExportWrapper(nn.Module):
    """Wraps the base model to bake the Softmax into the ONNX graph for C++ inference."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        policy_logits, value_pred = self.base_model(x)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        return policy_probs, value_pred


def export_to_onnx(model, meta, save_path):
    print(f"\n[Export] Saving ONNX model → {save_path}")
    model.eval()
    export_model = ONNXExportWrapper(model)
    export_model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, meta["nnInputSize"]).to(device)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_state"],
        output_names=["policy_output", "value_output"],
        dynamic_axes={
            "input_state":    {0: "batch_size"},
            "policy_output":  {0: "batch_size"},
            "value_output":   {0: "batch_size"},
        },
    )
    print(f"[Export] ONNX saved successfully.")


def compile_tensorrt_engine(onnx_path, plan_path, meta, opt_batch, precision):
    print(f"\n[TensorRT] Compiling engine  precision={precision.upper()}  opt_batch={opt_batch} ...")
    nn_input_size = meta["nnInputSize"]

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--minShapes=input_state:1x{nn_input_size}",
        f"--optShapes=input_state:{opt_batch}x{nn_input_size}",
        f"--maxShapes=input_state:{opt_batch}x{nn_input_size}",
    ]
    if precision.lower() == "fp16":
        cmd.append("--fp16")

    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        print("[TensorRT] Engine compiled successfully.")
    except subprocess.CalledProcessError as e:
        print("\n[Fatal] TensorRT compilation failed.")
        print(f"TRT stderr:\n{e.stderr}")
        sys.exit(1)


# ==============================================================================
# --- 3. CHECKPOINT HELPERS ---
# ==============================================================================

def save_checkpoint(path, model, optimizer, global_step, iteration):
    """Save a full training checkpoint (model + optimizer state + counters)."""
    torch.save(
        {
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "global_step": global_step,
            "iteration":   iteration,
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}  (iter={iteration}, global_step={global_step})")


def load_checkpoint(path, model, optimizer, device):
    """
    Load a full checkpoint.  Falls back gracefully to a weights-only file
    (legacy format saved by earlier versions of this script).
    Returns (global_step, iteration).
    """
    raw = torch.load(path, map_location=device, weights_only=False)

    # Full checkpoint dict
    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"])
        if optimizer is not None and "optimizer" in raw and raw["optimizer"] is not None:
            try:
                optimizer.load_state_dict(raw["optimizer"])
                print("[Checkpoint] Optimizer state restored.")
            except Exception as e:
                print(f"[Checkpoint] Warning: could not restore optimizer state ({e}). "
                      "Starting optimizer from scratch.")
        global_step = raw.get("global_step", 0)
        iteration   = raw.get("iteration", 0)
        print(f"[Checkpoint] Loaded  iter={iteration}  global_step={global_step}")
        return global_step, iteration

    # Legacy: raw state dict (model weights only)
    print("[Checkpoint] Legacy weights-only file detected. Optimizer starts from scratch.")
    model.load_state_dict(raw)
    return 0, 0


# ==============================================================================
# --- 4. TRAINING LOOP ---
# ==============================================================================

def run_training(config_path: str):
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    game_name = config["name"]
    data_dir  = Path(f"data/{game_name}")

    # ------------------------------------------------------------------
    # 2. Resolve dataset files — sorted so the sliding window is stable
    # ------------------------------------------------------------------
    bin_files = sorted(glob.glob(str(data_dir / "iteration_*.bin")))

    if not bin_files:
        print(f"[Error] No dataset files found in {data_dir}. Run Self-Play first!")
        return

    # Safety net: enforce the sliding window directly in the trainer.
    # The orchestrator already handles deletion, but this double-check
    # prevents silently bloated buffers if the orchestrator is bypassed.
    hp = config["training"]
    max_window_files = hp.get("maxWindowIterations", 10)
    if len(bin_files) > max_window_files:
        discarded = len(bin_files) - max_window_files
        print(f"[Train] Sliding window: keeping {max_window_files} most recent files "
              f"(discarding {discarded} oldest from this run — files NOT deleted on disk).")
        bin_files = bin_files[-max_window_files:]

    # ------------------------------------------------------------------
    # 3. Load metadata
    # ------------------------------------------------------------------
    meta_path = data_dir / f"{game_name}_training_data.bin.meta.json"
    if not meta_path.exists():
        print(f"[Error] Meta file missing at {meta_path}. Run MetaExport first!")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # ------------------------------------------------------------------
    # 4. Hyperparameters
    # ------------------------------------------------------------------
    train_batch_size  = hp.get("trainBatchSize", 1024)
    epochs            = hp.get("epochs", 1)
    lr                = float(hp.get("learningRate", 1e-3))
    lr_min_factor     = float(hp.get("lrMinFactor", 0.1))   # cosine decay floor = lr * factor
    weight_decay      = float(hp.get("weightDecay", 1e-4))
    # FIX: valueLossWeight scales the VALUE loss, not the policy loss.
    # A value of 0.25 matches Lc0's recommendation (prevents value head overfitting).
    value_loss_weight = float(hp.get("valueLossWeight", 0.25))
    current_iteration = hp.get("currentIteration", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"[Train] Game: {game_name}  |  Iteration: {current_iteration}")
    print(f"[Train] Device: {device}  |  Batch: {train_batch_size}  |  Epochs: {epochs}")
    print(f"[Train] LR: {lr}  |  valueLossWeight: {value_loss_weight}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 5. Dataset & DataLoader
    # ------------------------------------------------------------------
    print(f"[Train] Loading {len(bin_files)} iteration file(s)...")
    datasets     = [OneMindArmyDataset(f) for f in bin_files]
    full_dataset = ConcatDataset(datasets)
    print(f"[Train] Total samples in replay buffer: {len(full_dataset):,}")

    dataloader = DataLoader(
        full_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # 6. Model & Optimizer
    # ------------------------------------------------------------------
    model_dir      = Path(f"models/{game_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "latest_checkpoint.pt"   # full checkpoint
    onnx_path       = model_dir / "latest_model.onnx"
    plan_path       = model_dir / "latest_model.plan"

    model     = OneMindArmyNet(config, meta).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    if checkpoint_path.exists():
        print(f"[Train] Resuming from checkpoint: {checkpoint_path.name}")
        global_step, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
    else:
        print("[Train] No checkpoint found. Starting from random weights.")

    # ------------------------------------------------------------------
    # 7. LR Scheduler — cosine decay within the epoch
    #    Each training session gets a fresh cosine cycle.
    #    The optimizer state (m, v) is preserved across restarts.
    # ------------------------------------------------------------------
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=lr * lr_min_factor,
    )

    mse_criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        epoch_v_loss  = 0.0
        epoch_p_loss  = 0.0
        epoch_start   = time.time()

        for batch_idx, (states, target_policies, legal_masks, target_results) in enumerate(dataloader):
            states          = states.to(device, non_blocking=True)
            target_policies = target_policies.to(device, non_blocking=True)
            legal_masks     = legal_masks.to(device, non_blocking=True)
            target_results  = target_results.to(device, non_blocking=True)

            # Normalize policy targets (visit counts → probabilities)
            target_policies = target_policies / (
                target_policies.sum(dim=-1, keepdim=True) + 1e-9
            )

            # Forward pass
            policy_logits, pred_values = model(states)

            # Mask illegal moves with -inf before softmax
            illegal_mask  = (legal_masks == 0)
            policy_logits = policy_logits.masked_fill(illegal_mask, -1e9)
            pred_log_probs = torch.log_softmax(policy_logits, dim=-1)

            # -------------------------------------------------------
            # LOSS COMPUTATION
            # FIX: valueLossWeight is applied to the VALUE loss.
            #   total = (w * v_loss) + p_loss
            # The policy gradient flows fully; the value head is regularised.
            # This prevents the value head from dominating the trunk gradients.
            # -------------------------------------------------------
            v_loss     = mse_criterion(pred_values, target_results)
            p_loss     = -torch.sum(target_policies * pred_log_probs, dim=1).mean()
            total_loss = (value_loss_weight * v_loss) + p_loss

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step  += 1
            epoch_v_loss += v_loss.item()
            epoch_p_loss += p_loss.item()

            if batch_idx % 100 == 0 and batch_idx > 0:
                elapsed        = time.time() - epoch_start
                samples_per_s  = (batch_idx * train_batch_size) / elapsed
                current_lr     = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx:>6}/{len(dataloader)}] | "
                    f"V-Loss: {v_loss.item():.4f}  "
                    f"P-Loss: {p_loss.item():.4f}  "
                    f"GradNorm: {grad_norm:.3f}  "
                    f"LR: {current_lr:.2e}  "
                    f"{samples_per_s:,.0f} spl/s"
                )

        avg_v_loss = epoch_v_loss / len(dataloader)
        avg_p_loss = epoch_p_loss / len(dataloader)
        elapsed    = time.time() - epoch_start
        print(
            f"\n>>> End of Epoch {epoch+1}/{epochs} | "
            f"Avg V-Loss: {avg_v_loss:.4f} | "
            f"Avg P-Loss: {avg_p_loss:.4f} | "
            f"Time: {elapsed:.1f}s\n"
        )

    # ------------------------------------------------------------------
    # 9. Save checkpoint + export
    # ------------------------------------------------------------------
    save_checkpoint(checkpoint_path, model, optimizer, global_step, current_iteration)

    export_to_onnx(model, meta, str(onnx_path))
    compile_tensorrt_engine(
        str(onnx_path),
        str(plan_path),
        meta,
        config["backend"].get("inferenceBatchSize", 1024),
        config["backend"].get("precision", "fp16"),
    )

    print(f"\n[Train] Cycle complete. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneMindArmy Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_training(args.config)
