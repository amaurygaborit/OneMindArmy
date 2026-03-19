import os
import sys
import yaml
import json
import argparse
import subprocess
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# --- 1. TRAINING LOGGER ---
# ==============================================================================

class TrainingLogger:
    def __init__(self, log_path: Path, log_every: int = 100):
        self.log_path  = log_path
        self.log_every = log_every
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", buffering=1)

    def _write(self, record: dict):
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def log_batch(self, *, iteration, epoch, global_step, batch_idx,
                  v_loss, p_loss, total_loss, grad_norm, lr, samples_per_s):
        if batch_idx % self.log_every != 0:
            return
        self._write({
            "type":          "batch",
            "iteration":     iteration,
            "epoch":         epoch,
            "global_step":   global_step,
            "batch_idx":     batch_idx,
            "v_loss":        round(v_loss,      6),
            "p_loss":        round(p_loss,      6),
            "total_loss":    round(total_loss,  6),
            "grad_norm":     round(grad_norm,   6),
            "lr":            round(lr,          8),
            "samples_per_s": round(samples_per_s, 1),
        })

    def log_epoch(self, *, iteration, epoch, avg_v_loss, avg_p_loss,
                  avg_total_loss, time_s, num_samples, num_batches):
        self._write({
            "type":           "epoch",
            "iteration":      iteration,
            "epoch":          epoch,
            "avg_v_loss":     round(avg_v_loss,     6),
            "avg_p_loss":     round(avg_p_loss,     6),
            "avg_total_loss": round(avg_total_loss, 6),
            "time_s":         round(time_s,         2),
            "num_samples":    num_samples,
            "num_batches":    num_batches,
        })

    def close(self):
        self._f.close()


# ==============================================================================
# --- 2. NEURAL NETWORK ARCHITECTURE (TRANSFORMER) ---
# ==============================================================================

class OneMindArmyNet(nn.Module):
    def __init__(self, config, meta):
        super().__init__()

        self.action_space  = meta["actionSpace"]
        self.num_players   = meta["numPlayers"]
        self.num_pos       = meta["numPos"]
        self.nn_input_size = meta["nnInputSize"]

        self.kTokenDim = 4 + self.num_pos
        self.seq_len   = self.nn_input_size // self.kTokenDim

        if self.nn_input_size % self.kTokenDim != 0:
            print(f"[Warning] nnInputSize ({self.nn_input_size}) is not divisible by "
                  f"kTokenDim ({self.kTokenDim}).")

        self.d_model  = config["network"].get("dModel", 256)
        self.n_heads  = config["network"].get("nHeads", 16)
        self.n_layers = config["network"].get("nLayers", 8)
        self.dim_ff   = config["network"].get("dimFeedforward", 512)

        self.embedding   = nn.Linear(self.kTokenDim, self.d_model)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.dim_ff,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.action_space),
        )

        # NEW WDL HEAD: Output is [Batch, num_players * 3] raw logits.
        # No Tanh/Softmax here, it will be handled by CrossEntropyLoss and the ONNX Wrapper.
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_players * 3), 
        )

    def forward(self, x):
        x = x.view(-1, self.seq_len, self.kTokenDim)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        global_repr   = x.mean(dim=1)
        
        policy_logits = self.policy_head(global_repr)
        wdl_logits    = self.value_head(global_repr)
        
        return policy_logits, wdl_logits


# ==============================================================================
# --- 3. EXPORT & COMPILATION ---
# ==============================================================================

class ONNXExportWrapper(nn.Module):
    def __init__(self, base_model, num_players):
        super().__init__()
        self.base_model = base_model
        self.num_players = num_players

    def forward(self, x):
        policy_logits, wdl_logits = self.base_model(x)
        
        # Policy to Probabilities
        policy_probs = torch.softmax(policy_logits, dim=-1)
        
        # WDL to Probabilities (per player)
        # Reshape to [Batch, Players, 3], apply softmax over the 3 classes, then flatten back
        batch_size = wdl_logits.size(0)
        wdl_reshaped = wdl_logits.view(batch_size, self.num_players, 3)
        wdl_probs = torch.softmax(wdl_reshaped, dim=-1)
        wdl_out = wdl_probs.view(batch_size, self.num_players * 3)
        
        return policy_probs, wdl_out


def export_to_onnx(model, meta, save_path):
    print(f"\n[Export] Saving ONNX model → {save_path}")
    model.eval()
    export_model = ONNXExportWrapper(model, meta["numPlayers"])
    export_model.eval()
    device      = next(model.parameters()).device
    dummy_input = torch.randn(1, meta["nnInputSize"]).to(device)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model, dummy_input, save_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["input_state"], output_names=["policy_output", "value_output"],
        dynamic_axes={
            "input_state":   {0: "batch_size"},
            "policy_output": {0: "batch_size"},
            "value_output":  {0: "batch_size"},
        },
    )
    print("[Export] ONNX saved successfully.")


def compile_tensorrt_engine(onnx_path, plan_path, meta, opt_batch, precision):
    print(f"\n[TensorRT] Compiling engine  precision={precision.upper()}  opt_batch={opt_batch} ...")
    s = meta["nnInputSize"]
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}", f"--saveEngine={plan_path}",
        f"--minShapes=input_state:1x{s}",
        f"--optShapes=input_state:{opt_batch}x{s}",
        f"--maxShapes=input_state:{opt_batch}x{s}",
    ]
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, encoding="utf-8")
        print("[TensorRT] Engine compiled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[Fatal] TensorRT compilation failed.\n{e.stderr}")
        sys.exit(1)


# ==============================================================================
# --- 4. CHECKPOINT HELPERS ---
# ==============================================================================

def save_checkpoint(path, model, optimizer, global_step, iteration):
    torch.save({
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "global_step": global_step,
        "iteration":   iteration,
    }, path)
    print(f"[Checkpoint] Saved → {path}  (iter={iteration}, global_step={global_step})")


def load_checkpoint(path, model, optimizer, device):
    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"])
        if optimizer is not None and raw.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(raw["optimizer"])
                print("[Checkpoint] Optimizer state restored.")
            except Exception as e:
                print(f"[Checkpoint] Warning: optimizer state incompatible ({e}). Starting fresh.")
        gs  = raw.get("global_step", 0)
        itr = raw.get("iteration",   0)
        print(f"[Checkpoint] Loaded  iter={itr}  global_step={gs}")
        return gs, itr

    print("[Checkpoint] Legacy weights-only format. Optimizer starts from scratch.")
    model.load_state_dict(raw)
    return 0, 0


# ==============================================================================
# --- 5. TRAINING LOOP ---
# ==============================================================================

def run_training(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    game_name = config["name"]
    data_dir  = Path(f"data/{game_name}")

    bin_files = sorted(glob.glob(str(data_dir / "iteration_*.bin")))
    if not bin_files:
        print(f"[Error] No dataset files found in {data_dir}. Run Self-Play first!")
        return

    hp = config["training"]
    max_window_files = hp.get("maxWindowIterations", 10)
    if len(bin_files) > max_window_files:
        n = len(bin_files) - max_window_files
        print(f"[Train] Sliding window: discarding {n} oldest file(s) from this run.")
        bin_files = bin_files[-max_window_files:]

    meta_path = data_dir / f"{game_name}_training_data.bin.meta.json"
    if not meta_path.exists():
        print(f"[Error] Meta file missing at {meta_path}.")
        return
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    num_players = meta["numPlayers"]

    train_batch_size  = hp.get("trainBatchSize",   1024)
    epochs            = hp.get("epochs",           1)
    lr                = float(hp.get("learningRate",  1e-3))
    lr_min_factor     = float(hp.get("lrMinFactor",   0.1))
    weight_decay      = float(hp.get("weightDecay",   1e-4))
    value_loss_weight = float(hp.get("valueLossWeight", 0.5)) # Slightly higher for WDL
    current_iteration = hp.get("currentIteration",  0)
    log_every         = hp.get("logEveryNBatches",  100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"[Train] Game: {game_name}  |  Iteration: {current_iteration}")
    print(f"[Train] Device: {device}  |  Batch: {train_batch_size}  |  Epochs: {epochs}")
    print(f"[Train] LR: {lr}  |  valueLossWeight: {value_loss_weight}")
    print(f"{'='*60}\n")

    log_path = data_dir / "training_log.jsonl"
    tlogger  = TrainingLogger(log_path, log_every=log_every)

    print(f"[Train] Loading {len(bin_files)} iteration file(s)...")
    full_dataset = ConcatDataset([OneMindArmyDataset(f) for f in bin_files])
    print(f"[Train] Total samples: {len(full_dataset):,}")

    dataloader = DataLoader(
        full_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model_dir       = Path(f"models/{game_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "latest_checkpoint.pt"
    onnx_path       = model_dir / "latest_model.onnx"
    plan_path       = model_dir / "latest_model.plan"

    model     = OneMindArmyNet(config, meta).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    if checkpoint_path.exists():
        print(f"[Train] Resuming from: {checkpoint_path.name}")
        global_step, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        print("[Train] No checkpoint — starting from random weights.")

    total_steps = len(dataloader) * epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * lr_min_factor,
    )

    for epoch in range(epochs):
        model.train()
        sum_v = sum_p = sum_t = 0.0
        epoch_start = time.time()

        for batch_idx, (states, target_policies, legal_masks, target_wdl) in enumerate(dataloader):
            states          = states.to(device,          non_blocking=True)
            target_policies = target_policies.to(device, non_blocking=True)
            legal_masks     = legal_masks.to(device,     non_blocking=True)
            target_wdl      = target_wdl.to(device,      non_blocking=True) # Shape: [Batch, Players, 3]

            target_policies = target_policies / (
                target_policies.sum(dim=-1, keepdim=True) + 1e-9
            )

            policy_logits, wdl_logits = model(states)
            
            # --- POLICY LOSS ---
            policy_logits  = policy_logits.masked_fill(legal_masks == 0, -1e9)
            pred_log_probs = torch.log_softmax(policy_logits, dim=-1)
            p_loss         = -torch.sum(target_policies * pred_log_probs, dim=1).mean()

            # --- VALUE LOSS (WDL Multi-Player Cross Entropy) ---
            # wdl_logits is [Batch, Players * 3] -> reshape to [Batch, Players, 3]
            wdl_logits_reshaped = wdl_logits.view(-1, num_players, 3)
            
            v_loss = 0.0
            for p in range(num_players):
                # We use cross_entropy with soft labels (target_wdl is probabilities)
                v_loss += F.cross_entropy(wdl_logits_reshaped[:, p, :], target_wdl[:, p, :])
            
            # Average the WDL loss across all players
            v_loss = v_loss / num_players

            # --- TOTAL LOSS ---
            total_loss = (value_loss_weight * v_loss) + p_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            sum_v += v_loss.item()
            sum_p += p_loss.item()
            sum_t += total_loss.item()

            current_lr    = scheduler.get_last_lr()[0]
            elapsed       = time.time() - epoch_start
            samples_per_s = ((batch_idx + 1) * train_batch_size) / max(elapsed, 1e-6)

            if batch_idx > 0 and batch_idx % log_every == 0:
                print(
                    f"  Ep[{epoch+1}/{epochs}] "
                    f"Batch[{batch_idx:>6}/{len(dataloader)}] | "
                    f"V(wdl):{v_loss.item():.4f}  P:{p_loss.item():.4f}  "
                    f"Norm:{grad_norm:.3f}  LR:{current_lr:.2e}  "
                    f"{samples_per_s:,.0f} spl/s"
                )

            tlogger.log_batch(
                iteration=current_iteration, epoch=epoch + 1,
                global_step=global_step, batch_idx=batch_idx,
                v_loss=v_loss.item(), p_loss=p_loss.item(),
                total_loss=total_loss.item(), grad_norm=float(grad_norm),
                lr=current_lr, samples_per_s=samples_per_s,
            )

        n            = len(dataloader)
        avg_v        = sum_v / n
        avg_p        = sum_p / n
        avg_t        = sum_t / n
        elapsed      = time.time() - epoch_start

        print(f"\n>>> Iter {current_iteration} | Epoch {epoch+1}/{epochs} | "
              f"Avg V-Loss: {avg_v:.4f} | Avg P-Loss: {avg_p:.4f} | "
              f"Time: {elapsed:.1f}s\n")

        tlogger.log_epoch(
            iteration=current_iteration, epoch=epoch + 1,
            avg_v_loss=avg_v, avg_p_loss=avg_p, avg_total_loss=avg_t,
            time_s=elapsed, num_samples=len(full_dataset), num_batches=n,
        )

    tlogger.close()

    save_checkpoint(checkpoint_path, model, optimizer, global_step, current_iteration)
    export_to_onnx(model, meta, str(onnx_path))
    compile_tensorrt_engine(
        str(onnx_path), str(plan_path), meta,
        config["backend"].get("inferenceBatchSize", 1024),
        config["backend"].get("precision", "fp16"),
    )
    print(f"\n[Train] Done. Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneMindArmy Trainer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_training(args.config)