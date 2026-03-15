import os
from pyexpat import model
import sys
import yaml
import json
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from torch.utils.data import DataLoader

# Import local de ton dataset personnalisé
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
        
        # Metadata extraction
        self.action_space = meta["actionSpace"]
        self.num_players = meta["numPlayers"]
        self.num_pos = meta["numPos"]
        self.nn_input_size = meta["nnInputSize"]
        
        # Token geometry
        self.kTokenDim = 4 + self.num_pos 
        self.seq_len = self.nn_input_size // self.kTokenDim

        if self.nn_input_size % self.kTokenDim != 0:
            print(f"[Warning] nnInputSize ({self.nn_input_size}) is not divisible by kTokenDim!")
        
        # Architecture hyperparameters (From YAML)
        self.d_model = config["network"].get("dModel", 256)
        self.n_heads = config["network"].get("nHeads", 16)
        self.n_layers = config["network"].get("nLayers", 8)
        self.dim_ff = config["network"].get("dimFeedforward", 512)
        
        # Embedding & Positioning
        self.embedding = nn.Linear(self.kTokenDim, self.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)
        
        # Transformer Core (DROPOUT VERROUILLÉ À 0.0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.n_heads, 
            dim_feedforward=self.dim_ff, 
            dropout=0.0, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Policy Head (Output Logits)
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.action_space)
        )
        
        # Value Head (Output Scalar per player)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_players),
            nn.Tanh() 
        )

    def forward(self, x):
        # Reshape flat input into sequence of tokens
        x = x.view(-1, self.seq_len, self.kTokenDim)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        
        # Global Pooling (Mean representation)
        global_repr = x.mean(dim=1)
        
        # Policy output (Log Softmax for KL Divergence or NLL)
        policy_logits = self.policy_head(global_repr)
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        
        # Value output (Predicted game outcome)
        value_pred = self.value_head(global_repr)
        
        return policy_log_probs, value_pred


# ==============================================================================
# --- 2. EXPORT & COMPILATION PIPELINE ---
# ==============================================================================
def export_to_onnx(model, meta, save_path):
    print(f"\n[Export] Saving ONNX model to: {save_path}")
    model.eval()
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, meta["nnInputSize"]).to(device)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True, 
        opset_version=17, 
        do_constant_folding=True, 
        input_names=['input_state'], 
        output_names=['policy_output', 'value_output'],
        dynamic_axes={
            'input_state': {0: 'batch_size'}, 
            'policy_output': {0: 'batch_size'}, 
            'value_output': {0: 'batch_size'}
        }
    )

def compile_tensorrt_engine(onnx_path, plan_path, meta, opt_batch, precision):
    print(f"\n[TensorRT] Compiling Engine (Precision: {precision.upper()}, OptBatch: {opt_batch})...")
    
    nn_input_size = meta["nnInputSize"]
    
    # 1. On crée la liste de base avec les arguments obligatoires
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--minShapes=input_state:1x{nn_input_size}",
        f"--optShapes=input_state:{opt_batch}x{nn_input_size}",
        f"--maxShapes=input_state:{opt_batch}x{nn_input_size}"
    ]
    
    # 2. On ajoute les options conditionnelles uniquement si nécessaire
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    
    # Si tu veux le mode verbose pour débugger l'erreur actuelle, décommente ceci :
    # cmd.append("--verbose")
    
    try:
        # On utilise stderr=subprocess.PIPE pour capturer les erreurs détaillées de NVIDIA
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, encoding='utf-8')
        print(f"[TensorRT] Engine compiled successfully!")
    except subprocess.CalledProcessError as e:
        print("\n[Fatal] TensorRT compilation failed.")
        print(f"TRT Error Output: {e.stderr}") # Affiche l'erreur réelle de NVIDIA
        sys.exit(1)


# ==============================================================================
# --- 3. TRAINING LOOP ---
# ==============================================================================
def run_training(config_path: str):
    # 1. Load Configs
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    game_name = config["name"]
    dataset_path = Path(f"data/{game_name}/{game_name}_training_data.bin")
    meta_path = dataset_path.with_suffix(".bin.meta.json")
    
    if not dataset_path.exists() or not meta_path.exists():
        print(f"[Error] Dataset or Meta file missing at {dataset_path}")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Output Paths
    model_dir = Path(f"models/{game_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = model_dir / "latest_model.pt"
    onnx_path = model_dir / "latest_model.onnx"
    plan_path = model_dir / "latest_model.plan"

    # Hyperparameters
    hp = config["training"]
    batch_size = hp.get("batchSize", 512)
    epochs = hp.get("epochs", 5)
    lr = float(hp.get("learningRate", 1e-3))
    weight_decay = float(hp.get("weightDecay", 1e-4))
    value_loss_weight = float(hp.get("valueLossWeight", 1.0))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device} | Iteration Batch: {batch_size}")

    # 2. Dataset & Loader
    dataset = OneMindArmyDataset(str(dataset_path))
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # 3. Model & Optimizer
    model = OneMindArmyNet(config, meta).to(device)
    
    if pt_path.exists():
        print(f"[Train] Resuming from existing weights: {pt_path.name}")
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()

    # 4. Loop
    for epoch in range(epochs):
        model.train()
        epoch_v_loss = 0.0
        epoch_p_loss = 0.0
        
        # CORRECTION 1: On lance le chrono au début de chaque epoch !
        epoch_start_time = time.time()
        
        for batch_idx, (states, target_policies, target_results) in enumerate(dataloader):
            states = states.to(device, non_blocking=True)
            target_policies = target_policies.to(device, non_blocking=True)
            legal_masks = legal_masks.to(device, non_blocking=True)
            target_results = target_results.to(device, non_blocking=True)

            # Normalization (Strict)
            target_policies = target_policies / (target_policies.sum(dim=-1, keepdim=True) + 1e-9)
            
            # Forward
            policy_logits, pred_values = model(states)

            bool_mask = (legal_masks < 0.5) 
            policy_logits = policy_logits.masked_fill(bool_mask, -1e9)
            pred_log_probs = torch.log_softmax(policy_logits, dim=-1)

            # Losses
            v_loss = mse_criterion(pred_values, target_results)
            p_loss = -torch.sum(target_policies * pred_log_probs, dim=1).mean()
            total_loss = v_loss + (value_loss_weight * p_loss)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_v_loss += v_loss.item()
            epoch_p_loss += p_loss.item()

            # On n'affiche pas au batch 0 car (0 * batch_size) / elapsed = 0
            if batch_idx % 100 == 0 and batch_idx > 0:
                elapsed = time.time() - epoch_start_time
                samples_per_sec = (batch_idx * batch_size) / elapsed
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] | "
                      f"V-Loss: {v_loss.item():.4f} | P-Loss: {p_loss.item():.4f} | {samples_per_sec:.0f} spl/s")

        # CORRECTION 2: On affiche la moyenne de l'epoch à la fin !
        avg_v_loss = epoch_v_loss / len(dataloader)
        avg_p_loss = epoch_p_loss / len(dataloader)
        print(f">>> End of Epoch {epoch+1} | Avg V-Loss: {avg_v_loss:.4f} | Avg P-Loss: {avg_p_loss:.4f}\n")

    # 5. Save & Export
    torch.save(model.state_dict(), pt_path)
    export_to_onnx(model, meta, str(onnx_path))
    
    compile_tensorrt_engine(
        str(onnx_path), 
        str(plan_path), 
        meta, 
        config["backend"].get("maxBatchSize", 1024), 
        config["backend"].get("precision", "fp16")
    )

    print(f"\n[Pipeline] Training Cycle Finished. Model: {pt_path.parent.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneMindArmy Trainer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_training(args.config)