import os
import sys
import yaml
import json
import argparse
import subprocess
from pathlib import Path

# Import logic directly from our training script
from train import OneMindArmyNet, export_to_onnx, compile_tensorrt_engine

def bootstrap_v0(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    game_name = config["name"]
    trt_opt_batch = config["backend"].get("inferenceBatchSize", 256)
    
    # NOUVEAU : On récupère la précision depuis le YAML (par défaut on met fp16 pour les RTX/GTX récentes)
    trt_precision = config["backend"].get("precision", "fp16").lower()
    
    print(f"============================================================")
    print(f"=== Bootstrapping v0 random model for [{game_name}] ===")
    print(f"=== Batch Size: {trt_opt_batch} | Precision: {trt_precision.upper()} ===")
    print(f"============================================================")
    
    # -------------------------------------------------------------------------
    # 1. CALL C++ TO EXPORT METADATA (SSOT)
    # -------------------------------------------------------------------------
    print("\n[Init] Calling C++ Engine to extract constexpr game dimensions...")
    
    exe_name = "onemindarmy.exe" if os.name == 'nt' else "onemindarmy"
    
    # Recherche robuste du binaire
    possible_paths = [
        Path(f"build/x64-Release/bin/{exe_name}"),
        Path(f"build/bin/{exe_name}")
    ]
    
    cpp_executable = next((p for p in possible_paths if p.exists()), None)
    
    if not cpp_executable:
        print(f"\n[Fatal Error] C++ Executable not found.")
        print(f"Checked paths: {[str(p) for p in possible_paths]}")
        print("Please compile your C++ project first!")
        sys.exit(1)
        
    try:
        subprocess.run([str(cpp_executable), config_path, "--mode", "export-meta"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Fatal Error] C++ Engine failed to export metadata. Exit code: {e.returncode}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. READ GENERATED METADATA
    # -------------------------------------------------------------------------
    data_dir = Path(f"data/{game_name}")
    meta_path = data_dir / f"{game_name}_training_data.bin.meta.json"
    
    print(f"\n[Init] Reading generated SSOT metadata from {meta_path}...")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    print(f"[Init] Dimensions loaded -> Input Size: {meta['nnInputSize']}, Actions: {meta['actionSpace']}")
    
    # -------------------------------------------------------------------------
    # 3. GENERATE PYTORCH MODEL & COMPILE TO TENSORRT
    # -------------------------------------------------------------------------
    model_dir = Path(f"models/{game_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Init] Generating PyTorch Transformer with Xavier/Kaiming random weights...")
    model = OneMindArmyNet(config, meta)
    
    onnx_save_path = str(model_dir / "latest_model.onnx")
    plan_save_path = str(model_dir / "best_model.plan") 
    
    export_to_onnx(model, meta, onnx_save_path)
    
    # NOUVEAU : On passe l'argument precision ici !
    compile_tensorrt_engine(onnx_save_path, plan_save_path, meta, trt_opt_batch, precision=trt_precision)
    
    print(f"\n[Success] v0 Bootstrapping complete! You can now start the Orchestrator.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    bootstrap_v0(args.config)