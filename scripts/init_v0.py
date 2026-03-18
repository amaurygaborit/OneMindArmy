import os
import sys
import yaml
import json
import argparse
import subprocess
import torch
from pathlib import Path

# Import from our training script (SSOT for architecture + export helpers)
from train import OneMindArmyNet, export_to_onnx, compile_tensorrt_engine


def bootstrap_v0(config_path: str):
    """
    One-time initialization pipeline:
      1. Call C++ engine to export game dimension constants (SSOT).
      2. Build a random PyTorch model with those dimensions.
      3. Save it as `latest_checkpoint.pt` so train.py can resume from it.
      4. Export to ONNX + compile TensorRT engine (`best_model.plan`).

    After this runs, the orchestrator can start self-play immediately.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    game_name     = config["name"]
    trt_opt_batch = config["backend"].get("inferenceBatchSize", 256)
    trt_precision = config["backend"].get("precision", "fp16").lower()

    print("=" * 60)
    print(f"  Bootstrapping v0 random model  —  game: [{game_name}]")
    print(f"  Batch: {trt_opt_batch}  |  Precision: {trt_precision.upper()}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Call C++ engine to export metadata (game dimensions)
    # ------------------------------------------------------------------
    print("\n[Init] Calling C++ engine to export constexpr game dimensions...")

    exe_name = "onemindarmy.exe" if os.name == "nt" else "onemindarmy"
    possible_paths = [
        Path(f"build/x64-Release/bin/{exe_name}"),
        Path(f"build/bin/Release/{exe_name}"),
        Path(f"build/bin/{exe_name}"),
        Path(f"bin/{exe_name}"),
    ]

    cpp_executable = next((p for p in possible_paths if p.exists()), None)
    if not cpp_executable:
        print(f"\n[Fatal] C++ executable '{exe_name}' not found.")
        print(f"  Searched: {[str(p) for p in possible_paths]}")
        print("  Please compile the C++ project first.")
        sys.exit(1)

    try:
        subprocess.run(
            [str(cpp_executable), config_path, "--mode", "export-meta"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[Fatal] C++ engine failed to export metadata (exit code {e.returncode}).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Read generated metadata
    # ------------------------------------------------------------------
    data_dir  = Path(f"data/{game_name}")
    meta_path = data_dir / f"{game_name}_training_data.bin.meta.json"

    if not meta_path.exists():
        print(f"\n[Fatal] Expected meta file not found at: {meta_path}")
        print("  Check that the C++ export-meta mode writes to the correct path.")
        sys.exit(1)

    print(f"\n[Init] Reading metadata from {meta_path} ...")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(
        f"[Init] Dimensions: nnInputSize={meta['nnInputSize']}  "
        f"actionSpace={meta['actionSpace']}  "
        f"numPlayers={meta['numPlayers']}"
    )

    # ------------------------------------------------------------------
    # 3. Build the v0 random model
    # ------------------------------------------------------------------
    model_dir = Path(f"models/{game_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "latest_checkpoint.pt"
    onnx_path       = model_dir / "latest_model.onnx"
    best_plan_path  = model_dir / "best_model.plan"

    print("\n[Init] Building PyTorch Transformer with random (Xavier/Kaiming) weights ...")
    model = OneMindArmyNet(config, meta)

    # ------------------------------------------------------------------
    # 4. Save as latest_checkpoint.pt
    #    This ensures that train.py (iteration 1) starts from exactly the
    #    same weights that were compiled into best_model.plan for self-play.
    #    Without this, train.py would create a second independent random
    #    model, making the v0 TRT engine inconsistent with training.
    # ------------------------------------------------------------------
    if checkpoint_path.exists():
        print(f"[Init] Checkpoint already exists at {checkpoint_path}. Skipping save to avoid overwrite.")
    else:
        torch.save(
            {
                "model":       model.state_dict(),
                "optimizer":   None,   # no optimizer state yet
                "global_step": 0,
                "iteration":   0,
            },
            checkpoint_path,
        )
        print(f"[Init] Checkpoint saved → {checkpoint_path}")

    # ------------------------------------------------------------------
    # 5. Export ONNX + compile TensorRT engine (best_model.plan)
    # ------------------------------------------------------------------
    export_to_onnx(model, meta, str(onnx_path))
    compile_tensorrt_engine(
        str(onnx_path),
        str(best_plan_path),
        meta,
        trt_opt_batch,
        precision=trt_precision,
    )

    print(f"\n[Init] v0 bootstrap complete.")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  TRT engine : {best_plan_path}")
    print("  You can now start the Orchestrator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneMindArmy — v0 Bootstrapper")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    bootstrap_v0(args.config)
