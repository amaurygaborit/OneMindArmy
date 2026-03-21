"""
compile_engine.py — Standalone TensorRT engine compiler
=========================================================

Compiles a .onnx model into a TensorRT .plan engine file,
ready to be used by the C++ self-play / inference engine.

Usage:
    # Compile with default settings from chess_train.yaml
    python compile_engine.py --config chess_train.yaml

    # Override precision and batch size
    python compile_engine.py --config chess_train.yaml --precision fp32 --batch 512

    # Specify paths explicitly (ignore config)
    python compile_engine.py --onnx models/chess/latest_model.onnx \\
                              --plan models/chess/best_model.plan   \\
                              --nn-input-size 4544 --batch 2048 --precision fp16

Arguments:
    --config        Path to YAML config (reads nnInputSize from meta.json,
                    inferenceBatchSize and precision from backend section)
    --onnx          Path to the .onnx file (overrides config path)
    --plan          Path for the output .plan engine (overrides config path)
    --nn-input-size Flat NN input size in floats (overrides meta.json)
    --batch         Inference batch size (overrides config)
    --precision     fp16 or fp32 (overrides config, default: fp16)
    --game          Game name (overrides config["name"])
"""

import argparse
import json
import subprocess
import sys
import yaml
from pathlib import Path


# ==============================================================================
# --- HELPERS ---
# ==============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_meta(game_name: str) -> dict:
    meta_path = Path("data") / game_name / f"{game_name}_training_data.bin.meta.json"
    if not meta_path.exists():
        print(f"[Error] Meta file not found: {meta_path}")
        print("  Run C++ export-meta first, or pass --nn-input-size manually.")
        sys.exit(1)
    with open(meta_path, "r") as f:
        return json.load(f)


def compile_engine(onnx_path: str,
                   plan_path: str,
                   nn_input_size: int,
                   opt_batch: int,
                   precision: str) -> None:
    """
    Calls trtexec to compile the ONNX model into a TensorRT .plan engine.

    Shape naming must match the ONNX export in train.py:
        input tensor name = "input_state"
        shape             = [batch_size, nn_input_size]
    """
    print(f"\n{'='*60}")
    print(f"  TensorRT Engine Compilation")
    print(f"{'='*60}")
    print(f"  ONNX           : {onnx_path}")
    print(f"  Plan output    : {plan_path}")
    print(f"  NN input size  : {nn_input_size}")
    print(f"  Batch size     : {opt_batch}")
    print(f"  Precision      : {precision.upper()}")
    print(f"{'='*60}\n")

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        print(f"[Error] ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Create output directory if needed
    Path(plan_path).parent.mkdir(parents=True, exist_ok=True)

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

    print(f"[CMD] {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        print(f"\n[OK] Engine compiled successfully → {plan_path}")

    except subprocess.CalledProcessError as e:
        print(f"\n[Fatal] trtexec failed (exit code {e.returncode})")
        print(f"\n--- trtexec stderr ---")
        print(e.stderr)
        print("----------------------")
        print("\nMake sure trtexec is in your PATH.")
        print("  On Linux with TensorRT installed:")
        print("    export PATH=$PATH:/usr/src/tensorrt/bin")
        sys.exit(1)

    except FileNotFoundError:
        print("[Fatal] 'trtexec' not found in PATH.")
        print("  Install TensorRT and add trtexec to your PATH.")
        sys.exit(1)


# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compile a .onnx model into a TensorRT .plan engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config-based arguments (optional if explicit paths are given)
    parser.add_argument("--config",       type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--game",         type=str, default=None,
                        help="Game name (overrides config['name'])")

    # Explicit path overrides
    parser.add_argument("--onnx",         type=str, default=None,
                        help="Path to the .onnx input file")
    parser.add_argument("--plan",         type=str, default=None,
                        help="Path for the output .plan engine file")

    # Compilation parameters
    parser.add_argument("--nn-input-size", type=int, default=None,
                        help="Flat NN input size in floats (overrides meta.json)")
    parser.add_argument("--batch",        type=int, default=None,
                        help="Inference batch size (overrides config)")
    parser.add_argument("--precision",    type=str, default=None,
                        choices=["fp16", "fp32"],
                        help="Precision mode (overrides config, default: fp16)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve all parameters
    # ------------------------------------------------------------------

    # 1. Load YAML config if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # 2. Game name
    game_name = args.game or config.get("name")

    # 3. Paths
    if args.onnx:
        onnx_path = args.onnx
    elif game_name:
        onnx_path = str(Path("models") / game_name / "latest_model.onnx")
    else:
        print("[Error] Provide --onnx or --config (with a 'name' field).")
        sys.exit(1)

    if args.plan:
        plan_path = args.plan
    elif game_name:
        # Default: compile as best_model.plan so the C++ engine can use it
        plan_path = str(Path("models") / game_name / "best_model.plan")
    else:
        print("[Error] Provide --plan or --config (with a 'name' field).")
        sys.exit(1)

    # 4. NN input size
    if args.nn_input_size:
        nn_input_size = args.nn_input_size
    elif game_name:
        meta = load_meta(game_name)
        nn_input_size = meta["nnInputSize"]
        print(f"[Info] nnInputSize = {nn_input_size} (from meta.json)")
    else:
        print("[Error] Provide --nn-input-size or --config with a valid game name.")
        sys.exit(1)

    # 5. Batch size
    if args.batch:
        opt_batch = args.batch
    elif "backend" in config:
        opt_batch = config["backend"].get("inferenceBatchSize", 1024)
    else:
        opt_batch = 1024
        print(f"[Info] No batch size specified, defaulting to {opt_batch}")

    # 6. Precision
    if args.precision:
        precision = args.precision
    elif "backend" in config:
        precision = config["backend"].get("precision", "fp16")
    else:
        precision = "fp16"

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    compile_engine(onnx_path, plan_path, nn_input_size, opt_batch, precision)


if __name__ == "__main__":
    main()