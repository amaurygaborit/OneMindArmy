"""
compile_engine.py — TensorRT Engine Compiler
=========================================================

Design Intent:
Acts as a bridge between the PyTorch training loop (ONNX export) and the 
C++ execution engine (TensorRT). Abstracts away the verbose `trtexec` CLI 
parameters, automatically resolving tensor dimensions from the C++ metadata 
to guarantee shape alignment.
"""

import argparse
import json
import subprocess
import sys
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_meta(game_name: str) -> dict:
    # Extracts the Single Source of Truth (SSOT) for tensor dimensions generated 
    # directly by the C++ engine to prevent architecture mismatch.
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
    Invokes NVIDIA's trtexec to compile the computational graph.
    The input node "input_state" strictly binds to the ONNX export configuration 
    defined in the training loop. Dynamic batching is configured to optimize 
    memory allocation up to `opt_batch`.
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
        subprocess.run(
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
        sys.exit(1)
    except FileNotFoundError:
        print("[Fatal] 'trtexec' not found in PATH.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compile an ONNX model into a TensorRT .plan engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config",       type=str, default=None)
    parser.add_argument("--game",         type=str, default=None)
    parser.add_argument("--onnx",         type=str, default=None)
    parser.add_argument("--plan",         type=str, default=None)
    parser.add_argument("--nn-input-size", type=int, default=None)
    parser.add_argument("--batch",        type=int, default=None)
    parser.add_argument("--precision",    type=str, default=None, choices=["fp16", "fp32"])

    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    game_name = args.game or config.get("name")

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
        plan_path = str(Path("models") / game_name / "best_model.plan")
    else:
        print("[Error] Provide --plan or --config (with a 'name' field).")
        sys.exit(1)

    if args.nn_input_size:
        nn_input_size = args.nn_input_size
    elif game_name:
        meta = load_meta(game_name)
        nn_input_size = meta["nnInputSize"]
    else:
        print("[Error] Provide --nn-input-size or --config with a valid game name.")
        sys.exit(1)

    if args.batch:
        opt_batch = args.batch
    elif "backend" in config:
        opt_batch = config["backend"].get("inferenceBatchSize", 1024)
    else:
        opt_batch = 1024

    if args.precision:
        precision = args.precision
    elif "backend" in config:
        precision = config["backend"].get("precision", "fp16")
    else:
        precision = "fp16"

    compile_engine(onnx_path, plan_path, nn_input_size, opt_batch, precision)

if __name__ == "__main__":
    main()