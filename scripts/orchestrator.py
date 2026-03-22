import os
import sys
import json
import time
import shutil
import logging
import platform
import subprocess
import argparse
import psutil
from pathlib import Path

# On utilise ruamel.yaml pour PRESERVER les commentaires et le formatage
try:
    from ruamel.yaml import YAML
except ImportError:
    print("[Fatal] Le module 'ruamel.yaml' est requis. Lancez: pip install ruamel.yaml")
    sys.exit(1)


# ==============================================================================
# --- LOGGER ---
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][Orchestrator] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# --- PIPELINE ---
# ==============================================================================

class OneMindArmyPipeline:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)

        self.yaml_handler = YAML()
        self.yaml_handler.preserve_quotes = True
        self.config = self._load_config()
        self.game_name = self.config.get("name", "unknown_game")

        # ------------------------------------------------------------------
        # Locate C++ binary
        # ------------------------------------------------------------------
        self.exe_name = "onemindarmy.exe" if platform.system() == "Windows" else "onemindarmy"
        possible_paths = [
            Path(f"build/bin/{self.exe_name}"),
            Path(f"build/bin/Release/{self.exe_name}"),
            Path(f"build/x64-Release/bin/{self.exe_name}"),
            Path(f"bin/{self.exe_name}"),
            Path(self.exe_name),
        ]
        self.cpp_engine_path = next((p for p in possible_paths if p.exists()), None)

        if not self.cpp_engine_path:
            logger.error(
                f"C++ executable '{self.exe_name}' not found. "
                "Please compile the C++ project first."
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Working directories
        # ------------------------------------------------------------------
        self.data_dir   = Path("data")   / self.game_name
        self.models_dir = Path("models") / self.game_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config helpers (Ruamel YAML)
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return self.yaml_handler.load(f)

    def update_config_iteration(self, iteration: int):
        """Inject currentIteration into YAML PRESERVING comments and layout."""
        cfg = self._load_config()
        if "training" not in cfg:
            cfg["training"] = {}
        cfg["training"]["currentIteration"] = iteration
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            self.yaml_handler.dump(cfg, f)
            
        # Keep in-memory copy in sync
        self.config = cfg

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def run_command(self, cmd_list: list, phase_name: str) -> float:
        cmd_str = " ".join(str(x) for x in cmd_list)
        logger.info(f"--- START : {phase_name} ---")
        logger.debug(f"Command: {cmd_str}")
        t0 = time.time()
        try:
            subprocess.run(cmd_list, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"FATAL FAILURE in phase [{phase_name}]. Exit code: {e.returncode}"
            )
            sys.exit(1)
        elapsed = time.time() - t0
        logger.info(f"--- DONE  : {phase_name}  ({elapsed:.1f}s) ---\n")
        return elapsed

    def wait_for_vram_cleanup(self, timeout_sec: int = 15) -> bool:
        if not shutil.which("nvidia-smi"):
            return True

        logger.info("[CleanUp] Waiting for GPU VRAM release...")
        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )
                if not result.strip():
                    logger.info(
                        f"[CleanUp] VRAM cleared ({time.time() - (deadline - timeout_sec):.1f}s)."
                    )
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        logger.warning(f"[CleanUp] Timeout ({timeout_sec}s). VRAM may still be in use.")
        return False

    def ensure_process_terminated(self):
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if proc.info["name"] == self.exe_name:
                    logger.warning(
                        f"[ZombieGuard] Killing leftover '{self.exe_name}' "
                        f"(PID {proc.info['pid']})..."
                    )
                    proc.kill()
                    proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # ------------------------------------------------------------------
    # Sliding window — Centralized Logic
    # ------------------------------------------------------------------

    def enforce_sliding_window(self):
        """
        Delete oldest files based on BOTH sample limit and maximum file count (iterations).
        Centralizes the sliding window logic here instead of doing it in train.py.
        """
        training_cfg = self.config.get("training", {})
        max_samples = training_cfg.get("slidingWindowSamples", 7_500_000)
        max_files = training_cfg.get("maxWindowIterations", 10)

        meta_files = sorted(self.data_dir.glob("*.meta.json"))
        if not meta_files:
            logger.warning("[SlidingWindow] meta.json not found — skipping window enforcement.")
            return

        canonical_meta = self.data_dir / f"{self.game_name}_training_data.bin.meta.json"
        meta_path = canonical_meta if canonical_meta.exists() else meta_files[0]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        struct_size = meta.get("sizeofTrainingSample")
        if not struct_size:
            logger.warning("[SlidingWindow] 'sizeofTrainingSample' missing in meta.json — skipping.")
            return

        bin_files = sorted(
            self.data_dir.glob("iteration_*.bin"), 
            key=lambda x: int(x.stem.split('_')[1])
        )
        if not bin_files:
            return

        file_stats = [(f, f.stat().st_size // struct_size) for f in bin_files]
        total_samples = sum(s for _, s in file_stats)

        logger.info(
            f"[SlidingWindow] State before cleanup: {total_samples:,} samples | {len(file_stats)} files. "
            f"(Limits: max_samples={max_samples:,}, max_files={max_files})"
        )

        deleted = 0
        # Condition de suppression centralisée: trop de samples OU trop de fichiers
        while (total_samples > max_samples or len(file_stats) > max_files) and len(file_stats) > 1:
            oldest_file, oldest_count = file_stats.pop(0)
            try:
                os.remove(oldest_file)
                total_samples -= oldest_count
                deleted += 1
                logger.info(
                    f"[SlidingWindow] Deleted {oldest_file.name} "
                    f"(-{oldest_count:,} samples → {total_samples:,} remaining, {len(file_stats)} files left)"
                )
            except OSError as e:
                logger.error(f"[SlidingWindow] Could not delete {oldest_file.name}: {e}")
                break

        if deleted:
            logger.info(f"[SlidingWindow] Cleanup complete. Removed {deleted} file(s).")

    # ------------------------------------------------------------------
    # Resume detection
    # ------------------------------------------------------------------

    def get_start_iteration(self) -> int:
        bin_files = sorted(self.data_dir.glob("iteration_*.bin"))
        if not bin_files:
            return 1

        last_stem = bin_files[-1].stem
        try:
            last_iter = int(last_stem.split("_")[1])
            logger.info(
                f"[AutoResume] Existing data found up to iteration {last_iter}. "
                f"Resuming from iteration {last_iter + 1}."
            )
            return last_iter + 1
        except (IndexError, ValueError):
            return 1

    # ------------------------------------------------------------------
    # Pipeline phases
    # ------------------------------------------------------------------

    def phase_bootstrap(self):
        cmd = [sys.executable, "scripts/init_v0.py", "--config", str(self.config_path)]
        self.run_command(cmd, "Bootstrap v0 (random model initialization)")

    def phase_self_play(self, model_name: str):
        cmd = [
            str(self.cpp_engine_path),
            str(self.config_path),
            "--mode", "train",
            "--model", model_name,
        ]
        self.run_command(cmd, f"Self-Play  (model: {model_name})")

    def phase_train(self):
        cmd = [sys.executable, "scripts/train.py", "--config", str(self.config_path)]
        self.run_command(cmd, "Neural Network Training + TRT Compilation")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(self, total_iterations: int):
        logger.info(
            f"\n{'='*60}\n"
            f"  OMA PIPELINE START  —  game: {self.game_name.upper()}\n"
            f"  Planned iterations: {total_iterations}\n"
            f"{'='*60}"
        )

        best_model_name = "best_model.plan"
        best_model_path  = self.models_dir / best_model_name
        latest_model_path = self.models_dir / "latest_model.plan"

        if not best_model_path.exists():
            logger.warning(f"No '{best_model_name}' found — running v0 initialization ...")
            self.phase_bootstrap()
            self.wait_for_vram_cleanup()

        pipeline_start = time.time()
        start_iter     = self.get_start_iteration()

        for iteration in range(start_iter, start_iter + total_iterations):
            iter_start = time.time()
            logger.info(
                f"\n{'='*60}\n"
                f"  ITERATION {iteration:04d}\n"
                f"{'='*60}"
            )

            self.update_config_iteration(iteration)

            self.phase_self_play(best_model_name)
            self.wait_for_vram_cleanup()

            self.enforce_sliding_window()

            self.phase_train()
            self.wait_for_vram_cleanup()

            if latest_model_path.exists():
                shutil.copy(latest_model_path, best_model_path)
                logger.info(
                    f"[ContinuousDeployment] Overwriting {best_model_name} with newly trained weights. "
                    f"(iteration {iteration} complete)"
                )
            else:
                logger.error("latest_model.plan was not produced by train.py!")
                sys.exit(1)

            iter_elapsed = time.time() - iter_start
            logger.info(f"[Iteration {iteration:04d}] Wall-clock time: {iter_elapsed/60:.1f} min")

        total_hours = (time.time() - pipeline_start) / 3600
        logger.info(
            f"\n{'='*60}\n"
            f"  PIPELINE FINISHED\n"
            f"  {total_iterations} iterations in {total_hours:.2f} hours\n"
            f"{'='*60}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneMindArmy — Orchestrator")
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of RL iterations to run (default: 100)"
    )
    args = parser.parse_args()

    pipeline = OneMindArmyPipeline(args.config)
    pipeline.ensure_process_terminated()
    pipeline.wait_for_vram_cleanup(timeout_sec=5)

    try:
        pipeline.run_loop(args.iterations)
    except KeyboardInterrupt:
        logger.warning("\n[Interrupted] Pipeline stopped by user. Cleaning up ...")
        pipeline.ensure_process_terminated()
        sys.exit(0)