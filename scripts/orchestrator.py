import os
import sys
import yaml
import json
import time
import shutil
import logging
import platform
import subprocess
import argparse
import psutil
import glob
from pathlib import Path

# ==============================================================================
# --- LOGGER CONFIGURATION ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][Orchestrator] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OneMindArmyPipeline:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.game_name = self.config.get("name", "unknown_game")
        
        # --- Recherche Robuste du Binaire C++ ---
        self.exe_name = "onemindarmy.exe" if platform.system() == "Windows" else "onemindarmy"
        possible_paths = [
            Path(f"build/bin/{self.exe_name}"),
            Path(f"build/bin/Release/{self.exe_name}"),
            Path(f"build/x64-Release/bin/{self.exe_name}"),
            Path(f"bin/{self.exe_name}"),
            Path(self.exe_name)
        ]
        
        self.cpp_engine_path = next((p for p in possible_paths if p.exists()), None)
        
        if not self.cpp_engine_path:
            logger.error(f"C++ Executable '{self.exe_name}' not found. Please compile the C++ project first.")
            sys.exit(1)
            
        # --- Dossiers de travail ---
        self.data_dir = Path("data") / self.game_name
        self.models_dir = Path("models") / self.game_name
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd_list: list, phase_name: str):
        """Exécute une commande système en temps réel."""
        cmd_str = " ".join([str(x) for x in cmd_list])
        logger.info(f"--- STARTING PHASE : {phase_name} ---")
        logger.debug(f"Command: {cmd_str}")
        
        try:
            subprocess.run(cmd_list, check=True)
            logger.info(f"--- PHASE SUCCESS : {phase_name} ---\n")
        except subprocess.CalledProcessError as e:
            logger.error(f"FATAL FAILURE during phase [{phase_name}]. Exit code: {e.returncode}")
            sys.exit(1)

    def wait_for_vram_cleanup(self, timeout_sec: int = 15):
        """S'assure que la VRAM est libérée avant la phase suivante."""
        if platform.system() == "Windows" and not shutil.which("nvidia-smi"):
            return 
            
        logger.info("[CleanUp] Waiting for GPU VRAM release...")
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                    encoding='utf-8', stderr=subprocess.DEVNULL
                )
                if not result.strip():
                    logger.info(f"[CleanUp] VRAM cleared successfully in {time.time() - start_time:.1f}s.")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
            
        logger.warning(f"[CleanUp] Timeout ({timeout_sec}s). VRAM might still be occupied.")
        return False

    def ensure_process_terminated(self):
        """Tue les instances zombies de l'engine C++ (anti-fuite de mémoire)."""
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if proc.info['name'] == self.exe_name:
                    logger.warning(f"Terminating zombie C++ engine (PID: {proc.info['pid']})...")
                    proc.kill()
                    proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def update_config_iteration(self, iteration: int):
        """Injecte l'itération courante dans le YAML pour que le C++ nomme bien le fichier."""
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        if "training" not in cfg:
            cfg["training"] = {}
        cfg["training"]["currentIteration"] = iteration
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        # Met à jour la config en RAM
        self.config = cfg

    def get_start_iteration(self) -> int:
        """Détecte la dernière itération jouée pour reprendre l'entraînement proprement."""
        bin_files = sorted(self.data_dir.glob("iteration_*.bin"))
        if not bin_files:
            return 1
        
        # Extrait "0042" de "iteration_0042.bin"
        last_file = bin_files[-1].stem
        try:
            last_iter = int(last_file.split("_")[1])
            logger.info(f"[Auto-Resume] Found existing dataset up to iteration {last_iter}.")
            return last_iter + 1
        except Exception:
            return 1

    def enforce_sliding_window(self):
        """
        Sliding Window dynamique basée sur le NOMBRE DE SAMPLES.
        Supprime les fichiers entiers les plus vieux quand la limite est atteinte.
        """
        max_samples = self.config.get("training", {}).get("slidingWindowSamples", 2_000_000)
        meta_path = self.data_dir / f"{self.game_name}_training_data.bin.meta.json"
        
        if not meta_path.exists():
            return

        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        struct_size = meta.get("sizeofTrainingSample")
        if not struct_size:
            logger.warning("[Sliding Window] 'sizeofTrainingSample' missing in meta. Skipping.")
            return

        # 1. Lister et trier tous les fichiers d'itération du plus vieux au plus récent
        bin_files = sorted(self.data_dir.glob("iteration_*.bin"))
        if not bin_files:
            return

        # 2. Calculer le nombre de samples par fichier
        file_stats = []
        total_samples = 0
        
        for b_file in bin_files:
            f_size = b_file.stat().st_size
            samples = f_size // struct_size
            file_stats.append((b_file, samples))
            total_samples += samples

        logger.info(f"[Sliding Window] Current Capacity: {total_samples} / {max_samples} samples.")

        # 3. Supprimer les vieux fichiers si on dépasse la limite
        files_deleted = 0
        samples_removed = 0
        
        while total_samples > max_samples and len(file_stats) > 1:
            oldest_file, samples_in_file = file_stats.pop(0)
            
            try:
                os.remove(oldest_file)
                total_samples -= samples_in_file
                samples_removed += samples_in_file
                files_deleted += 1
                logger.info(f"[Sliding Window] Deleted {oldest_file.name} ({samples_in_file} samples).")
            except Exception as e:
                logger.error(f"[Sliding Window] Failed to delete {oldest_file.name}: {e}")
                break

        if files_deleted > 0:
            logger.info(f"[Sliding Window] Trimmed {files_deleted} files. New total: {total_samples} samples.")

    def phase_bootstrap(self):
        cmd = [sys.executable, "scripts/init_v0.py", "--config", str(self.config_path)]
        self.run_command(cmd, "Bootstrap v0 (Initialization)")

    def phase_self_play(self, model_name: str):
        cmd = [
            str(self.cpp_engine_path),
            str(self.config_path),
            "--mode", "train",
            "--model", model_name
        ]
        self.run_command(cmd, f"Self-Play Generation (Model: {model_name})")

    def phase_train(self):
        cmd = [sys.executable, "scripts/train.py", "--config", str(self.config_path)]
        self.run_command(cmd, "Neural Network Training & TRT Compilation")

    def run_loop(self, total_iterations: int):
        logger.info(f"=== STARTING OMA PIPELINE | GAME: {self.game_name.upper()} ===")
        
        best_model_name = "best_model.plan"
        best_model_path = self.models_dir / best_model_name
        latest_model_path = self.models_dir / "latest_model.plan"

        if not best_model_path.exists():
            logger.warning("No 'best_model.plan' found. Triggering v0 initialization...")
            self.phase_bootstrap()
            self.wait_for_vram_cleanup()

        start_time = time.time()
        start_iter = self.get_start_iteration()

        for iteration in range(start_iter, start_iter + total_iterations):
            logger.info(f"\n{'='*60}\n=== ITERATION {iteration} ===\n{'='*60}")

            # 0. Update YAML to inform C++ of the current iteration
            self.update_config_iteration(iteration)

            # 1. Génération de données (C++ / TensorRT)
            self.phase_self_play(best_model_name)
            self.wait_for_vram_cleanup()

            # 1.5 Sliding Window Dynamique (Basée sur les samples)
            self.enforce_sliding_window()

            # 2. Entraînement (Python / PyTorch)
            self.phase_train()
            self.wait_for_vram_cleanup()

            # 3. Promotion du Modèle
            if latest_model_path.exists():
                shutil.copy(latest_model_path, best_model_path)
                logger.info(f"Iteration {iteration} complete. 'latest_model' promoted to 'best_model'.")
            else:
                logger.error("latest_model.plan was not generated! Check train.py output.")
                sys.exit(1)

        total_hours = (time.time() - start_time) / 3600
        logger.info(f"=== PIPELINE FINISHED : {total_iterations} ITERATIONS COMPLETED IN {total_hours:.2f} HOURS ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One Mind Army - Orchestrator")
    parser.add_argument("config", help="Path to the YAML config file")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    pipeline = OneMindArmyPipeline(args.config)
    
    pipeline.ensure_process_terminated()
    pipeline.wait_for_vram_cleanup(timeout_sec=5)
    
    try:
        pipeline.run_loop(args.iterations)
    except KeyboardInterrupt:
        logger.warning("\n[Interruption] Pipeline manually stopped by the user. Cleaning up...")
        pipeline.ensure_process_terminated()
        sys.exit(0)