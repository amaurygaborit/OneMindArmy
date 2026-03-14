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
        """Exécute une commande système. Laisse stdout libre pour les logs en temps réel."""
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
        """Interroge nvidia-smi pour s'assurer que la VRAM est totalement libérée."""
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

    def enforce_sliding_window(self, max_iterations_to_keep: int = 5):
        """
        Conserve exactement les X dernières itérations.
        MÉTHODE TOTALEMENT GAME-AGNOSTIC : On traque les ajouts réels à chaque cycle.
        """
        dataset_path = self.data_dir / f"{self.game_name}_training_data.bin"
        meta_path = self.data_dir / f"{self.game_name}_training_data.bin.meta.json"
        window_path = self.data_dir / "sliding_window.json"
        
        if not dataset_path.exists() or not meta_path.exists():
            return

        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        struct_size = meta.get("sizeofTrainingSample")
        if not struct_size:
            logger.warning("[Sliding Window] 'sizeofTrainingSample' missing. Skipping.")
            return

        # 1. Lire l'historique des itérations
        window_state = {"iters": []}
        if window_path.exists():
            with open(window_path, 'r') as f:
                window_state = json.load(f)

        # 2. Calculer le nombre exact de samples fraîchement générés
        file_size = dataset_path.stat().st_size
        total_samples_now = file_size // struct_size
        samples_previously_tracked = sum(window_state["iters"])
        
        new_samples = total_samples_now - samples_previously_tracked
        
        if new_samples > 0:
            window_state["iters"].append(new_samples)
            logger.info(f"[Sliding Window] Tracked {new_samples} new samples for this iteration.")

        # 3. Appliquer la coupe si on dépasse le buffer
        if len(window_state["iters"]) > max_iterations_to_keep:
            iters_to_remove = len(window_state["iters"]) - max_iterations_to_keep
            samples_to_remove = sum(window_state["iters"][:iters_to_remove])
            
            logger.info(f"[Sliding Window] Max iterations ({max_iterations_to_keep}) exceeded.")
            logger.info(f"[Sliding Window] Trimming the oldest {iters_to_remove} iteration(s) ({samples_to_remove} samples)...")
            
            bytes_to_remove = samples_to_remove * struct_size
            temp_path = dataset_path.with_suffix('.bin.tmp')
            
            # Transfert sécurisé par blocs (Évite la saturation de RAM sur les gros fichiers)
            with open(dataset_path, 'rb') as f_in, open(temp_path, 'wb') as f_out:
                f_in.seek(bytes_to_remove)
                shutil.copyfileobj(f_in, f_out, length=1024*1024*64) # Copie par blocs de 64 Mo
                
            # Écrase l'ancien fichier
            os.replace(temp_path, dataset_path)
                
            # Mettre à jour l'état
            window_state["iters"] = window_state["iters"][iters_to_remove:]
            logger.info("[Sliding Window] Dataset trimmed successfully (RAM Safe Mode).")

        # 4. Sauvegarder l'état pour la prochaine boucle
        with open(window_path, 'w') as f:
            json.dump(window_state, f)

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

        for iteration in range(1, total_iterations + 1):
            logger.info(f"\n{'='*60}\n=== ITERATION {iteration} / {total_iterations} ===\n{'='*60}")

            # 1. Génération de données (C++ / TensorRT)
            self.phase_self_play(best_model_name)
            self.wait_for_vram_cleanup()

            # 1.5 Sliding Window (Python - Agnostique et Sûr)
            self.enforce_sliding_window(max_iterations_to_keep=5)

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
        logger.info(f"=== PIPELINE FINISHED : {total_iterations} ITERATIONS IN {total_hours:.2f} HOURS ===")

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