import os
import sys
import yaml
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
        
        # --- Recherche du binaire C++ ---
        self.exe_name = "onemindarmy.exe" if platform.system() == "Windows" else "onemindarmy"
        possible_paths = [
            Path(f"build/x64-Release/bin/{self.exe_name}"),
            Path(f"build/bin/{self.exe_name}")
        ]
        
        self.cpp_engine_path = next((p for p in possible_paths if p.exists()), None)
        
        if not self.cpp_engine_path:
            logger.error(f"C++ Executable '{self.exe_name}' not found. Did you compile?")
            sys.exit(1)
            
        self.data_dir = Path("data") / self.game_name
        self.models_dir = Path("models") / self.game_name
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd_list: list, phase_name: str):
        cmd_str = " ".join([str(x) for x in cmd_list])
        logger.info(f"--- STARTING PHASE : {phase_name} ---")
        logger.debug(f"Command: {cmd_str}")
        
        try:
            # On utilise stdout=None pour laisser le C++ afficher ses logs m/s en temps réel
            subprocess.run(cmd_list, check=True)
            logger.info(f"--- PHASE SUCCESS : {phase_name} ---")
        except subprocess.CalledProcessError as e:
            logger.error(f"FATAL FAILURE during phase [{phase_name}]. Code: {e.returncode}")
            sys.exit(1)

    def wait_for_vram_cleanup(self, timeout_sec: int = 15):
        """Interroge nvidia-smi pour s'assurer que la VRAM est libérée."""
        if platform.system() == "Windows" and not shutil.which("nvidia-smi"):
            return # Skip if no nvidia-smi
            
        logger.info("[CleanUp] Waiting for VRAM release...")
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                    encoding='utf-8', stderr=subprocess.DEVNULL
                )
                if not result.strip():
                    logger.info(f"[CleanUp] VRAM cleared in {time.time()-start_time:.1f}s")
                    return True
            except:
                pass
            time.sleep(0.5)
        logger.warning("VRAM cleanup timeout reached.")

    def ensure_process_terminated(self):
        """Tue les instances zombies de l'engine C++."""
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] == self.exe_name:
                    logger.warning(f"Terminating zombie engine (PID: {proc.pid})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def phase_bootstrap(self):
        """Génère le modèle v0 aléatoire."""
        cmd = [sys.executable, "scripts/init_v0.py", "--config", str(self.config_path)]
        self.run_command(cmd, "Bootstrap v0")

    def phase_self_play(self, model_name: str):
        """Lance le moteur C++ pour générer des parties."""
        # Note: On passe uniquement le NOM du fichier, le C++ construit le chemin models/game/name
        cmd = [
            str(self.cpp_engine_path),
            str(self.config_path),
            "--mode", "train",
            "--model", model_name
        ]
        self.run_command(cmd, f"Self-Play (Model: {model_name})")

    def phase_train(self):
        """Lance l'entraînement PyTorch (scripts/train.py)."""
        cmd = [sys.executable, "scripts/train.py", "--config", str(self.config_path)]
        self.run_command(cmd, "Neural Network Training")

    def run_loop(self, total_iterations: int):
        logger.info(f"=== STARTING OMA PIPELINE | GAME: {self.game_name} ===")
        
        best_model_name = "best_model.plan"
        best_model_path = self.models_dir / best_model_name
        latest_model_path = self.models_dir / "latest_model.plan"

        # 0. Initialisation
        if not best_model_path.exists():
            logger.warning("No best_model.plan found. Initializing v0...")
            self.phase_bootstrap()
            self.wait_for_vram_cleanup()

        for iteration in range(1, total_iterations + 1):
            logger.info(f"\n{'='*60}\nITERATION {iteration} / {total_iterations}\n{'='*60}")

            # 1. Génération de données (C++ / TensorRT)
            # Utilise l'architecture multi-arbres (kNumPlayers * ParallelGames)
            self.phase_self_play(best_model_name)
            self.wait_for_vram_cleanup()

            # 2. Entraînement (Python / PyTorch)
            # Produit 'latest_model.onnx' puis le compile en 'latest_model.plan'
            self.phase_train()
            self.wait_for_vram_cleanup()

            # 3. Promotion du modèle
            # On remplace l'ancien 'best' par le 'latest' qui vient d'être entraîné
            if latest_model_path.exists():
                shutil.copy(latest_model_path, best_model_path)
                logger.info(f"Iteration {iteration} success. 'latest_model' promoted to 'best_model'.")
            else:
                logger.error("latest_model.plan was not generated! Check train.py logs.")
                sys.exit(1)

        logger.info(f"=== PIPELINE FINISHED : {total_iterations} ITERATIONS COMPLETED ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One Mind Army Orchestrator")
    parser.add_argument("config", help="Path to the config (use chess_train.yaml)")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    pipeline = OneMindArmyPipeline(args.config)
    
    # Nettoyage initial
    pipeline.ensure_process_terminated()
    
    try:
        pipeline.run_loop(args.iterations)
    except KeyboardInterrupt:
        logger.warning("\nUser interrupted the pipeline. Cleaning up...")
        pipeline.ensure_process_terminated()
        sys.exit(0)