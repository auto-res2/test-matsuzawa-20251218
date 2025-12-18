import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main orchestrator for training with Hydra configuration."""
    
    # Validate required parameters
    if cfg.run is None:
        raise ValueError("run_id must be specified via CLI: run={run_id}")
    if cfg.results_dir is None:
        raise ValueError("results_dir must be specified via CLI: results_dir={path}")
    if cfg.mode is None:
        raise ValueError("mode must be specified via CLI: mode={trial|full}")
    
    run_id = cfg.run
    results_dir = cfg.results_dir
    mode = cfg.mode
    
    # Validate mode
    if mode not in ["trial", "full"]:
        raise ValueError(f"mode must be 'trial' or 'full', got '{mode}'")
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Load run configuration
    run_config_path = Path("config/runs") / f"{run_id}.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}. Searched in: {run_config_path.resolve()}")
    
    # Load run-specific config
    run_cfg = OmegaConf.load(run_config_path)
    
    # Merge with base config
    cfg = OmegaConf.merge(cfg, run_cfg)
    
    # Add mode and results_dir to config
    cfg.mode = mode
    cfg.results_dir = results_dir
    
    # Mode-specific configuration adjustments
    if mode == "trial":
        cfg.training.epochs = 1
        cfg.training.batch_size = min(cfg.training.batch_size, 32)
        cfg.wandb.mode = "disabled"
        if "optuna" in cfg:
            cfg.optuna.enabled = False
            cfg.optuna.n_trials = 0
        print(f"[INFO] Trial mode: epochs={cfg.training.epochs}, wandb disabled, optuna disabled")
    elif mode == "full":
        cfg.wandb.mode = "online"
        print(f"[INFO] Full mode: epochs={cfg.training.epochs}, wandb enabled")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Validate WANDB_API_KEY for full mode
    if mode == "full" and cfg.wandb.mode == "online" and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set. WandB functionality may be limited.")
    
    # Save merged config to results directory
    config_output = Path(results_dir) / f"{run_id}_config.yaml"
    OmegaConf.save(cfg, config_output)
    logger.info(f"Saved merged config to {config_output}")
    
    # Import and run train module directly with hydra config
    print(f"\n{'='*80}")
    print(f"Launching training for run_id={run_id}, mode={mode}")
    print(f"{'='*80}\n")
    
    # Import train function
    from src.train import train_main
    
    # Pass configuration to train function
    train_main(cfg)
    
    print(f"\n{'='*80}")
    print(f"Training completed for run_id={run_id}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
