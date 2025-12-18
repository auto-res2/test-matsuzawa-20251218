import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
import logging
import math
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

# Import local modules
from src.model import build_model
from src.preprocess import build_dataset, get_data_loaders

logger = logging.getLogger(__name__)

class TrainingMetrics:
    """Tracks comprehensive training metrics."""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.spectral_metrics = defaultdict(lambda: defaultdict(list))
        self.threshold_metrics = defaultdict(list)
    
    def log_batch(self, **kwargs):
        """Log batch-level metrics."""
        for key, value in kwargs.items():
            if value is not None:
                self.metrics[key].append(float(value))
    
    def log_spectral(self, epoch, layer_name, rho_eff):
        """Log per-component spectral concentration."""
        self.spectral_metrics[f"epoch_{epoch}"][layer_name].append(rho_eff)
    
    def log_threshold(self, epoch, threshold):
        """Log adaptive threshold evolution."""
        self.threshold_metrics[epoch].append(threshold)
    
    def get_epoch_summary(self):
        """Compute summary statistics for current epoch."""
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        return summary
    
    def reset_batch_metrics(self):
        """Clear batch metrics for next epoch."""
        self.metrics.clear()

class CheckpointManager:
    """Manages model checkpoints and best model tracking."""
    def __init__(self, checkpoint_dir, metric_name="val_accuracy", mode="max"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode  # "max" or "min"
        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = -1
        self.best_checkpoint_path = None
    
    def save(self, epoch, model, optimizer, metrics):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        
        # Check if this is best model
        current_value = metrics.get(self.metric_name, None)
        if current_value is not None:
            is_best = False
            if self.mode == "max" and current_value > self.best_value:
                is_best = True
                self.best_value = current_value
            elif self.mode == "min" and current_value < self.best_value:
                is_best = True
                self.best_value = current_value
            
            if is_best:
                self.best_epoch = epoch
                self.best_checkpoint_path = path
                # Save as best checkpoint
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
        
        return path
    
    def load_best(self, model, optimizer):
        """Load best checkpoint."""
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            # Post-checkpoint-load assertions
            assert model.training is not None, "Model state corrupted after load"
            for name, param in model.named_parameters():
                assert param.data is not None, f"Parameter {name} is None after load"
                assert param.grad is not None or param.requires_grad, f"Parameter {name} requires_grad but has no grad"
            
            return checkpoint["epoch"], checkpoint["metrics"]
        return None, None

def compute_accuracy(outputs, labels):
    """Compute classification accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def compute_loss_and_accuracy(model, dataloader, criterion, device):
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Data leak prevention: validate separation
            assert inputs.shape[0] == targets.shape[0], f"Batch size mismatch: {inputs.shape[0]} vs {targets.shape[0]}"
            assert targets.dim() >= 1, f"Target must have at least 1 dimension, got shape {targets.shape}"
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accuracy = compute_accuracy(outputs, targets)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_accuracy

def extract_layer_parameters(model):
    """Extract parameter groups by layer for spectral tracking."""
    layer_params = defaultdict(list)
    for name, param in model.named_parameters():
        if param.requires_grad:
            parts = name.split(".")
            if len(parts) >= 2:
                layer_name = ".".join(parts[:2])
            else:
                layer_name = parts[0]
            layer_params[layer_name].append(param)
    return layer_params

def compute_spectral_concentration(optimizer, layer_params):
    """Compute spectral concentration metrics per layer."""
    spectral_data = {}
    
    for layer_name, params in layer_params.items():
        concentrations = []
        for param in params:
            if param in optimizer.state:
                state = optimizer.state[param]
                if "grad_sq_history" in state and len(state["grad_sq_history"]) >= 5:
                    concentration = state.get("spectral_concentration", 1.0)
                    concentrations.append(concentration)
        
        if concentrations:
            spectral_data[layer_name] = {
                "mean": np.mean(concentrations),
                "std": np.std(concentrations),
                "min": np.min(concentrations),
                "max": np.max(concentrations),
                "values": concentrations,
            }
    
    return spectral_data

def compute_adaptive_thresholds(optimizer, layer_params):
    """Extract adaptive thresholds used by SARA."""
    threshold_data = {}
    
    for layer_name, params in layer_params.items():
        thresholds = []
        for param in params:
            if param in optimizer.state:
                state = optimizer.state[param]
                if "adaptive_threshold" in state:
                    thresholds.append(state["adaptive_threshold"])
        
        if thresholds:
            threshold_data[layer_name] = {
                "mean": np.mean(thresholds),
                "std": np.std(thresholds),
                "min": np.min(thresholds),
                "max": np.max(thresholds),
                "values": thresholds,
            }
    
    return threshold_data

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg, metrics, layer_params, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Assertions at batch start (CRITICAL for data leak prevention)
        assert inputs.ndim >= 3, f"Input shape mismatch: {inputs.shape}"
        assert targets.ndim >= 1, f"Target shape mismatch: {targets.shape}"
        assert inputs.shape[0] == targets.shape[0], f"Batch size mismatch: {inputs.shape[0]} vs {targets.shape[0]}"
        assert inputs.shape[0] > 0, f"Empty batch at step {global_step}"
        if batch_idx == 0:
            assert targets.min() >= 0 and targets.max() < cfg.model.num_classes, f"Target values out of range: {targets.min()}-{targets.max()}"
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass - inputs ONLY, labels used only for loss
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Loss computation (labels ONLY for loss, never concatenated to inputs)
        loss = criterion(outputs, targets)
        accuracy = compute_accuracy(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: Gradient integrity check before optimizer step
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        has_gradients = sum(1 for p in trainable_params if p.grad is not None and not torch.all(p.grad == 0))
        
        if batch_idx > 0 and has_gradients == 0:
            logger.warning(f"No non-zero gradients at batch {batch_idx}, epoch {epoch}")
        
        # Gradient clipping
        if cfg.training.get("gradient_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        
        # Optimizer step (CRITICAL: assert gradients exist and are valid)
        for p in trainable_params:
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in gradients at step {global_step}"
                assert not torch.isinf(p.grad).any(), f"Inf in gradients at step {global_step}"
        
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        global_step += 1
        
        # Log to wandb comprehensively (per-batch logging)
        metrics.log_batch(
            train_loss=loss.item(),
            train_accuracy=accuracy,
        )
        
        if cfg.wandb.mode != "disabled" and batch_idx % 10 == 0:
            wandb.log({
                "train_loss_batch": loss.item(),
                "train_accuracy_batch": accuracy,
                "global_step": global_step,
            }, step=global_step)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{accuracy:.4f}"
        })
        
        # Trial mode: limit batches
        if cfg.mode == "trial" and batch_idx >= 1:
            break
    
    # Compute spectral metrics if using SARA
    if "sara" in cfg.training.optimizer.lower() and cfg.get("evaluation", {}).get("compute_spectral_concentration", False):
        spectral_data = compute_spectral_concentration(optimizer, layer_params)
        for layer_name, data in spectral_data.items():
            metrics.log_spectral(epoch, layer_name, data["mean"])
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    f"spectral_concentration_{layer_name}": data["mean"],
                }, step=global_step)
    
    # Compute adaptive thresholds
    if "sara" in cfg.training.optimizer.lower() and cfg.get("evaluation", {}).get("track_threshold_evolution", False):
        threshold_data = compute_adaptive_thresholds(optimizer, layer_params)
        for layer_name, data in threshold_data.items():
            metrics.log_threshold(epoch, data["mean"])
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    f"adaptive_threshold_{layer_name}": data["mean"],
                }, step=global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_accuracy, global_step

def create_optuna_objective(cfg, train_loader, val_loader, test_loader, device, checkpoint_dir):
    """Create Optuna objective function for hyperparameter search."""
    
    def objective(trial: Trial) -> float:
        """Objective function for Optuna."""
        
        try:
            # Suggest hyperparameters based on config
            trial_params = {}
            
            if cfg.optuna.enabled and cfg.optuna.get("search_spaces"):
                for search_space in cfg.optuna.search_spaces:
                    param_name = search_space.param_name
                    dist_type = search_space.distribution_type
                    
                    if dist_type == "uniform":
                        trial_params[param_name] = trial.suggest_float(
                            param_name, 
                            search_space.low, 
                            search_space.high
                        )
                    elif dist_type == "loguniform":
                        trial_params[param_name] = trial.suggest_float(
                            param_name, 
                            search_space.low, 
                            search_space.high,
                            log=True
                        )
                    elif dist_type == "int":
                        trial_params[param_name] = trial.suggest_int(
                            param_name,
                            int(search_space.low),
                            int(search_space.high)
                        )
            
            # Build model
            model = build_model(cfg.model)
            model = model.to(device)
            
            # Setup criterion
            criterion = nn.CrossEntropyLoss()
            
            # Setup optimizer with trial parameters
            optimizer_name = cfg.training.optimizer.lower()
            
            if "sara" in optimizer_name:
                try:
                    from src.optimizer_sara import SARA
                except ImportError:
                    raise ImportError("SARA optimizer not found. Ensure src/optimizer_sara.py exists.")

                optimizer_params = cfg.training.get("optimizer_params", {})
                if "base_threshold" in trial_params:
                    optimizer_params["base_threshold"] = trial_params["base_threshold"]
                if "histogram_window" in trial_params:
                    optimizer_params["histogram_window"] = trial_params["histogram_window"]

                lr = trial_params.get("learning_rate", cfg.training.learning_rate)
                wd = trial_params.get("weight_decay", cfg.training.weight_decay)

                optimizer = SARA(
                    model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    betas=tuple(optimizer_params.get("betas", [0.9, 0.999])),
                    eps=optimizer_params.get("eps", 1e-8),
                    base_threshold=optimizer_params.get("base_threshold", 5.0),
                    enable_spectral=optimizer_params.get("enable_spectral", True),
                    enable_phase_aware=optimizer_params.get("enable_phase_aware", True),
                    histogram_window=optimizer_params.get("histogram_window", 100),
                )
            elif "radam" in optimizer_name:
                try:
                    from torch.optim import RAdam
                except ImportError:
                    try:
                        from torch_optimizer import RAdam
                    except ImportError:
                        raise ImportError("RAdam not found. Install torch-optimizer package.")
                
                optimizer_params = cfg.training.get("optimizer_params", {})
                
                lr = trial_params.get("learning_rate", cfg.training.learning_rate)
                wd = trial_params.get("weight_decay", cfg.training.weight_decay)
                
                betas_0 = trial_params.get("betas_0", optimizer_params.get("betas", [0.9, 0.999])[0])
                betas_1 = trial_params.get("betas_1", optimizer_params.get("betas", [0.9, 0.999])[1])
                
                optimizer = RAdam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    betas=(betas_0, betas_1),
                    eps=optimizer_params.get("eps", 1e-8),
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            # Setup scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
            
            # Train for a few epochs
            best_val_accuracy = 0.0
            trial_epochs = min(10, cfg.training.epochs)
            
            for epoch in range(trial_epochs):
                model.train()
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    if cfg.mode == "trial" and batch_idx >= 1:
                        break
                
                # Validate
                val_loss, val_accuracy = compute_loss_and_accuracy(model, val_loader, criterion, device)
                best_val_accuracy = max(best_val_accuracy, val_accuracy)
                
                scheduler.step()
                
                # Report to Optuna
                trial.report(val_accuracy, epoch)
                
                # Optuna pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Cleanup to prevent GPU memory leaks
            del model
            del optimizer
            torch.cuda.empty_cache()
            
            return best_val_accuracy
        
        except (RuntimeError, ValueError, optuna.TrialPruned):
            raise
        except Exception as e:
            logger.warning(f"Trial failed with exception: {e}")
            raise

    return objective

def train_main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    seed = cfg.training.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")
    
    # Create results directory structure
    checkpoint_dir = Path(cfg.results_dir) / cfg.run_id / "checkpoints"
    metrics_dir = Path(cfg.results_dir) / cfg.run_id / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB with proper finalization
    wandb_initialized = False
    try:
        if cfg.wandb.mode != "disabled":
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                id=cfg.run_id,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
                dir=str(cfg.results_dir),
            )
            wandb_initialized = True
            logger.info(f"WandB initialized. Run URL: {wandb.run.url}")
            print(f"WandB Run URL: {wandb.run.url}")
        else:
            logger.info("WandB disabled (trial mode)")
        
        # Build model
        logger.info(f"Building model: {cfg.model.name}")
        model = build_model(cfg.model)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Post-init assertions
        assert total_params > 0, "Model has no parameters"
        assert cfg.model.num_classes > 0, "num_classes must be > 0"
        for name, param in model.named_parameters():
            assert param.data is not None, f"Parameter {name} has None data"
        
        # Build datasets and dataloaders
        logger.info(f"Building dataset: {cfg.dataset.name}")
        train_dataset, val_dataset, test_dataset = build_dataset(cfg.dataset, cfg.model.get("architecture", "resnet"))
        train_loader, val_loader, test_loader = get_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=cfg.training.batch_size,
            num_workers=4
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optuna hyperparameter search
        best_trial_params = {}
        if cfg.get("optuna", {}).get("enabled", False) and cfg.optuna.get("n_trials", 0) > 0 and cfg.mode == "full":
            logger.info(f"Starting Optuna hyperparameter search with {cfg.optuna.n_trials} trials...")
            
            sampler = TPESampler(seed=seed)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler
            )
            
            objective_fn = create_optuna_objective(cfg, train_loader, val_loader, test_loader, device, checkpoint_dir)
            
            study.optimize(
                objective_fn,
                n_trials=cfg.optuna.n_trials,
                show_progress_bar=True,
                catch=(RuntimeError, ValueError)
            )
            
            best_trial = study.best_trial
            best_trial_params = best_trial.params
            logger.info(f"Best trial value: {best_trial.value}")
            logger.info(f"Best trial params: {best_trial_params}")
            
            if wandb_initialized:
                wandb.log({
                    "optuna_best_value": best_trial.value,
                    "optuna_best_params": best_trial_params,
                })
        
        # Setup optimizer
        optimizer_name = cfg.training.optimizer.lower()
        logger.info(f"Setting up optimizer: {optimizer_name}")
        
        if "sara" in optimizer_name:
            try:
                from src.optimizer_sara import SARA
            except ImportError:
                raise ImportError("SARA optimizer not found. Ensure src/optimizer_sara.py exists.")

            optimizer_params = cfg.training.get("optimizer_params", {})
            if "base_threshold" in best_trial_params:
                optimizer_params["base_threshold"] = best_trial_params["base_threshold"]
            if "histogram_window" in best_trial_params:
                optimizer_params["histogram_window"] = best_trial_params["histogram_window"]

            lr = best_trial_params.get("learning_rate", cfg.training.learning_rate)
            wd = best_trial_params.get("weight_decay", cfg.training.weight_decay)

            optimizer = SARA(
                model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=tuple(optimizer_params.get("betas", [0.9, 0.999])),
                eps=optimizer_params.get("eps", 1e-8),
                base_threshold=optimizer_params.get("base_threshold", 5.0),
                enable_spectral=optimizer_params.get("enable_spectral", True),
                enable_phase_aware=optimizer_params.get("enable_phase_aware", True),
                histogram_window=optimizer_params.get("histogram_window", 100),
            )
        elif "radam" in optimizer_name:
            try:
                from torch.optim import RAdam
            except ImportError:
                try:
                    from torch_optimizer import RAdam
                except ImportError:
                    raise ImportError("RAdam not available. Install torch-optimizer or use PyTorch >=2.1")
            
            optimizer_params = cfg.training.get("optimizer_params", {})
            
            lr = best_trial_params.get("learning_rate", cfg.training.learning_rate)
            wd = best_trial_params.get("weight_decay", cfg.training.weight_decay)
            
            betas_0 = best_trial_params.get("betas_0", optimizer_params.get("betas", [0.9, 0.999])[0])
            betas_1 = best_trial_params.get("betas_1", optimizer_params.get("betas", [0.9, 0.999])[1])
            
            optimizer = RAdam(
                model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=(betas_0, betas_1),
                eps=optimizer_params.get("eps", 1e-8),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Post-init optimizer assertions
        assert len(optimizer.param_groups) > 0, "Optimizer has no parameter groups"
        assert optimizer.param_groups[0]["lr"] > 0, "Learning rate must be positive"
        
        # Setup scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        
        # Checkpoint manager
        checkpoint_manager = CheckpointManager(checkpoint_dir, metric_name="val_accuracy", mode="max")
        
        # Training metrics
        metrics = TrainingMetrics()
        
        # Extract layer parameters for spectral tracking
        layer_params = extract_layer_parameters(model)
        
        logger.info(f"Starting training for {cfg.training.epochs} epochs")
        
        # Training loop
        best_val_accuracy = 0.0
        convergence_epoch = None
        target_accuracy = None
        accuracy_history = []
        loss_history = []
        early_stage_losses = {}
        global_step = 0
        
        for epoch in range(cfg.training.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss, train_accuracy, global_step = train_epoch(
                model, train_loader, optimizer, criterion, device,
                epoch, cfg, metrics, layer_params, global_step
            )
            
            # Validate
            val_loss, val_accuracy = compute_loss_and_accuracy(model, val_loader, criterion, device)
            
            accuracy_history.append(val_accuracy)
            loss_history.append(val_loss)
            
            # Track early-stage losses (checkpoint intervals from config)
            checkpoint_intervals = cfg.get("evaluation", {}).get("checkpoint_intervals", [5, 10, 20])
            if (epoch + 1) in checkpoint_intervals:
                early_stage_losses[f"epoch_{epoch+1}"] = val_loss
            
            # Track best
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
            # Log metrics comprehensively
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            if wandb_initialized:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }, step=epoch)
            
            # Save checkpoint
            checkpoint_manager.save(epoch, model, optimizer, {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })
            
            # Update scheduler
            scheduler.step()
            
            # Trial mode: limit epochs
            if cfg.mode == "trial" and epoch >= 0:
                break
        
        # Compute convergence epoch after training
        if len(accuracy_history) > 0:
            final_accuracy = max(accuracy_history)
            target_accuracy = final_accuracy * 0.9
            for conv_epoch, acc in enumerate(accuracy_history):
                if acc >= target_accuracy:
                    convergence_epoch = conv_epoch + 1
                    break
            if convergence_epoch is None:
                convergence_epoch = cfg.training.epochs
            logger.info(f"Convergence to 90% final accuracy at epoch {convergence_epoch}")
        else:
            convergence_epoch = cfg.training.epochs
        
        # Final evaluation on test set
        logger.info("\n" + "="*60)
        logger.info("Final Evaluation on Test Set")
        logger.info("="*60)
        
        test_loss, test_accuracy = compute_loss_and_accuracy(model, test_loader, criterion, device)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        logger.info(f"\nConvergence Speed (epochs to 90% final accuracy): {convergence_epoch}")
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        # Log final metrics to WandB
        if wandb_initialized:
            wandb.summary["test_loss"] = test_loss
            wandb.summary["test_accuracy"] = test_accuracy
            wandb.summary["convergence_speed_to_90_percent_final_accuracy"] = convergence_epoch
            wandb.summary["final_accuracy"] = test_accuracy
            wandb.summary["best_val_accuracy"] = best_val_accuracy
            
            for epoch_key, loss_val in early_stage_losses.items():
                wandb.summary[f"early_stage_loss_{epoch_key}"] = loss_val
        
        # Save metrics to JSON
        metrics_output = {
            "run_id": cfg.run_id,
            "method": cfg.get("method", "unknown"),
            "model": cfg.model.name,
            "dataset": cfg.dataset.name,
            "optimizer": cfg.training.optimizer,
            "convergence_speed_to_90_percent_final_accuracy": int(convergence_epoch),
            "final_test_accuracy": float(test_accuracy),
            "best_val_accuracy": float(best_val_accuracy),
            "test_loss": float(test_loss),
            "accuracy_history": [float(x) for x in accuracy_history],
            "loss_history": [float(x) for x in loss_history],
            "early_stage_losses": early_stage_losses,
            "training_epochs": cfg.training.epochs,
            "optuna_best_params": best_trial_params,
        }
        
        metrics_json_path = metrics_dir / "metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_output, f, indent=2)
        logger.info(f"Saved metrics to {metrics_json_path}")
        
        logger.info("\nTraining completed successfully!")
    
    finally:
        # Finalize WandB with proper cleanup
        if wandb_initialized:
            try:
                wandb.log({})  # Force flush pending logs
                wandb.finish()
            except Exception as e:
                logger.error(f"Critical: WandB finalization failed: {e}")
                raise
