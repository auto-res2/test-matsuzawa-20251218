import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import logging
import os
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SARA experiment results")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory path")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON string list of run IDs")
    return parser.parse_args()

def load_wandb_config():
    """Load WandB configuration from config.yaml."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {"entity": "gengaru617-personal", "project": "2025-11-19", "mode": "online"}
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    return {
        "entity": cfg.get("wandb", {}).get("entity", "gengaru617-personal"),
        "project": cfg.get("wandb", {}).get("project", "2025-11-19"),
        "mode": cfg.get("wandb", {}).get("mode", "online"),
    }

def fetch_run_data_from_wandb(run_id: str, entity: str, project: str) -> Optional[Dict]:
    """Fetch comprehensive run data from WandB API."""
    
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Check if run is in trial or disabled mode
        if run.config.get("mode") == "trial" or run.config.get("wandb", {}).get("mode") == "disabled":
            logger.error(f"Cannot evaluate trial run {run_id} - insufficient WandB data")
            return None
        
        # Fetch history (time-series metrics)
        history = run.history()
        
        # Fetch summary (final metrics)
        summary = run.summary._json_dict if run.summary else {}
        
        # Fetch config
        config = dict(run.config) if run.config else {}
        
        return {
            "run_id": run_id,
            "history": history,
            "summary": summary,
            "config": config,
            "status": run.state,
        }
    except Exception as e:
        logger.error(f"Error fetching run {run_id} from WandB: {e}")
        return None

def process_run_metrics(run_data: Optional[Dict], results_dir: Path) -> Optional[Dict]:
    """Process metrics for a single run."""
    if run_data is None:
        return None
    
    run_id = run_data["run_id"]
    history = run_data["history"]
    summary = run_data["summary"]
    
    # Create run directory
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key metrics from history and summary
    metrics = {
        "run_id": run_id,
        "convergence_speed_to_90_percent_final_accuracy": summary.get("convergence_speed_to_90_percent_final_accuracy"),
        "final_test_accuracy": summary.get("test_accuracy"),
        "final_val_accuracy": summary.get("best_val_accuracy"),
        "test_loss": summary.get("test_loss"),
    }
    
    # Extract epoch-wise metrics from history
    if isinstance(history, pd.DataFrame) and len(history) > 0:
        metrics["train_loss_history"] = history.get("train_loss", []).tolist() if "train_loss" in history.columns else []
        metrics["val_loss_history"] = history.get("val_loss", []).tolist() if "val_loss" in history.columns else []
        metrics["train_accuracy_history"] = history.get("train_accuracy", []).tolist() if "train_accuracy" in history.columns else []
        metrics["val_accuracy_history"] = history.get("val_accuracy", []).tolist() if "val_accuracy" in history.columns else []
        
        # Extract spectral and threshold metrics if present
        spectral_cols = [col for col in history.columns if "spectral_concentration" in col]
        threshold_cols = [col for col in history.columns if "adaptive_threshold" in col]
        
        if spectral_cols:
            for col in spectral_cols:
                metrics[col] = history[col].tolist()
        
        if threshold_cols:
            for col in threshold_cols:
                metrics[col] = history[col].tolist()
    
    # Save metrics to JSON
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics for {run_id} to {metrics_file}")
    print(f"Saved metrics: {metrics_file}")
    
    # Generate run-specific figures
    generate_run_figures(run_data, run_dir)
    
    return metrics

def generate_run_figures(run_data: Dict, run_dir: Path) -> None:
    """Generate per-run figures."""
    run_id = run_data["run_id"]
    history = run_data["history"]
    
    if not isinstance(history, pd.DataFrame) or len(history) == 0:
        logger.warning(f"No history data for {run_id}, skipping figure generation")
        return
    
    # Learning curve
    if "train_loss" in history.columns or "val_loss" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        if "train_loss" in history.columns:
            ax.plot(history["train_loss"], label="Train Loss", alpha=0.7, linewidth=2)
        if "val_loss" in history.columns:
            ax.plot(history["val_loss"], label="Val Loss", alpha=0.7, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(f"Learning Curve - {run_id}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = run_dir / f"{run_id}_learning_curve.pdf"
        plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
        logger.info(f"Saved learning curve figure to {fig_path}")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Accuracy curve
    if "train_accuracy" in history.columns or "val_accuracy" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        if "train_accuracy" in history.columns:
            ax.plot(history["train_accuracy"], label="Train Accuracy", alpha=0.7, linewidth=2)
        if "val_accuracy" in history.columns:
            ax.plot(history["val_accuracy"], label="Val Accuracy", alpha=0.7, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"Accuracy Curve - {run_id}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = run_dir / f"{run_id}_accuracy_curve.pdf"
        plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
        logger.info(f"Saved accuracy curve figure to {fig_path}")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Early-stage loss checkpoint comparison
    early_checkpoints = [5, 10, 20]
    if "val_loss" in history.columns and len(history) >= max(early_checkpoints):
        fig, ax = plt.subplots(figsize=(10, 6))
        early_losses = []
        for checkpoint in early_checkpoints:
            if checkpoint - 1 < len(history):
                early_losses.append(history["val_loss"].iloc[checkpoint - 1])
            else:
                early_losses.append(None)
        
        valid_checkpoints = [cp for cp, loss in zip(early_checkpoints, early_losses) if loss is not None]
        valid_losses = [loss for loss in early_losses if loss is not None]
        
        if valid_losses:
            ax.bar(range(len(valid_losses)), valid_losses, color="steelblue", alpha=0.7)
            for i, (cp, loss) in enumerate(zip(valid_checkpoints, valid_losses)):
                ax.text(i, loss + 0.01, f"{loss:.4f}", ha="center", va="bottom", fontweight="bold")
            ax.set_xticks(range(len(valid_checkpoints)))
            ax.set_xticklabels([f"Epoch {cp}" for cp in valid_checkpoints])
            ax.set_ylabel("Validation Loss", fontsize=11)
            ax.set_title(f"Early-Stage Loss - {run_id}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            
            fig_path = run_dir / f"{run_id}_early_stage_loss.pdf"
            plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
            logger.info(f"Saved early-stage loss figure to {fig_path}")
            print(f"Generated figure: {fig_path}")
            plt.close()
    
    # Spectral concentration heatmap
    spectral_cols = [col for col in history.columns if "spectral_concentration" in col]
    if spectral_cols and len(history) > 0:
        try:
            spectral_data = history[spectral_cols].values
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(spectral_data.T, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_yticklabels([col.replace("spectral_concentration_", "") for col in spectral_cols])
            ax.set_title(f"Spectral Concentration Evolution - {run_id}", fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax, label="Concentration")
            plt.tight_layout()
            
            fig_path = run_dir / f"{run_id}_spectral_concentration_heatmap.pdf"
            plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
            logger.info(f"Saved spectral concentration heatmap to {fig_path}")
            print(f"Generated figure: {fig_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate spectral heatmap for {run_id}: {e}")
    
    # Adaptive threshold evolution
    threshold_cols = [col for col in history.columns if "adaptive_threshold" in col]
    if threshold_cols and len(history) > 0:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in threshold_cols:
                ax.plot(history[col], label=col.replace("adaptive_threshold_", ""), alpha=0.7, linewidth=2)
            ax.axhline(y=5.0, color="r", linestyle="--", label="RAdam threshold (5.0)", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Adaptive Threshold", fontsize=11)
            ax.set_title(f"Adaptive Threshold Evolution - {run_id}", fontsize=12, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            fig_path = run_dir / f"{run_id}_adaptive_threshold_evolution.pdf"
            plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
            logger.info(f"Saved adaptive threshold evolution to {fig_path}")
            print(f"Generated figure: {fig_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate threshold evolution figure for {run_id}: {e}")

def generate_comparison_figures(all_metrics: Dict[str, Dict], results_dir: Path) -> None:
    """Generate comparison figures across runs."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract run IDs and metrics
    run_ids = list(all_metrics.keys())
    convergence_speeds = {rid: all_metrics[rid].get("convergence_speed_to_90_percent_final_accuracy") for rid in run_ids}
    final_accuracies = {rid: all_metrics[rid].get("final_test_accuracy") for rid in run_ids}
    
    # Filter out None values
    convergence_speeds = {k: v for k, v in convergence_speeds.items() if v is not None}
    final_accuracies = {k: v for k, v in final_accuracies.items() if v is not None}
    
    # Convergence speed comparison
    if convergence_speeds:
        fig, ax = plt.subplots(figsize=(12, 6))
        run_names = list(convergence_speeds.keys())
        speeds = list(convergence_speeds.values())
        bars = ax.bar(range(len(run_names)), speeds, color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5)
        
        # Annotate bars
        for i, (bar, speed) in enumerate(zip(bars, speeds)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
                   f"{int(speed)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45, ha="right")
        ax.set_ylabel("Epochs to 90% Final Accuracy", fontsize=12)
        ax.set_title("Convergence Speed Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        fig_path = comparison_dir / "comparison_convergence_speed_bar.pdf"
        plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
        logger.info(f"Saved convergence speed comparison to {fig_path}")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Final accuracy comparison
    if final_accuracies:
        fig, ax = plt.subplots(figsize=(12, 6))
        run_names = list(final_accuracies.keys())
        accuracies = list(final_accuracies.values())
        bars = ax.bar(range(len(run_names)), accuracies, color="forestgreen", alpha=0.7, edgecolor="black", linewidth=1.5)
        
        # Annotate bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.005, 
                   f"{acc:.4f}", ha="center", va="top", fontsize=11, fontweight="bold", color="white")
        
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45, ha="right")
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title("Final Test Accuracy Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0.85, 1.0])
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        fig_path = comparison_dir / "comparison_final_accuracy_bar.pdf"
        plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
        logger.info(f"Saved final accuracy comparison to {fig_path}")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Relative speedup plot
    if len(convergence_speeds) >= 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate speedups relative to slowest method
        max_speed = max(convergence_speeds.values())
        speedups = {rid: ((max_speed - convergence_speeds[rid]) / max_speed * 100) for rid in convergence_speeds}
        
        run_names = list(speedups.keys())
        speedup_values = list(speedups.values())
        colors = ["green" if s >= 0 else "red" for s in speedup_values]
        bars = ax.bar(range(len(run_names)), speedup_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        
        # Annotate
        for bar, speedup in zip(bars, speedup_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, 
                   f"{speedup:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45, ha="right")
        ax.set_ylabel("Relative Speedup (%)", fontsize=12)
        ax.set_title("Relative Convergence Speed Improvement", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        fig_path = comparison_dir / "comparison_relative_speedup.pdf"
        plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
        logger.info(f"Saved relative speedup figure to {fig_path}")
        print(f"Generated figure: {fig_path}")
        plt.close()

def compute_aggregated_metrics(all_metrics: Dict[str, Dict]) -> Dict:
    """Compute aggregated metrics across runs."""
    
    # Separate proposed and baseline methods
    proposed_runs = {rid: m for rid, m in all_metrics.items() if "proposed" in rid.lower()}
    baseline_runs = {rid: m for rid, m in all_metrics.items() if "comparative" in rid.lower() or "baseline" in rid.lower()}
    
    # Extract convergence speeds
    proposed_speeds = {rid: m.get("convergence_speed_to_90_percent_final_accuracy") 
                      for rid, m in proposed_runs.items() 
                      if m.get("convergence_speed_to_90_percent_final_accuracy") is not None}
    baseline_speeds = {rid: m.get("convergence_speed_to_90_percent_final_accuracy") 
                      for rid, m in baseline_runs.items() 
                      if m.get("convergence_speed_to_90_percent_final_accuracy") is not None}
    
    # Extract final accuracies
    proposed_accuracies = {rid: m.get("final_test_accuracy") 
                          for rid, m in proposed_runs.items() 
                          if m.get("final_test_accuracy") is not None}
    baseline_accuracies = {rid: m.get("final_test_accuracy") 
                          for rid, m in baseline_runs.items() 
                          if m.get("final_test_accuracy") is not None}
    
    # Find best runs
    best_proposed = min(proposed_speeds.items(), key=lambda x: x[1]) if proposed_speeds else (None, None)
    best_baseline = min(baseline_speeds.items(), key=lambda x: x[1]) if baseline_speeds else (None, None)
    
    # Compute gap (for convergence speed: lower is better, so gap is positive when proposed is faster)
    gap = None
    if best_proposed[1] is not None and best_baseline[1] is not None:
        gap = (best_baseline[1] - best_proposed[1]) / best_baseline[1] * 100
    
    # Collect all metrics
    all_convergence_speeds = {}
    all_final_accuracies = {}
    
    for rid, m in all_metrics.items():
        if m.get("convergence_speed_to_90_percent_final_accuracy") is not None:
            all_convergence_speeds[rid] = m.get("convergence_speed_to_90_percent_final_accuracy")
        if m.get("final_test_accuracy") is not None:
            all_final_accuracies[rid] = m.get("final_test_accuracy")
    
    aggregated = {
        "primary_metric": "convergence_speed_to_90_percent_final_accuracy",
        "metrics": {
            "convergence_speed_to_90_percent_final_accuracy": all_convergence_speeds,
            "final_test_accuracy": all_final_accuracies,
        },
        "best_proposed": {
            "run_id": best_proposed[0],
            "value": best_proposed[1],
        } if best_proposed[0] else None,
        "best_baseline": {
            "run_id": best_baseline[0],
            "value": best_baseline[1],
        } if best_baseline[0] else None,
        "gap": gap,
    }
    
    return aggregated

def main():
    """Main evaluation script."""
    args = parse_args()
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    if not isinstance(run_ids, list):
        raise ValueError(f"run_ids must be a JSON list, got {type(run_ids)}")
    
    logger.info(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    # Load WandB config
    wandb_cfg = load_wandb_config()
    
    # Check if WandB is disabled
    if wandb_cfg.get("mode") == "disabled":
        logger.error("Cannot run evaluate.py in trial/disabled mode - no WandB data available")
        raise RuntimeError("evaluate.py requires full mode with WandB enabled")
    
    logger.info(f"Using WandB entity={wandb_cfg['entity']}, project={wandb_cfg['project']}")
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch and process all runs
    all_metrics = {}
    for run_id in run_ids:
        logger.info(f"\nProcessing run: {run_id}")
        
        # Fetch from WandB
        run_data = fetch_run_data_from_wandb(run_id, wandb_cfg["entity"], wandb_cfg["project"])
        
        # Process metrics
        if run_data:
            metrics = process_run_metrics(run_data, results_dir)
            if metrics:
                all_metrics[run_id] = metrics
        else:
            logger.warning(f"Failed to fetch data for run {run_id}")
    
    if not all_metrics:
        logger.error("No metrics collected, exiting")
        return
    
    logger.info(f"\nSuccessfully processed {len(all_metrics)} runs")
    
    # Generate comparison figures
    logger.info("\nGenerating comparison figures...")
    generate_comparison_figures(all_metrics, results_dir)
    
    # Compute aggregated metrics
    logger.info("\nComputing aggregated metrics...")
    aggregated = compute_aggregated_metrics(all_metrics)
    
    # Save aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    logger.info(f"Saved aggregated metrics to {aggregated_file}")
    print(f"Saved aggregated metrics: {aggregated_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Primary Metric: {aggregated['primary_metric']}")
    if aggregated["best_proposed"]:
        logger.info(f"Best Proposed (SARA): {aggregated['best_proposed']['run_id']} = {aggregated['best_proposed']['value']} epochs")
    if aggregated["best_baseline"]:
        logger.info(f"Best Baseline (RAdam): {aggregated['best_baseline']['run_id']} = {aggregated['best_baseline']['value']} epochs")
    if aggregated["gap"] is not None:
        logger.info(f"Performance Gap: {aggregated['gap']:.2f}% speedup")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()
