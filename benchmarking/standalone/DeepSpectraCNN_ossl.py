import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
import torch
from torch import nn, optim
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.net.base_net import DeepSpectraCNN
from src.training.trainer import Trainer
from src.utils.misc import TrainerConfig
from src.utils.dataset_loader import DatasetLoader
from src.utils.testing import RMSEP, ccc

# Constants
SEARCH_MAX_EPOCHS = 200
SEARCH_PATIENCE = 30
FINAL_MAX_EPOCHS = 300
FINAL_PATIENCE = 50

N_TRIALS_ARCH = 30
N_TRIALS_HP = 30
FINAL_SEEDS = list(range(10))

MODEL_TYPE = "DeepSpectraCNN_ossl"
DATASET_TYPE = "ossl"
BATCH_SIZE = 512

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_conv_config(depth, nf, ks):
    """Build convolutional layer configuration."""
    convs = []
    for i in range(depth):
        stride = 2 if i > 0 else 1
        n_filters = nf * (2 ** i)
        convs.append((n_filters, ks, stride))
    return convs

def build_loaders(data, batch_size):
    """Build data loaders."""
    import torch.utils.data as data_utils
    
    cal_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_cal"], dtype=torch.float32),
            torch.tensor(data["y_cal"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_val"], dtype=torch.float32),
            torch.tensor(data["y_val"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_test"], dtype=torch.float32),
            torch.tensor(data["y_test"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )
    return cal_loader, val_loader, test_loader

def make_model(arch_params, dropout, spec_dims, y_dim, mean, std, device):
    """Create DeepSpectraCNN instance."""
    model = DeepSpectraCNN(
        input_dim=spec_dims,
        mean=mean,
        std=std,
        kernel_size=arch_params["KS"],
        dropout=dropout,
        out_dims=y_dim,
    ).to(device)
    return model

def run_training(model, hp_params, cal_loader, val_loader, num_epochs, 
                 early_stopping_patience, trial=None, verbose=False, 
                 save_path=None, use_cosine_lr=True):
    """Run training with checkpointing."""
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=hp_params["LR"], weight_decay=hp_params["WD"])

    config = TrainerConfig(model_name=MODEL_TYPE)
    config.update_config(
        batch_size=hp_params.get("batch_size", BATCH_SIZE),
        learning_rate=hp_params["LR"],
        num_epochs=num_epochs,
        classification=False,
        save_path=save_path,
        use_cosine_lr=use_cosine_lr
    )

    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion,
        train_loader=cal_loader, 
        val_loader=val_loader, 
        config=config, 
        verbose=verbose,
        early_stopping_patience=early_stopping_patience, 
        trial=trial
    )
    
    train_losses, val_losses, val_metrics = trainer.train()
    best_val_score = max(float(np.mean(np.atleast_1d(m))) for m in val_metrics)
    
    return trainer, train_losses, val_losses, val_metrics, best_val_score

def cleanup_checkpoint(checkpoint_dir):
    """Clean up checkpoint directory after trial."""
    if checkpoint_dir is None:
        return
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        try:
            import shutil
            shutil.rmtree(checkpoint_dir)
        except Exception as e:
            print(f"Could not delete checkpoint {checkpoint_dir}: {e}")

# Stage 1: Architecture Search
def objective_architecture(trial, data, mean, std, spec_dims, y_dim, device, search_dir):
    """Optuna objective for DeepSpectraCNN architecture search."""
    arch_params = {
        "KS": trial.suggest_categorical("KS", [3, 5, 7, 11]),
    }
    fixed_hp = {"LR": 0.0005, "WD": 0.0015}
    dropout = 0.1
    set_seed(42)
    cal_loader, val_loader, _ = build_loaders(data, BATCH_SIZE)
    model = make_model(arch_params, dropout, spec_dims, y_dim, mean, std, device)
    trial_dir = search_dir / f"arch_trial_{trial.number:03d}"
    _, _, _, _, best_val_score = run_training(
        model, fixed_hp, cal_loader, val_loader,
        num_epochs=SEARCH_MAX_EPOCHS,
        early_stopping_patience=SEARCH_PATIENCE,
        trial=trial,
        verbose=False,
        save_path=trial_dir / MODEL_TYPE,
        use_cosine_lr=True,
    )
    cleanup_checkpoint(trial_dir)
    return best_val_score

def objective_hyperparams(trial, data, mean, std, spec_dims, y_dim, fixed_arch, device, search_dir):
    """Optuna objective for hyperparameter search."""
    hp_params = {
        "LR": trial.suggest_float("LR", 1e-4, 1e-2, log=True),
        "WD": trial.suggest_float("WD", 1e-5, 1e-2, log=True),
        "DP": trial.suggest_float("DP", 0.0, 0.5),
    }

    set_seed(42)
    cal_loader, val_loader, _ = build_loaders(data, BATCH_SIZE)
    model = make_model(fixed_arch, hp_params["DP"], spec_dims, y_dim, mean, std, device)

    trial_dir = search_dir / f"hp_trial_{trial.number:03d}"
    _, _, _, _, best_val_score = run_training(
        model, hp_params, cal_loader, val_loader,
        num_epochs=SEARCH_MAX_EPOCHS, 
        early_stopping_patience=SEARCH_PATIENCE,
        trial=trial, 
        verbose=False,
        save_path=trial_dir / MODEL_TYPE,
        use_cosine_lr=True
    )
    
    cleanup_checkpoint(trial_dir)
    return best_val_score

# Stage 3: Final Multi-seed Evaluation
def evaluate_test(model, model_path, test_loader, config):
    """Evaluate model on test set."""
    from src.utils import test_benchmark
    
    Y, y_pred = test_benchmark(model, model_path, test_loader, config)
    
    # Compute metrics
    y_pred_np = np.array(y_pred)
    Y_np = np.array(Y)
    
    perf = {
        "ccc": ccc(Y_np, y_pred_np),
        "r2": 1 - np.sum((Y_np - y_pred_np) ** 2) / (np.sum((Y_np - np.mean(Y_np)) ** 2) + 1e-12),
        "rmsep": RMSEP(Y_np, y_pred_np)
    }
    perf = {k: float(np.ravel(v)[0]) for k, v in perf.items()}
    return perf, Y_np, y_pred_np

def plot_diagnostics(Y, y_pred, perf, out_dir, tag):
    """Create diagnostic plots."""
    import matplotlib.pyplot as plt
    
    lims = [min(np.min(Y), np.min(y_pred)), max(np.max(Y), np.max(y_pred))]
    typ = tag

    fig, ax = plt.subplots(figsize=(8, 6))
    hexbin = ax.hexbin(Y, y_pred, gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
    cb.set_label('Density')
    ax.plot(lims, lims, 'k-', label=typ)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Expected Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('')
    ax.text(
        0.98, 0.02,
        f"CCC: {perf['ccc']:.2f}\nR²: {perf['r2']:.2f}\nRMSEP: {perf['rmsep']:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        va='bottom', ha='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
        color='red',
        fontweight='bold',
        fontfamily='serif'
    )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=5, fontsize=12)
    plt.tight_layout()
    plt.grid()
    hexbin_pdf_path = out_dir / "fig_hexbin.pdf"
    plt.savefig(hexbin_pdf_path, format='pdf')
    plt.close('all')

def plot_seed_prediction_variability(predictions, true_values, out_dir, tag):
    """Summarize prediction variability across seeds, plotted against true values."""
    import matplotlib.pyplot as plt

    preds = np.stack(predictions, axis=0)
    true_vals = np.asarray(true_values).reshape(-1)
    mean_pred = np.mean(preds, axis=0).reshape(-1)
    std_pred = np.std(preds, axis=0).reshape(-1)

    order = np.argsort(true_vals)
    true_sorted = true_vals[order]
    mean_sorted = mean_pred[order]
    std_sorted = std_pred[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_sorted, mean_sorted, color='#1f77b4', linewidth=2, label='Mean prediction')
    ax.fill_between(true_sorted, mean_sorted - std_sorted, mean_sorted + std_sorted,
                    color='#1f77b4', alpha=0.2, label='±1 SD')
    lims = [min(true_sorted.min(), mean_sorted.min()), max(true_sorted.max(), mean_sorted.max())]
    ax.plot(lims, lims, color='#d62728', linewidth=1.5, linestyle='--', label='Identity (y = x)')
    ax.set_xlabel('True value')
    ax.set_ylabel('Prediction')
    ax.set_title(f'Prediction variability across {len(predictions)} seeds ({tag})')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='best')
    fig.tight_layout()
    pdf_path = out_dir / f"prediction_variability_{tag}.pdf"
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)

def plot_training_history(train_losses, val_losses, val_metrics, out_dir, tag, maxplot_loss=10):
    """Plot training and validation loss with metric curves."""
    import matplotlib.pyplot as plt

    train_losses_np = [loss.detach().cpu().numpy() if torch.is_tensor(loss) else np.asarray(loss) for loss in train_losses]
    val_losses_np = [loss.detach().cpu().numpy() if torch.is_tensor(loss) else np.asarray(loss) for loss in val_losses]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    train_color = '#1f77b4'
    val_color = '#ff7f0e'
    metric_color = '#2ca02c'

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=train_color)
    ax1.plot(train_losses_np, label='Training Loss', color=train_color, linewidth=2)
    ax1.plot(val_losses_np, label='Validation Loss', color=val_color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=train_color)
    ax1.set_ylim(0, min(maxplot_loss, max(max(train_losses_np), max(val_losses_np)) * 1.1))
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Metric', color=metric_color)
    ax2.tick_params(axis='y', labelcolor=metric_color)

    if len(val_metrics) > 0 and isinstance(val_metrics[0], (list, tuple, np.ndarray)):
        for i in range(len(val_metrics[0])):
            metric_scores = [scores[i] for scores in val_metrics]
            ax2.plot(metric_scores, label=f'R² Score y{i}', linestyle='--', color=metric_color, linewidth=2)
    else:
        ax2.plot(val_metrics, label='R² Score', linestyle='--', color=metric_color, linewidth=2)

    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Loss and Metrics')
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    pdf_path = out_dir / f"Training_{tag}.pdf"
    plt.savefig(pdf_path, format='pdf')
    plt.close(fig)

def save_retained_architecture_training_plot(data, mean, std, spec_dims, y_dim, best_arch, device, out_dir):
    """Save a training-history plot for the retained architecture after phase 1."""
    fixed_hp = {"LR": 0.0001, "WD": 0.0015}
    dropout = 0.1

    set_seed(42)
    cal_loader, val_loader, _ = build_loaders(data, BATCH_SIZE)
    model = make_model(best_arch, dropout, spec_dims, y_dim, mean, std, device)

    trainer, train_losses, val_losses, val_metrics, _ = run_training(
        model,
        fixed_hp,
        cal_loader,
        val_loader,
        num_epochs=SEARCH_MAX_EPOCHS,
        early_stopping_patience=SEARCH_PATIENCE,
        trial=None,
        verbose=False,
        save_path=out_dir / f"{MODEL_TYPE}_retained_arch",
        use_cosine_lr=True,
    )
    plot_training_history(train_losses, val_losses, val_metrics, out_dir, tag='retained_architecture')
    return trainer

def run_final_multiseed(data, mean, std, spec_dims, y_dim, best_arch, best_hp, device, final_dir):
    """Run multi-seed final evaluation."""
    seed_metrics = []
    seed_predictions = []
    seed_true_values = None
    
    for seed in FINAL_SEEDS:
        seed_dir = final_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        cal_loader, val_loader, test_loader = build_loaders(data, BATCH_SIZE)
        model = make_model(best_arch, best_hp["DP"], spec_dims, y_dim, mean, std, device)

        trainer, _, _, _, _ = run_training(
            model, best_hp, cal_loader, val_loader,
            num_epochs=FINAL_MAX_EPOCHS, 
            early_stopping_patience=FINAL_PATIENCE,
            trial=None, 
            verbose=False,
            save_path=seed_dir / MODEL_TYPE,
            use_cosine_lr=True
        )

        # Load best model for testing
        best_model_path = seed_dir / f"{MODEL_TYPE}_best.pth"
        if not best_model_path.exists():
            # If best model wasn't saved, use the final checkpoint
            checkpoint_path = seed_dir / "checkpoint.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                torch.save(checkpoint['model_state_dict'], best_model_path)

        perf, Y, y_pred = evaluate_test(model, best_model_path, test_loader, trainer.config)
        plot_diagnostics(Y, y_pred, perf, seed_dir, tag=DATASET_TYPE)
        if seed_true_values is None:
            seed_true_values = Y
        seed_predictions.append(y_pred)

        # Count parameters
        nb_params = sum(p.numel() for p in model.parameters())
        
        metrics_dict = {
            "seed": seed,
            "ccc": perf["ccc"],
            "r2": perf["r2"],
            "rmsep": perf["rmsep"],
            "best_epoch": trainer.best_epoch,
            "n_parameters": nb_params,
        }
        
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
        seed_metrics.append(metrics_dict)
    
    if seed_true_values is not None and len(seed_predictions) > 0:
        plot_seed_prediction_variability(seed_predictions, seed_true_values, final_dir, tag=DATASET_TYPE)
    
    return seed_metrics

def summarise_seed_metrics(seed_metrics):
    """Summarize metrics across seeds."""
    summary = {}
    for key in ["ccc", "r2", "rmsep"]:
        values = np.array([m[key] for m in seed_metrics])
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary

def main():
    """Main benchmarking pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data path for OSSL
    root = os.getcwd()
    data_path = "D:/data/dataset/Ossl/ossl_database.mat"
    dataset = {"data_path": data_path, "dataset_type": DATASET_TYPE}

    set_seed(42)
    data = DatasetLoader.load(dataset)
    mean = np.mean(data["x_cal"], axis=0)
    std = np.std(data["x_cal"], axis=0)
    spec_dims = data["x_cal"].shape[1]
    y_dim = data["y_cal"].shape[1]
    
    print(f"Spectra dimensions: {spec_dims}, Output dimensions: {y_dim}")

    # Setup directories
    root = Path(__file__).parent.parent
    benchmark_root = root / "Benchmark" / DATASET_TYPE / MODEL_TYPE
    arch_search_dir = benchmark_root / "stage1_architecture_search"
    hp_search_dir = benchmark_root / "stage2_hyperparameter_search"
    final_dir = benchmark_root / "stage3_final"
    
    for d in (arch_search_dir, hp_search_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Architecture Search ---
    print("\n" + "="*60)
    print("Stage 1: Architecture Search")
    print("="*60)
    
    study_arch = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(n_warmup_steps=10, n_min_trials=5)
    )
    
    study_arch.optimize(
        lambda trial: objective_architecture(trial, data, mean, std, spec_dims, y_dim, device, arch_search_dir),
        n_trials=N_TRIALS_ARCH,
        timeout=None,
        show_progress_bar=True
    )
    
    best_arch = study_arch.best_params
    print(f"\nBest architecture: {best_arch}")
    print(f"Best validation score: {study_arch.best_value:.4f}")

    save_retained_architecture_training_plot(
        data, mean, std, spec_dims, y_dim, best_arch, device, arch_search_dir
    )
    
    # Save results
    study_arch.trials_dataframe().to_csv(arch_search_dir / "trials.csv", index=False)
    with open(arch_search_dir / "best_architecture.json", "w") as f:
        json.dump(best_arch, f, indent=2)
    
    # Save study for later analysis
    with open(arch_search_dir / "study.pkl", "wb") as f:
        import pickle
        pickle.dump(study_arch, f)

    # --- Stage 2: Hyperparameter Search ---
    print("\n" + "="*60)
    print("Stage 2: Hyperparameter Search")
    print("="*60)
    
    study_hp = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(n_warmup_steps=10, n_min_trials=5)
    )
    
    study_hp.optimize(
        lambda trial: objective_hyperparams(trial, data, mean, std, spec_dims, y_dim, best_arch, device, hp_search_dir),
        n_trials=N_TRIALS_HP,
        timeout=None,
        show_progress_bar=True
    )
    
    best_hp = study_hp.best_params
    print(f"\nBest hyperparameters: {best_hp}")
    print(f"Best validation score: {study_hp.best_value:.4f}")
    
    # Save results
    study_hp.trials_dataframe().to_csv(hp_search_dir / "trials.csv", index=False)
    with open(hp_search_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_hp, f, indent=2)
    
    with open(hp_search_dir / "study.pkl", "wb") as f:
        pickle.dump(study_hp, f)

    # --- Stage 3: Multi-seed Final Evaluation ---
    print("\n" + "="*60)
    print("Stage 3: Multi-seed Final Evaluation")
    print("="*60)
    
    seed_metrics = run_final_multiseed(
        data, mean, std, spec_dims, y_dim, best_arch, best_hp, device, final_dir
    )
    
    summary = summarise_seed_metrics(seed_metrics)
    
    # Save final results
    with open(final_dir / "seed_metrics.json", "w") as f:
        json.dump(seed_metrics, f, indent=2)
    with open(final_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value['mean']:.4f} ± {value['std']:.4f}")

if __name__ == "__main__":
    main()