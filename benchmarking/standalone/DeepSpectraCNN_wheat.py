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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.net.base_net import DeepSpectraCNN
from src.training.trainer import Trainer
from src.utils.misc import TrainerConfig
from src.utils.dataset_loader import DatasetLoader

# Constants
SEARCH_MAX_EPOCHS = 500
SEARCH_PATIENCE = 30
FINAL_MAX_EPOCHS = 1000
FINAL_PATIENCE = 50

N_TRIALS_ARCH = 100
N_TRIALS_HP = 100
FINAL_SEEDS = list(range(10))

MODEL_TYPE = "DeepSpectraCNN_wheat"
DATASET_TYPE = "wheat"
BATCH_SIZE = 512
NUM_CLASSES = None  # Will be determined from data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_loaders(data, batch_size):
    """Build data loaders."""
    import torch.utils.data as data_utils
    
    cal_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_cal"], dtype=torch.float32),
            torch.tensor(np.argmax(data["y_cal"], axis=1), dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_val"], dtype=torch.float32),
            torch.tensor(np.argmax(data["y_val"], axis=1), dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=False
    )
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_test"], dtype=torch.float32),
            torch.tensor(np.argmax(data["y_test"], axis=1), dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=False
    )
    return cal_loader, val_loader, test_loader

def make_model(arch_params, dropout, spec_dims, y_dim, mean, std, device):
    model = DeepSpectraCNN(
        input_dim=spec_dims,
        mean=mean,
        std=std,
        kernel_size=arch_params['KS'],
        dropout=dropout,
        out_dims=y_dim,
    ).to(device)
    return model

def run_training(model, hp_params, cal_loader, val_loader, num_epochs, 
                 early_stopping_patience, trial=None, verbose=False, 
                 save_path=None, use_cosine_lr=True):
    """Run training with checkpointing for classification."""
    criterion = nn.CrossEntropyLoss()  # Classification loss
    optimizer = optim.Adam(model.parameters(), lr=hp_params["LR"], weight_decay=hp_params["WD"])

    config = TrainerConfig(model_name=MODEL_TYPE)
    config.update_config(
        batch_size=hp_params.get("batch_size", BATCH_SIZE),
        learning_rate=hp_params["LR"],
        num_epochs=num_epochs,
        classification=True,  # Enable classification mode
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
    # For classification, val_metrics typically contains accuracy or F1 score
    metric_values = []
    for m in val_metrics:
        if torch.is_tensor(m):
            metric_values.append(float(m.detach().cpu().mean().item()))
        else:
            metric_values.append(float(np.mean(np.atleast_1d(np.asarray(m)))))
    best_val_score = max(metric_values) if metric_values else float("-inf")
    
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
    """Optuna objective for architecture search."""
    arch_params = {
        "KS": trial.suggest_categorical("KS", [3, 5, 7, 11]),
        "NF": trial.suggest_int("NF", 1, 8),
        "FC": trial.suggest_categorical("FC", [32, 64, 128, 256]),
    }
    
    # Fixed hyperparameters for architecture search
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
        use_cosine_lr=True
    )
    
    cleanup_checkpoint(trial_dir)
    return best_val_score

# Stage 2: Hyperparameter Search
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
def evaluate_test_classification(model, model_path, test_loader, config):
    """Evaluate model on test set for classification."""
    from src.utils import test_benchmark
    
    Y, y_pred = test_benchmark(model, model_path, test_loader, config)
    
    # Convert predictions to class labels (assuming logits)
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    Y_np = np.argmax(np.asarray(Y), axis=1) if np.asarray(Y).ndim > 1 else np.asarray(Y).flatten()
    
    # Compute metrics
    perf = {
        "accuracy": accuracy_score(Y_np, y_pred_labels),
        "f1_macro": f1_score(Y_np, y_pred_labels, average='macro'),
        "f1_weighted": f1_score(Y_np, y_pred_labels, average='weighted'),
        "confusion_matrix": confusion_matrix(Y_np, y_pred_labels).tolist()
    }
    
    return perf, Y_np, y_pred_labels

def plot_diagnostics_classification(Y, y_pred, perf, out_dir, tag):
    """Create diagnostic plots for classification."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array(perf["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='viridis', values_format='d')
    ax.set_title(f'Confusion Matrix - {tag}')
    plt.tight_layout()
    cm_pdf_path = out_dir / f"confusion_matrix_{tag}.pdf"
    plt.savefig(cm_pdf_path, format='pdf')
    plt.close('all')

def plot_seed_prediction_variability_classification(predictions, true_values, out_dir, tag):
    """Summarize prediction variability across seeds, plotted against true values."""
    import matplotlib.pyplot as plt
    
    # Calculate prediction stability
    preds = np.stack(predictions, axis=0)
    true_vals = np.asarray(true_values)
    
    # Calculate majority vote and agreement
    majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
    agreement = np.mean(preds == majority_vote, axis=0)
    
    order = np.argsort(true_vals)
    true_sorted = true_vals[order]
    majority_vote_sorted = majority_vote[order]
    agreement_sorted = agreement[order]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot predictions against true values
    ax1.plot(true_sorted, majority_vote_sorted, color='#1f77b4', linewidth=2, label='Majority vote')
    lims = [min(true_sorted.min(), majority_vote_sorted.min()), max(true_sorted.max(), majority_vote_sorted.max())]
    ax1.plot(lims, lims, color='#d62728', linewidth=1.5, linestyle='--', label='Identity (y = x)')
    ax1.set_xlabel('True class')
    ax1.set_ylabel('Predicted class')
    ax1.set_title(f'Majority vote predictions across {len(predictions)} seeds ({tag})')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='best')
    
    # Plot agreement against true values
    ax2.bar(true_sorted, agreement_sorted, color='#2ca02c', alpha=0.7)
    ax2.set_xlabel('True class')
    ax2.set_ylabel('Agreement among seeds')
    ax2.set_title('Prediction agreement across seeds')
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
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
            ax2.plot(metric_scores, label=f'Metric y{i}', linestyle='--', color=metric_color, linewidth=2)
    else:
        ax2.plot(val_metrics, label='Validation Metric', linestyle='--', color=metric_color, linewidth=2)

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

def run_final_multiseed_classification(data, mean, std, spec_dims, y_dim, best_arch, best_hp, device, final_dir):
    """Run multi-seed final evaluation for classification."""
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
            checkpoint_path = seed_dir / "checkpoint.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                torch.save(checkpoint['model_state_dict'], best_model_path)

        perf, Y, y_pred = evaluate_test_classification(model, best_model_path, test_loader, trainer.config)
        plot_diagnostics_classification(Y, y_pred, perf, seed_dir, tag=DATASET_TYPE)
        
        if seed_true_values is None:
            seed_true_values = Y
        seed_predictions.append(y_pred)

        # Count parameters
        nb_params = sum(p.numel() for p in model.parameters())
        
        metrics_dict = {
            "seed": seed,
            "accuracy": perf["accuracy"],
            "f1_macro": perf["f1_macro"],
            "f1_weighted": perf["f1_weighted"],
            "best_epoch": trainer.best_epoch,
            "n_parameters": nb_params,
        }
        
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
        seed_metrics.append(metrics_dict)
    
    if seed_true_values is not None and len(seed_predictions) > 0:
        plot_seed_prediction_variability_classification(
            seed_predictions, seed_true_values, final_dir, tag=DATASET_TYPE
        )
    
    return seed_metrics

def summarise_seed_metrics_classification(seed_metrics):
    """Summarize metrics across seeds."""
    summary = {}
    for key in ["accuracy", "f1_macro", "f1_weighted"]:
        values = np.array([m[key] for m in seed_metrics])
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary

def main():
    """Main benchmarking pipeline for classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data path for Wheat
    root = os.getcwd()
    data_path = "D:/data/dataset/Wheat_dt/"
    dataset = {"data_path": data_path, "dataset_type": DATASET_TYPE}

    set_seed(42)
    data = DatasetLoader.load(dataset)
    mean = np.mean(data["x_cal"], axis=0)
    std = np.std(data["x_cal"], axis=0)
    spec_dims = data["x_cal"].shape[1]
    y_dim = data["y_cal"].shape[1]
    
    # Determine number of classes
    global NUM_CLASSES
    NUM_CLASSES = y_dim
    
    print(f"Spectra dimensions: {spec_dims}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Output dimensions: {y_dim}")

    # Setup directories
    root_path = Path(__file__).parent.parent
    benchmark_root = root_path / "Benchmark" / DATASET_TYPE / MODEL_TYPE
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
    
    seed_metrics = run_final_multiseed_classification(
        data, mean, std, spec_dims, y_dim, best_arch, best_hp, device, final_dir
    )
    
    summary = summarise_seed_metrics_classification(seed_metrics)
    
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