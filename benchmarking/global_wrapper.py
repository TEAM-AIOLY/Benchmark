"""Shared Optuna benchmark runner for all model/dataset combinations."""
from __future__ import annotations

import json
import pickle
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import torch
from torch import nn, optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.training.trainer import Trainer
from src.utils.misc import TrainerConfig
from src.utils.dataset_loader import DatasetLoader


@dataclass
class BenchmarkSpec:
    model_type: str
    dataset_type: str
    data_path: str
    build_model: Callable[..., torch.nn.Module]
    architecture_space: Callable[[optuna.Trial], dict[str, Any]]
    hyperparameter_space: Callable[[optuna.Trial], dict[str, Any]]
    batch_size: int = 512
    search_max_epochs: int = 200
    search_patience: int = 30
    final_max_epochs: int = 300
    final_patience: int = 50
    n_trials_arch: int = 30
    n_trials_hp: int = 30
    final_seeds: tuple[int, ...] = tuple(range(20))
    classification: bool = False
    fixed_architecture_lr: float = 1e-5
    fixed_architecture_wd: float = 1.5e-3
    architecture_plot: Callable[..., None] | None = None
    evaluation_plot: Callable[..., None] | None = None
    variability_plot: Callable[..., None] | None = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(data: dict[str, Any], batch_size: int, classification: bool):
    import torch.utils.data as data_utils

    def loader(split: str, shuffle: bool, drop_last: bool = False):
        y = data[f"y_{split}"]
        if classification:
            y = np.argmax(y, axis=1) if np.asarray(y).ndim > 1 else y
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        return data_utils.DataLoader(
            data_utils.TensorDataset(
                torch.tensor(data[f"x_{split}"], dtype=torch.float32), y_tensor
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

    return loader("cal", True, True), loader("val", False), loader("test", False)


def _best_validation_score(metrics: list[Any], classification: bool) -> float:
    values = []
    for metric in metrics:
        if torch.is_tensor(metric):
            values.append(float(metric.detach().cpu().mean().item()))
        else:
            values.append(float(np.asarray(metric).mean()))
    return max(values) if values else float("-inf")


def run_training(spec: BenchmarkSpec, model: nn.Module, hp: dict[str, Any],
                 cal_loader, val_loader, epochs: int, patience: int,
                 device: torch.device, trial=None, save_path=None):
    criterion = nn.CrossEntropyLoss() if spec.classification else nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(
        model.parameters(), lr=hp["LR"], weight_decay=hp["WD"]
    )
    config = TrainerConfig(model_name=spec.model_type)
    config.update_config(
        batch_size=hp.get("batch_size", spec.batch_size),
        learning_rate=hp["LR"],
        num_epochs=epochs,
        classification=spec.classification,
        save_path=save_path,
        use_cosine_lr=True,
        device=device,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=cal_loader,
        val_loader=val_loader,
        config=config,
        verbose=False,
        early_stopping_patience=patience,
        trial=trial,
    )
    train_losses, val_losses, val_metrics = trainer.train()
    return (
        trainer,
        train_losses,
        val_losses,
        val_metrics,
        _best_validation_score(val_metrics, spec.classification),
    )


def _cleanup(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _objective_architecture(spec, context, device, search_dir):
    def objective(trial):
        arch = spec.architecture_space(trial)
        print(f"Starting architecture trial {trial.number + 1}", flush=True)
        set_seed(42)
        cal, val, _ = build_loaders(context["data"], spec.batch_size, spec.classification)
        model = spec.build_model(arch, 0.1, **context["model_args"], device=device)
        trial_dir = search_dir / f"arch_trial_{trial.number:03d}"
        result = run_training(
            spec, model, {"LR": spec.fixed_architecture_lr, "WD": spec.fixed_architecture_wd},
            cal, val, spec.search_max_epochs, spec.search_patience, device, trial,
            trial_dir / spec.model_type,
        )
        _cleanup(trial_dir)
        print(f"Finished architecture trial {trial.number + 1}", flush=True)
        return result[-1]
    return objective


def _objective_hyperparameters(spec, context, best_arch, device, search_dir):
    def objective(trial):
        hp = spec.hyperparameter_space(trial)
        print(f"Starting hyperparameter trial {trial.number + 1}", flush=True)
        set_seed(42)
        cal, val, _ = build_loaders(context["data"], spec.batch_size, spec.classification)
        model = spec.build_model(best_arch, hp.get("DP", 0.1), **context["model_args"], device=device)
        trial_dir = search_dir / f"hp_trial_{trial.number:03d}"
        result = run_training(
            spec, model, hp, cal, val, spec.search_max_epochs, spec.search_patience,
            device, trial, trial_dir / spec.model_type,
        )
        _cleanup(trial_dir)
        print(f"Finished hyperparameter trial {trial.number + 1}", flush=True)
        return result[-1]
    return objective


def _study(objective, trials):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(n_warmup_steps=10, n_min_trials=5),
    )
    study.optimize(objective, n_trials=trials, timeout=None, show_progress_bar=True)
    return study


def _default_metrics(y_true, predictions, classification: bool) -> dict[str, Any]:
    if classification:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        true = np.asarray(y_true).reshape(-1)
        pred = np.asarray(predictions)
        pred = np.argmax(pred, axis=1) if pred.ndim > 1 else pred.reshape(-1)
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "f1_macro": float(f1_score(true, pred, average="macro")),
            "f1_weighted": float(f1_score(true, pred, average="weighted")),
            "confusion_matrix": confusion_matrix(true, pred).tolist(),
        }
    from src.utils.testing import RMSEP, ccc
    true = np.asarray(y_true)
    pred = np.asarray(predictions)
    return {
        "ccc": float(np.ravel(ccc(true, pred))[0]),
        "r2": float(1 - np.sum((true - pred) ** 2) / (np.sum((true - np.mean(true)) ** 2) + 1e-12)),
        "rmsep": float(np.ravel(RMSEP(true, pred))[0]),
    }


def run_benchmark(spec: BenchmarkSpec) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    data = DatasetLoader.load({"data_path": spec.data_path, "dataset_type": spec.dataset_type})
    model_args = {
        "spec_dims": data["x_cal"].shape[1],
        "y_dim": data["y_cal"].shape[1],
        "mean": np.mean(data["x_cal"], axis=0),
        "std": np.std(data["x_cal"], axis=0),
    }
    context = {"data": data, "model_args": model_args}
    root = Path(__file__).parent.parent / "Benchmark" / spec.dataset_type / spec.model_type
    arch_dir, hp_dir, final_dir = (root / name for name in ("stage1_architecture_search", "stage2_hyperparameter_search", "stage3_final"))
    for directory in (arch_dir, hp_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60 + "\nStage 1: Architecture Search\n" + "=" * 60)
    arch_study = _study(_objective_architecture(spec, context, device, arch_dir), spec.n_trials_arch)
    best_arch = arch_study.best_params
    arch_study.trials_dataframe().to_csv(arch_dir / "trials.csv", index=False)
    (arch_dir / "best_architecture.json").write_text(json.dumps(best_arch, indent=2))
    with (arch_dir / "study.pkl").open("wb") as handle:
        pickle.dump(arch_study, handle)

    print("\n" + "=" * 60 + "\nStage 2: Hyperparameter Search\n" + "=" * 60)
    hp_study = _study(_objective_hyperparameters(spec, context, best_arch, device, hp_dir), spec.n_trials_hp)
    best_hp = hp_study.best_params
    hp_study.trials_dataframe().to_csv(hp_dir / "trials.csv", index=False)
    (hp_dir / "best_hyperparameters.json").write_text(json.dumps(best_hp, indent=2))
    with (hp_dir / "study.pkl").open("wb") as handle:
        pickle.dump(hp_study, handle)

    print("\n" + "=" * 60 + "\nStage 3: Multi-seed Final Evaluation\n" + "=" * 60)
    seed_metrics = []
    for seed in spec.final_seeds:
        seed_dir = final_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        cal, val, test = build_loaders(data, spec.batch_size, spec.classification)
        model = spec.build_model(best_arch, best_hp.get("DP", 0.1), **model_args, device=device)
        trainer, _, _, _, _ = run_training(spec, model, best_hp, cal, val, spec.final_max_epochs, spec.final_patience, device, save_path=seed_dir / spec.model_type)
        best_path = seed_dir / f"{spec.model_type}_best.pth"
        if trainer.best_checkpoint_path and trainer.best_checkpoint_path.exists():
            checkpoint = torch.load(trainer.best_checkpoint_path, map_location=device)
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
            torch.save(state, best_path)
        elif not best_path.exists():
            torch.save(trainer.model.state_dict(), best_path)
        from src.utils.testing import test_benchmark

        y_true, predictions = test_benchmark(model, best_path, test, trainer.config)
        if spec.classification:
            y_true_for_metrics = np.argmax(y_true, axis=1) if np.asarray(y_true).ndim > 1 else y_true
        else:
            y_true_for_metrics = y_true
        metrics = _default_metrics(y_true_for_metrics, predictions, spec.classification)
        metrics.update({"seed": seed, "best_epoch": trainer.best_epoch, "n_parameters": sum(p.numel() for p in model.parameters())})
        (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        seed_metrics.append(metrics)

    metric_keys = [key for key in seed_metrics[0] if key not in {"seed", "best_epoch", "n_parameters", "confusion_matrix"}]
    summary = {key: {"mean": float(np.mean([item[key] for item in seed_metrics])), "std": float(np.std([item[key] for item in seed_metrics]))} for key in metric_keys}
    (final_dir / "seed_metrics.json").write_text(json.dumps(seed_metrics, indent=2))
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for key, values in summary.items():
        print(f"{key}: {values['mean']:.4f} +/- {values['std']:.4f}")
