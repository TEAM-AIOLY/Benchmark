"""Per-model/per-dataset configuration for :mod:`global_wrapper`."""
from __future__ import annotations

from functools import partial
from typing import Any

import optuna

from src.net import Arioul_net, Hmar_Net, ViT_1D
from src.net.base_net import DeepSpectraCNN, ResNet18_1D, ResNet34_1D, ResNet50_1D, ResNet101_1D
from .global_wrapper import BenchmarkSpec


_DATA_PATHS = {
    "mango_new": "D:/data/dataset/Mango/mango_splits.mat",
    "ossl": "D:/data/dataset/OSSL/ossl_splits.mat",
    "wheat": "D:/data/dataset/Wheat_dt/",
}


def _conv_config(depth, nf, ks):
    return [(nf * (2 ** index), ks, 1 if index == 0 else 2) for index in range(depth)]


def _arioul(arch, dropout, spec_dims, y_dim, mean, std, device):
    return Arioul_net(input_dims=spec_dims, conv_config=_conv_config(arch["DEPTH"], arch["NF"], arch["KS"]), fc1_dims=arch["FC"], dropout=dropout, out_dims=y_dim, mean=mean, std=std).to(device)


def _hmar(arch, dropout, spec_dims, y_dim, mean, std, device):
    return Hmar_Net(input_dims=spec_dims, mean=mean, std=std, n_filters=arch["NF"], kernel_size=arch["KS"], fc1_dims=arch["FC"], dropout=dropout, out_dims=y_dim).to(device)


def _deep_spectra(arch, dropout, spec_dims, y_dim, mean, std, device):
    return DeepSpectraCNN(input_dim=spec_dims, mean=mean, std=std, kernel_size=arch["KS"], dropout=dropout, out_dims=y_dim).to(device)


def _resnet(factory, arch, dropout, spec_dims, y_dim, mean, std, device):
    return factory(in_channel=1, out_dims=y_dim, mean=mean, std=std, dropout=dropout, inplanes=arch["NF"]).to(device)


def _vit(arch, dropout, spec_dims, y_dim, mean, std, device):
    return ViT_1D(mean=mean, std=std, seq_len=spec_dims, patch_size=arch["PS"], dim_embed=arch["DE"], trans_layers=arch["TL"], heads=arch["HDS"], mlp_dim=arch["MLP"], dropout=dropout, out_dims=y_dim).to(device)


def _cnn_architecture(trial: optuna.Trial):
    return {"KS": trial.suggest_categorical("KS", [3, 5, 7, 11]), "NF": trial.suggest_int("NF", 1, 8), "FC": trial.suggest_categorical("FC", [32, 64, 128, 256])}


def _arioul_architecture(dataset):
    def space(trial: optuna.Trial):
        if dataset == "ossl":
            return {"DEPTH": trial.suggest_int("DEPTH", 1, 3), "KS": trial.suggest_categorical("KS", [7, 11]), "NF": trial.suggest_int("NF", 1, 5), "FC": trial.suggest_categorical("FC", [128, 256])}
        if dataset == "wheat":
            return {"DEPTH": trial.suggest_int("DEPTH", 1, 4), "KS": trial.suggest_categorical("KS", [3, 5, 7, 11]), "NF": trial.suggest_int("NF", 1, 3), "FC": trial.suggest_categorical("FC", [32, 64, 128, 256])}
        return {"DEPTH": trial.suggest_int("DEPTH", 1, 5), "KS": trial.suggest_categorical("KS", [3, 5, 7, 11]), "NF": trial.suggest_int("NF", 1, 7), "FC": trial.suggest_categorical("FC", [32, 64, 128, 256])}
    return space


def _resnet_architecture(trial: optuna.Trial):
    return {"NF": trial.suggest_categorical("NF", [4, 8, 16, 32])}


def _vit_architecture(trial: optuna.Trial):
    return {"PS": trial.suggest_categorical("PS", [30, 40, 50, 100, 110]), "DE": trial.suggest_categorical("DE", [32, 64, 128]), "TL": trial.suggest_int("TL", 4, 20), "HDS": trial.suggest_int("HDS", 4, 20), "MLP": trial.suggest_categorical("MLP", [32, 64, 128])}


def _hp(trial: optuna.Trial):
    return {"LR": trial.suggest_float("LR", 1e-4, 1e-2, log=True), "WD": trial.suggest_float("WD", 1e-5, 1e-2, log=True), "DP": trial.suggest_float("DP", 0.0, 0.5)}


def _vit_hp(trial: optuna.Trial):
    return {"LR": trial.suggest_float("LR", 1e-5, 1e-2, log=True), "WD": trial.suggest_float("WD", 1e-5, 1e-2, log=True)}


def get_spec(model: str, dataset: str) -> BenchmarkSpec:
    classification = dataset == "wheat"
    model_key = model.lower()
    if model_key == "arioulnet":
        factory, architecture = _arioul, _arioul_architecture(dataset)
    elif model_key == "hmar_net":
        factory, architecture = _hmar, _cnn_architecture
    elif model_key == "deepspectracnn":
        factory, architecture = _deep_spectra, _cnn_architecture
    elif model_key.startswith("resnet"):
        factories = {"resnet18_1d": ResNet18_1D, "resnet34_1d": ResNet34_1D, "resnet50_1d": ResNet50_1D, "resnet101_1d": ResNet101_1D}
        factory = partial(_resnet, factories[model_key])
        architecture = _resnet_architecture
    elif model_key in {"vit", "vit_1d"}:
        factory, architecture = _vit, _vit_architecture
    else:
        raise ValueError(f"Unknown model: {model}")

    model_name = "ViT_1D" if model_key in {"vit", "vit_1d"} else model
    return BenchmarkSpec(
        model_type=f"{model_name}_{dataset}",
        dataset_type=dataset,
        data_path=_DATA_PATHS[dataset],
        build_model=factory,
        architecture_space=architecture,
        hyperparameter_space=_vit_hp if model_key in {"vit", "vit_1d"} else _hp,
        batch_size=256 if model_key in {"arioulnet", "vit", "vit_1d"} else 512,
        search_max_epochs=600 if model_key in {"vit", "vit_1d"} else (500 if classification else 200),
        search_patience=60 if model_key in {"vit", "vit_1d"} else 30,
        final_max_epochs=2000 if model_key in {"vit", "vit_1d"} else (1000 if classification else 300),
        final_patience=100 if model_key in {"vit", "vit_1d"} else 50,
        n_trials_arch=200 if model_key in {"vit", "vit_1d"} else (100 if classification else 30),
        n_trials_hp=100 if model_key in {"vit", "vit_1d"} else (100 if classification else 30),
        classification=classification,
    )
