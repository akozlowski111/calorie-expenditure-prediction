import os
import logging
import random
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .workout_dataset import WorkoutDataset
from .rmsle_loss import RMSLELoss


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    architecture: List[int]
    learning_rate: float
    batch_size: int
    momentum: float
    dropout: float
    val_rmsle: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(architecture: Sequence[int], dropout: float = 0.0) -> nn.Module:
    layers: List[nn.Module] = []
    for idx in range(len(architecture) - 1):
        in_features = architecture[idx]
        out_features = architecture[idx + 1]
        layers.append(nn.Linear(in_features, out_features))
        if idx < len(architecture) - 2:
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def create_dataloader(dataset: WorkoutDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    criterion = RMSLELoss().to(device)
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            losses.append(loss.item())
    return float(sum(losses) / len(losses))


def train_single_run(
    train_dataset: WorkoutDataset,
    val_dataset: WorkoutDataset,
    *,
    architecture: Sequence[int],
    learning_rate: float,
    batch_size: int,
    momentum: float,
    dropout: float,
    epochs: int,
    device: torch.device,
    seed: int
) -> float:
    set_seed(seed)
    model = build_model(architecture, dropout=dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = RMSLELoss().to(device)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    val_rmsle = evaluate(model, val_loader, device)
    return val_rmsle


def run_primary_search(
    train_dataset: WorkoutDataset,
    val_dataset: WorkoutDataset,
    cfg: DictConfig,
    device: torch.device
) -> List[SearchResult]:
    architectures = [list(arch) for arch in cfg.search_space.architectures]
    lrs = cfg.search_space.learning_rates
    batches = cfg.search_space.batch_sizes
    momentums = cfg.search_space.momentums
    epochs = cfg.training.tuning_epochs
    seed = cfg.training.seed

    results: List[SearchResult] = []
    total = len(architectures) * len(lrs) * len(batches) * len(momentums)
    logger.info("Uruchamiam grid search (%d kombinacji)...", total)

    for idx, (arch, lr, batch_size, momentum) in enumerate(
        product(architectures, lrs, batches, momentums), start=1
    ):
        val_rmsle = train_single_run(
            train_dataset,
            val_dataset,
            architecture=arch,
            learning_rate=lr,
            batch_size=batch_size,
            momentum=momentum,
            dropout=0.0,
            epochs=epochs,
            device=device,
            seed=seed + idx,
        )
        result = SearchResult(
            architecture=list(arch),
            learning_rate=float(lr),
            batch_size=int(batch_size),
            momentum=float(momentum),
            dropout=0.0,
            val_rmsle=val_rmsle,
        )
        results.append(result)
        logger.info(
            "[%d/%d] arch=%s lr=%.4g batch=%d mom=%.2f -> val_rmsle=%.4f",
            idx,
            total,
            "->".join(map(str, arch)),
            lr,
            batch_size,
            momentum,
            val_rmsle,
        )

    results.sort(key=lambda r: r.val_rmsle)
    logger.info("Najlepszy wynik bez dropoutu: %.4f", results[0].val_rmsle)
    return results


def run_dropout_search(
    train_dataset: WorkoutDataset,
    val_dataset: WorkoutDataset,
    base_result: SearchResult,
    cfg: DictConfig,
    device: torch.device
) -> List[SearchResult]:
    dropout_rates = cfg.search_space.dropout_rates
    epochs = cfg.training.tuning_epochs
    seed = cfg.training.seed * 10

    logger.info(
        "Testuję dropout dla najlepszej konfiguracji (arch=%s, lr=%.4g, batch=%d, mom=%.2f)",
        "->".join(map(str, base_result.architecture)),
        base_result.learning_rate,
        base_result.batch_size,
        base_result.momentum,
    )
    results: List[SearchResult] = []
    for idx, dropout in enumerate(dropout_rates, start=1):
        val_rmsle = train_single_run(
            train_dataset,
            val_dataset,
            architecture=base_result.architecture,
            learning_rate=base_result.learning_rate,
            batch_size=base_result.batch_size,
            momentum=base_result.momentum,
            dropout=dropout,
            epochs=epochs,
            device=device,
            seed=seed + idx,
        )
        results.append(
            SearchResult(
                architecture=base_result.architecture,
                learning_rate=base_result.learning_rate,
                batch_size=base_result.batch_size,
                momentum=base_result.momentum,
                dropout=dropout,
                val_rmsle=val_rmsle,
            )
        )
        logger.info("Dropout %.2f -> val_rmsle=%.4f", dropout, val_rmsle)

    results.sort(key=lambda r: r.val_rmsle)
    logger.info("Najlepszy wynik z dropout: %.4f (dropout=%.2f)", results[0].val_rmsle, results[0].dropout)
    return results


def train_with_early_stopping(
    train_dataset: WorkoutDataset,
    val_dataset: WorkoutDataset,
    result: SearchResult,
    cfg: DictConfig,
    device: torch.device
) -> Tuple[dict, float, int, List[float]]:
    max_epochs = cfg.training.final_epochs
    patience = cfg.training.patience
    seed = cfg.training.seed * 100

    set_seed(seed)
    model = build_model(result.architecture, dropout=result.dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=result.learning_rate, momentum=result.momentum)
    criterion = RMSLELoss().to(device)

    train_loader = create_dataloader(train_dataset, batch_size=result.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=min(result.batch_size, len(val_dataset)), shuffle=False)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history: List[float] = []
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        val_rmsle = evaluate(model, val_loader, device)
        history.append(val_rmsle)

        if val_rmsle + 1e-5 < best_val:
            best_val = val_rmsle
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping po %d epokach bez poprawy (epoka %d, najlepsze %.4f)",
                patience,
                epoch,
                best_val,
            )
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val, best_epoch, history


def create_indices(length: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    val_count = max(1, int(length * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator)
    val_indices = permutation[:val_count].tolist()
    train_indices = permutation[val_count:].tolist()
    return train_indices, val_indices


def prepare_datasets(cfg: DictConfig) -> Tuple[WorkoutDataset, WorkoutDataset, Tuple[torch.Tensor, torch.Tensor]]:
    csv_path = cfg.data.train
    length = len(pd.read_csv(csv_path, usecols=["id"]))
    train_idx, val_idx = create_indices(length, cfg.training.val_fraction, cfg.training.seed)

    train_dataset = WorkoutDataset(csv_path, indices=train_idx, normalize=True)
    stats = train_dataset.normalization_stats
    if stats is None:
        raise RuntimeError("Nie udało się obliczyć statystyk normalizacji.")
    val_dataset = WorkoutDataset(csv_path, indices=val_idx, normalize=True, normalization_stats=stats)
    return train_dataset, val_dataset, stats


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    seed = cfg.training.seed
    set_seed(seed)
    logger.info("Ustawiono seed: %d", seed)
    logger.info("Rozpoczynam trening z konfiguracją: %s", cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Wykryte urządzenie: %s", device)

    train_dataset, val_dataset, stats = prepare_datasets(cfg)
    logger.info("Próbki treningowe: %d | walidacyjne: %d", len(train_dataset), len(val_dataset))

    primary_results = run_primary_search(train_dataset, val_dataset, cfg, device)
    best_primary = primary_results[0]

    dropout_results = run_dropout_search(train_dataset, val_dataset, best_primary, cfg, device)
    best_overall = dropout_results[0]

    best_state, best_val, best_epoch, history = train_with_early_stopping(
        train_dataset, val_dataset, best_overall, cfg, device
    )

    run_dir = HydraConfig.get().run.dir
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "best_model.pth")
    torch.save(best_state, model_path)

    stats_path = os.path.join(run_dir, "normalization_stats.pt")
    torch.save(
        {
            "x_min": stats[0].cpu(),
            "x_max": stats[1].cpu(),
            "architecture": best_overall.architecture,
            "dropout": best_overall.dropout,
        },
        stats_path,
    )

    summary_path = os.path.join(run_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Najlepsze hiperparametry bez dropoutu:\n")
        handle.write(f"{best_primary}\n\n")
        handle.write("Najlepsze hiperparametry z dropoutem:\n")
        handle.write(f"{best_overall}\n\n")
        handle.write(f"Early stopping – epoka {best_epoch}, val_rmsle={best_val:.4f}\n")

    logger.info("Zapisano najlepszy model do %s", model_path)
    logger.info("Zapisano statystyki normalizacji do %s", stats_path)
    logger.info("Ostateczny wynik walidacyjny: %.4f (epoka %d)", best_val, best_epoch)
    logger.info("Historia walidacyjna: %s", history)


if __name__ == "__main__":
    main()
