import logging
from pathlib import Path
from typing import Tuple

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .train import build_model, set_seed


logger = logging.getLogger(__name__)


def load_test_features(csv_file: str, stats: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    data = pd.read_csv(csv_file)
    if "id" not in data.columns:
        raise ValueError("Plik testowy musi zawierać kolumnę 'id'.")

    data["Sex"] = data["Sex"].map({"male": 1, "female": 0}).astype(float)
    ids = torch.tensor(data["id"].values, dtype=torch.long)
    features_df = data.drop(columns=["id"])
    features = torch.tensor(features_df.values, dtype=torch.float32)

    x_min, x_max = stats
    features = (features - x_min) / (x_max - x_min + 1e-8)
    return ids, features


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.inference.model_path or not cfg.inference.stats_path:
        raise ValueError(
            "Podaj ścieżki inference.model_path oraz inference.stats_path "
            "(np. python -m src.predict inference.model_path=outputs/.../best_model.pth inference.stats_path=outputs/.../normalization_stats.pt)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.training.seed)

    artifact = torch.load(cfg.inference.stats_path, map_location="cpu")
    architecture = artifact["architecture"]
    dropout = float(artifact.get("dropout", 0.0))
    stats = (artifact["x_min"], artifact["x_max"])

    model = build_model(architecture, dropout=dropout).to(device)
    state_dict = torch.load(cfg.inference.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    ids, features = load_test_features(cfg.data.test, stats)
    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for (xb,) in dataloader:
            xb = xb.to(device)
            preds = model(xb).cpu()
            predictions.append(preds)

    predictions_tensor = torch.cat(predictions, dim=0).squeeze(1)
    submission = pd.DataFrame(
        {
            "id": ids.numpy(),
            "Calories": predictions_tensor.clamp(min=0).numpy(),
        }
    )

    output_path = Path(cfg.inference.submission_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    logger.info("Zapisano plik submission do %s", output_path)


if __name__ == "__main__":
    main()

