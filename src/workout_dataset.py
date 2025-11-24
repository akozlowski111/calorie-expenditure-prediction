import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Iterable, Optional, Tuple


class WorkoutDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        *,
        indices: Optional[Iterable[int]] = None,
        normalize: bool = True,
        normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> None:
        data: pd.DataFrame = pd.read_csv(csv_file)
        data["Sex"] = data["Sex"].map({"male": 1, "female": 0}).astype(float)

        features_df = data.drop(columns=["Calories", "id"])
        target_series = data["Calories"]

        if indices is not None:
            if isinstance(indices, torch.Tensor):
                idx_list = indices.tolist()
            else:
                idx_list = list(indices)
            features_df = features_df.iloc[idx_list]
            target_series = target_series.iloc[idx_list]

        x = torch.tensor(features_df.values, dtype=torch.float32)
        y = torch.tensor(target_series.values, dtype=torch.float32).view(-1, 1)

        self.x_min: Optional[torch.Tensor] = None
        self.x_max: Optional[torch.Tensor] = None

        if normalize:
            if normalization_stats is not None:
                x_min, x_max = normalization_stats
            else:
                x_min, _ = x.min(dim=0, keepdim=True)
                x_max, _ = x.max(dim=0, keepdim=True)
            self.x_min = x_min
            self.x_max = x_max
            x = (x - x_min) / (x_max - x_min + 1e-8)

        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

    @property
    def normalization_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.x_min is None or self.x_max is None:
            return None
        return self.x_min, self.x_max

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
