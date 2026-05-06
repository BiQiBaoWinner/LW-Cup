import os
import sys
import json
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (
    results_path,
    range_split,
    FACTORS,
    ob_cols_derive1,
    ob_cols_derive2,
    ob_cols_pro,
)
from factor_pool.pipeline import FactorPipeline


class DeepLOBNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.num_classes)
        
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64, device=x.device)
        c0 = torch.zeros(1, x.size(0), 64, device=x.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool_w(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class LOBWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y = torch.from_numpy(np.array(self.y[idx], dtype=np.int64))
        return x, y


def _make_windows(df: pd.DataFrame, feature_cols: List[str], label_col: str, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []

    for (_, _), g in df.groupby(["sym", "date"], sort=False):
        g = g.sort_values("timestamp")
        features = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g[label_col].to_numpy()

        if len(g) < window:
            continue

        for i in range(window, len(g) + 1):
            X_list.append(features[i - window:i, :])
            y_list.append(labels[i - 1])

    if not X_list:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    if y.min() == 1:
        y = y - 1

    return X, y


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        df = df.copy()
        df["Ndate"] = pd.to_datetime(df["date"].astype(int), unit="D", origin=pd.Timestamp("2020-01-01"))
        df["timestamp"] = pd.to_datetime(
            df["Ndate"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str),
            format="%Y-%m-%d %H:%M:%S",
        )
    return df


def _calc_factor_panel(tick_df: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    pip = FactorPipeline(tick_df, date_range=None)
    pip.Tick_Factor_Pool.registry = FACTORS
    max_workers = min(n_jobs, os.cpu_count() or 1)
    return pip.load_factor_exposure(n_jobs=max_workers)


class DLOB:
    def __init__(
        self,
        task_label: str = "label_20",
        feature_cols: Optional[List[str]] = None,
        use_pool: bool = True,
        extra_cols: Optional[List[str]] = None,
        window: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4,
        seed: int = 42,
        device: Optional[str] = None,
        save_dir: Optional[str] = None,
        n_jobs: int = 16,
    ):
        self.task_label = task_label
        self.feature_cols = feature_cols
        self.use_pool = use_pool
        self.extra_cols = extra_cols or []
        self.window = window
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_jobs = n_jobs
        self.save_dir = save_dir or os.path.expanduser(f"{results_path}/dlob_models")
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = None

    def DataPreparing(self, split: str = "train"):
        if not self.use_pool:
            if not self.feature_cols and not self.extra_cols:
                raise ValueError("feature_cols or extra_cols is required when use_pool is False.")

        dates = [str(i) for i in range(int(range_split[split][0]), int(range_split[split][1]) + 1)]
        df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
        df = df[df["date"].astype(str).isin(dates)].copy()

        df = _ensure_timestamp(df)

        if self.use_pool:
            factor_panel = _calc_factor_panel(df, self.n_jobs)
            if factor_panel.empty:
                raise ValueError("Factor panel is empty. Check FACTORS and input columns.")

            raw_indexed = df.set_index(["sym", "timestamp"]).sort_index()
            raw_cols = self.extra_cols
            if not raw_cols:
                raw_cols = []
            raw_features = raw_indexed.loc[:, raw_cols] if raw_cols else pd.DataFrame(index=raw_indexed.index)
            task_label = raw_indexed.loc[:, self.task_label] if self.task_label in raw_indexed.columns else pd.Series(index=raw_indexed.index)
            if task_label.isnull().all():
                raise ValueError(f"Task label column '{self.task_label}' is missing or empty in the input data.")
            combined = factor_panel.join(raw_features, how="inner")
            combined = combined.join(task_label, how="inner", rsuffix="")
            # 检查label列是否存在且不含NaN
            if f"{self.task_label}" not in combined.columns or combined[f"{self.task_label}"].isnull().all():
                raise ValueError(f"After joining, task label column '{self.task_label}' is missing or empty. Check the input data and task_label.")
            combined["date"] = raw_indexed.reindex(combined.index)["date"].values
            combined["sym"] = combined.index.get_level_values("sym")
            combined["timestamp"] = combined.index.get_level_values("timestamp")

            feature_cols = list(factor_panel.columns) + raw_cols
            df_for_windows = combined.reset_index(drop=True)
        else:
            feature_cols = self.feature_cols or self.extra_cols
            df_for_windows = df

        df_for_windows = df_for_windows.dropna(subset=feature_cols + [self.task_label])
        
        X, y = _make_windows(df_for_windows, feature_cols, self.task_label, self.window)
        dataset = LOBWindowDataset(X, y)
        return dataset

    def fit_model(self, epochs: int = 10, early_stopping: Optional[int] = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_dataset = self.DataPreparing(split="train")
        valid_dataset = self.DataPreparing(split="valid")

        if len(train_dataset) == 0:
            raise ValueError("No training samples after windowing. Check feature_cols and window.")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        num_classes = int(np.max(train_dataset.y) + 1)
        self.model = DeepLOBNet(num_classes=num_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        log_dir = os.path.join(self.save_dir, "tensorboard")
        writer = SummaryWriter(log_dir=log_dir)

        best_state = None
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_count = 0

            train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
            for x, y in train_bar:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                train_count += x.size(0)
                train_bar.set_postfix(loss=loss.item())

            self.model.eval()
            val_loss = 0.0
            val_count = 0
            val_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [valid]", leave=False)
            with torch.no_grad():
                for x, y in val_bar:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = self.model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * x.size(0)
                    val_count += x.size(0)
                    val_bar.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / train_count if train_count else float("nan")
            avg_val_loss = val_loss / val_count if val_count else float("nan")

            if val_count > 0 and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = self.model.state_dict()
                epochs_no_improve = 0
            elif val_count > 0:
                epochs_no_improve += 1

            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.6f}, val_loss: {avg_val_loss:.6f}")

            writer.add_scalar("loss/train", avg_train_loss, epoch + 1)
            writer.add_scalar("loss/valid", avg_val_loss, epoch + 1)

            if early_stopping is not None and val_count > 0 and epochs_no_improve >= early_stopping:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        writer.close()

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        self.model.eval()
        x = torch.from_numpy(X).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def save_model(self, filename: str = "dlob_model.pt"):
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        path = os.path.join(self.save_dir, filename)
        torch.save({"model_state": self.model.state_dict()}, path)
        return path

    def load_model(self, path: str, num_classes: int):
        self.model = DeepLOBNet(num_classes=num_classes).to(self.device)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        return self.model


if __name__ == "__main__":
    # Example usage
    feature_cols = [
        "bid1", "ask1", "bsize1", "asize1",
        "bid2", "ask2", "bsize2", "asize2",
        "bid3", "ask3", "bsize3", "asize3",
        "bid4", "ask4", "bsize4", "asize4",
        "bid5", "ask5", "bsize5", "asize5",
        "bid6", "ask6", "bsize6", "asize6",
        "bid7", "ask7", "bsize7", "asize7",
        "bid8", "ask8", "bsize8", "asize8",
        "bid9", "ask9", "bsize9", "asize9",
        "bid10", "ask10", "bsize10", "asize10",
    ]

    model = DLOB(task_label="label_20", feature_cols=feature_cols)
    model.fit_model(epochs=3)
    model.save_model()
