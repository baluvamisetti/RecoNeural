"""
utils/dataset.py — MovieLens data loader + NCF dataset with negative sampling
"""

import os, zipfile, requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def download_movielens(data_dir="data", version="ml-1m"):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    extract_path = os.path.join(data_dir, version)
    zip_path = os.path.join(data_dir, f"{version}.zip")

    if not os.path.exists(extract_path):
        print(f"Downloading {version} ...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        print("Download complete.")
    return extract_path


def load_movielens(data_dir="data", version="ml-1m"):
    path = download_movielens(data_dir, version)
    ratings = pd.read_csv(
        os.path.join(path, "ratings.dat"),
        sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        os.path.join(path, "movies.dat"),
        sep="::", engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def preprocess(ratings, min_rating=3.5):
    df = ratings[ratings["rating"] >= min_rating].copy()
    df["label"] = 1.0
    user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
    item2idx = {m: i for i, m in enumerate(df["movieId"].unique())}
    df["user"] = df["userId"].map(user2idx)
    df["item"] = df["movieId"].map(item2idx)
    return df[["user", "item", "label", "timestamp"]], user2idx, item2idx


def train_test_split_temporal(df):
    df = df.sort_values("timestamp")
    test = df.groupby("user").tail(1)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


class NCFDataset(Dataset):
    def __init__(self, df, num_items, num_negatives=4):
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_positives = df.groupby("user")["item"].apply(set).to_dict()
        self.users = df["user"].values
        self.items = df["item"].values

    def __len__(self):
        return len(self.users) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        sample_rank = idx % (1 + self.num_negatives)
        user = self.users[pos_idx]

        if sample_rank == 0:
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(self.items[pos_idx], dtype=torch.long),
                torch.tensor(1.0, dtype=torch.float),
            )
        else:
            positives = self.user_positives[user]
            neg = np.random.randint(0, self.num_items)
            while neg in positives:
                neg = np.random.randint(0, self.num_items)
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long),
                torch.tensor(0.0, dtype=torch.float),
            )
