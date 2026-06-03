"""
Training pipeline for NCF on MovieLens.

Usage:
    python train.py --version ml-1m --epochs 20 --emb_dim 64
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.ncf import NCF
from utils.dataset import load_movielens, preprocess, train_test_split_temporal, NCFDataset
from utils.metrics import evaluate


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MovieLens...")
    ratings, movies = load_movielens(args.data_dir, args.version)
    df, user2idx, item2idx = preprocess(ratings, min_rating=args.min_rating)

    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Users: {num_users:,} | Items: {num_items:,} | Interactions: {len(df):,}")

    train_df, test_df = train_test_split_temporal(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    train_dataset = NCFDataset(train_df, num_items, num_negatives=args.num_neg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = NCF(
        num_users=num_users,
        num_items=num_items,
        emb_dim=args.emb_dim,
        mlp_layers=[256, 128, 64],
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = nn.BCELoss()

    best_hr = 0.0
    no_improve = 0
    history = {"train_loss": [], "HR@10": [], "NDCG@10": []}

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        eval_sample = test_df.sample(min(500, len(test_df)), random_state=42)
        metrics = evaluate(model, eval_sample, train_df, num_items, k=10, device=device)

        hr = metrics["HR@10"]
        ndcg = metrics["NDCG@10"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["HR@10"].append(hr)
        history["NDCG@10"].append(ndcg)

        print(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
            f"HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(hr)

        if hr > best_hr:
            best_hr = hr
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "ncf_best.pt"))
            print(f"  ✓ Best model saved (HR@10={hr:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nRunning full evaluation on best model...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "ncf_best.pt")))
    full_metrics = evaluate(model, test_df, train_df, num_items, k=10, device=device)
    print(f"Final | HR@10: {full_metrics['HR@10']:.4f} | NDCG@10: {full_metrics['NDCG@10']:.4f}")

    idx2item = {v: k for k, v in item2idx.items()}
    idx2user = {v: k for k, v in user2idx.items()}

    artifacts = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "idx2user": idx2user,
        "num_users": num_users,
        "num_items": num_items,
        "movies": movies,
        "train_df": train_df,
        "history": history,
        "final_metrics": full_metrics,
        "model_config": {
            "emb_dim": args.emb_dim,
            "mlp_layers": [256, 128, 64],
            "dropout": args.dropout,
        },
    }

    with open(os.path.join(args.save_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\nAll artifacts saved to '{args.save_dir}/'")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",     default="ml-1m")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--save_dir",   default="saved")
    parser.add_argument("--emb_dim",    type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--num_neg",    type=int,   default=4)
    parser.add_argument("--min_rating", type=float, default=3.5)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()
    train(args)
