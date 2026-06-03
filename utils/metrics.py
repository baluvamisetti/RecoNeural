"""
utils/metrics.py — HR@K and NDCG@K evaluation (leave-one-out, 99 negatives)
"""

import numpy as np
import torch
import pandas as pd


def evaluate(model, test_df, train_df, num_items, k=10,
             num_neg=99, device=torch.device("cpu")):
    model.eval()
    seen = (
        pd.concat([train_df, test_df])
        .groupby("user")["item"].apply(set).to_dict()
    )
    hits, ndcgs = [], []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            user = int(row["user"])
            pos_item = int(row["item"])
            user_seen = seen.get(user, set())

            negs = []
            while len(negs) < num_neg:
                cand = np.random.randint(0, num_items)
                if cand not in user_seen and cand not in negs:
                    negs.append(cand)

            items = [pos_item] + negs
            u_t = torch.tensor([user] * len(items), dtype=torch.long).to(device)
            i_t = torch.tensor(items, dtype=torch.long).to(device)
            scores = model(u_t, i_t).cpu().numpy()

            ranked = np.argsort(-scores)[:k]
            if 0 in ranked:
                rank = int(np.where(ranked == 0)[0][0]) + 1
                hits.append(1)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                hits.append(0)
                ndcgs.append(0.0)

    return {f"HR@{k}": float(np.mean(hits)), f"NDCG@{k}": float(np.mean(ndcgs))}
