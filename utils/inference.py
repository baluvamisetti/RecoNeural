"""
utils/inference.py — RecommendationEngine for Streamlit app
"""

import os, pickle
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ncf import NCF


class RecommendationEngine:
    def __init__(self, save_dir="saved"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(save_dir, "artifacts.pkl"), "rb") as f:
            art = pickle.load(f)

        self.user2idx   = art["user2idx"]
        self.item2idx   = art["item2idx"]
        self.idx2item   = art["idx2item"]
        self.num_users  = art["num_users"]
        self.num_items  = art["num_items"]
        self.movies_df  = art["movies"]
        self.train_df   = art["train_df"]
        self.history    = art["history"]
        self.metrics    = art["final_metrics"]
        self.model_cfg  = art["model_config"]
        self.movie_meta = self.movies_df.set_index("movieId").to_dict(orient="index")

        self.model = NCF(
            num_users=self.num_users,
            num_items=self.num_items,
            emb_dim=self.model_cfg["emb_dim"],
            mlp_layers=self.model_cfg["mlp_layers"],
            dropout=self.model_cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(save_dir, "ncf_best.pt"),
                       map_location=self.device)
        )
        self.model.eval()

    def get_user_history(self, user_idx, n=5):
        rows = self.train_df[self.train_df["user"] == user_idx].tail(n)
        result = []
        for _, row in rows.iterrows():
            mid = self.idx2item.get(int(row["item"]))
            meta = self.movie_meta.get(mid, {})
            result.append({
                "movieId": mid,
                "title": meta.get("title", "Unknown"),
                "genres": meta.get("genres", ""),
            })
        return result

    def recommend(self, user_idx, top_k=10, exclude_seen=True):
        seen = set()
        if exclude_seen:
            seen = set(self.train_df[self.train_df["user"] == user_idx]["item"].tolist())

        all_items = [i for i in range(self.num_items) if i not in seen]
        u_t = torch.tensor([user_idx] * len(all_items), dtype=torch.long).to(self.device)
        i_t = torch.tensor(all_items, dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model(u_t, i_t).cpu().numpy()

        top_idx = np.argsort(-scores)[:top_k]
        recs = []
        for rank, idx in enumerate(top_idx):
            item_idx = all_items[idx]
            mid = self.idx2item.get(item_idx)
            meta = self.movie_meta.get(mid, {})
            recs.append({
                "rank": rank + 1,
                "movieId": mid,
                "title": meta.get("title", f"Movie {mid}"),
                "genres": meta.get("genres", "Unknown"),
                "score": float(scores[idx]),
            })
        return recs

    def get_similar_items(self, movie_title, top_k=10):
        # Exact match first, then partial
        exact = self.movies_df[self.movies_df["title"] == movie_title]
        if not exact.empty:
            matches = exact
        else:
            import re
            clean = re.sub(r"\s*\(\d{4}\)\s*$", "", movie_title).strip()
            matches = self.movies_df[
                self.movies_df["title"].str.contains(re.escape(clean), case=False, na=False)
            ]
        if matches.empty:
            return []

        # Find one that exists in item2idx
        mid = None
        for _, row in matches.iterrows():
            if row["movieId"] in self.item2idx:
                mid = row["movieId"]
                break
        if mid is None:
            return []

        query_idx = self.item2idx[mid]
        with torch.no_grad():
            all_ids = torch.arange(self.num_items, dtype=torch.long).to(self.device)
            embs = self.model.gmf_item(all_ids).cpu().numpy()

        q = embs[query_idx]
        q_norm = np.linalg.norm(q) + 1e-9
        norms = np.linalg.norm(embs, axis=1) + 1e-9
        sims = (embs @ q) / (norms * q_norm)
        top_idx = np.argsort(-sims)[1:top_k + 1]

        results = []
        for idx in top_idx:
            m = self.idx2item.get(int(idx))
            meta = self.movie_meta.get(m, {})
            results.append({
                "movieId": m,
                "title": meta.get("title", f"Movie {m}"),
                "genres": meta.get("genres", "Unknown"),
                "similarity": float(sims[idx]),
            })
        return results

    @property
    def all_user_ids(self):
        return list(range(self.num_users))

    @property
    def all_movie_titles(self):
        return self.movies_df["title"].dropna().tolist()
