content = open('utils/inference.py').read()

old = '''    def get_similar_items(self, movie_title, top_k=10):
        matches = self.movies_df[
            self.movies_df["title"].str.contains(movie_title, case=False, na=False)
        ]
        if matches.empty:
            return []
        mid = matches.iloc[0]["movieId"]
        if mid not in self.item2idx:
            return []

        query_idx = self.item2idx[mid]
        with torch.no_grad():
            all_ids = torch.arange(self.num_items, dtype=torch.long).to(self.device)
            embs = self.model.gmf_item(all_ids).cpu().numpy()

        q = embs[query_idx]
        norms = np.linalg.norm(embs, axis=1) + 1e-9
        sims = (embs @ q) / (norms * (np.linalg.norm(q) + 1e-9))
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
        return results'''

new = '''    def get_similar_items(self, movie_title, top_k=10):
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
        return results'''

content = content.replace(old, new)
open('utils/inference.py', 'w').write(content)
print('Fixed!')
