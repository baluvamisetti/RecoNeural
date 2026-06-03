"""
model/ncf.py — Neural Collaborative Filtering (NeuMF)
GMF + MLP fusion. He et al. 2017: https://arxiv.org/abs/1708.05031
"""

import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64,
                 mlp_layers=None, dropout=0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        # GMF embeddings
        self.gmf_user = nn.Embedding(num_users, emb_dim)
        self.gmf_item = nn.Embedding(num_items, emb_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(num_users, emb_dim)
        self.mlp_item = nn.Embedding(num_items, emb_dim)

        # MLP tower
        layers = []
        in_dim = emb_dim * 2
        for out_dim in mlp_layers:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp_tower = nn.Sequential(*layers)

        # Fusion layer
        self.fusion = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item,
                    self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp_tower:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, user, item):
        # GMF branch
        gmf_out = self.gmf_user(user) * self.gmf_item(item)

        # MLP branch
        mlp_in = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=-1)
        mlp_out = self.mlp_tower(mlp_in)

        # Fuse & predict
        out = self.fusion(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1)
        return torch.sigmoid(out)
