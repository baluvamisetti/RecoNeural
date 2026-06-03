# 🧠 RecoNeural — Neural Collaborative Filtering

Production-grade movie recommender on **MovieLens 1M** using **NCF (GMF + MLP)**.

## Architecture
NCF = Generalized Matrix Factorization + MLP tower → Fused output → Sigmoid score
Based on: *He et al. (2017) — "Neural Collaborative Filtering"*

## Project Structure
```
ncf_recommender/
├── model/ncf.py          # GMF + MLP fusion model
├── utils/dataset.py      # MovieLens loader + negative sampling
├── utils/metrics.py      # HR@K, NDCG@K evaluation
├── utils/inference.py    # RecommendationEngine (load + serve)
├── train.py              # Full training pipeline
├── app.py                # Streamlit web app
└── requirements.txt
```

## Quickstart
```bash
pip install -r requirements.txt
python train.py --version ml-1m --epochs 20 --emb_dim 64
streamlit run app.py
```

## Training Arguments
```bash
python train.py \
  --version ml-1m \    # ml-1m or ml-100k
  --emb_dim 64 \       # embedding dimension
  --epochs 20 \        # max epochs
  --batch_size 1024 \  # batch size
  --lr 1e-3 \          # learning rate
  --dropout 0.2 \      # dropout
  --num_neg 4 \        # negatives per positive
  --min_rating 3.5 \   # implicit feedback threshold
  --patience 5         # early stopping
```

## Expected Results (ml-1m)
| Metric | Score |
|---|---|
| HR@10 | ~0.68–0.72 |
| NDCG@10 | ~0.40–0.45 |

## Key Features
- NCF from scratch in PyTorch (He et al. 2017)
- Proper evaluation: HR@10, NDCG@10, leave-one-out protocol
- Negative sampling during training
- Item-item similarity via embedding cosine similarity
- Production inference engine with st.cache_resource
- Full Streamlit UI with TMDB posters + Plotly training curves

## App Tabs
1. **User Recommendations** — watch history + top-K NCF recommendations with posters
2. **Similar Movies** — embedding-space cosine similarity
3. **Training Curves** — loss + HR@10 + NDCG@10 per epoch

## Deploy
Push to GitHub → connect at share.streamlit.io → set TMDB_API_KEY in secrets → public URL.
