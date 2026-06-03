"""
RecoNeural — Neural Collaborative Filtering Movie Recommender
Streamlit frontend — poster images fetched server-side (no CORS issues)
"""

import os, sys, re, io
import streamlit as st
import requests
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="RecoNeural", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

TMDB_KEY = os.getenv("TMDB_API_KEY", "0236e660a30a82dfc93ee6e8da9a5605")

def _clean_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

def _extract_year(title):
    m = re.search(r"\((\d{4})\)", title)
    return m.group(1) if m else None

@st.cache_data(show_spinner=False)
def fetch_poster_bytes(title):
    """Fetch poster and return raw image bytes — avoids all CORS/browser issues."""
    clean = _clean_title(title)
    year  = _extract_year(title)

    poster_path = None
    for use_year in [True, False]:
        try:
            q = requests.utils.quote(clean)
            url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={q}&language=en-US"
            if use_year and year:
                url += f"&year={year}"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            for r in resp.json().get("results", []):
                if r.get("poster_path"):
                    poster_path = r["poster_path"]
                    break
            if poster_path:
                break
        except Exception:
            continue

    if not poster_path:
        return None

    # Download the actual image bytes
    try:
        img_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
        img_resp = requests.get(img_url, timeout=8)
        if img_resp.status_code == 200:
            return img_resp.content  # raw bytes
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=True)
def load_engine():
    from utils.inference import RecommendationEngine
    return RecommendationEngine(save_dir="saved")


def render_movie_card(col, title, genres, poster_bytes, badge=""):
    """Render movie card using st.image with bytes — works on all browsers."""
    with col:
        if poster_bytes:
            st.image(poster_bytes, use_container_width=True)
        else:
            st.markdown(
                '<div style="width:100%;aspect-ratio:2/3;background:linear-gradient(135deg,#1a1a35,#2d1b4e);'
                'border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:2.5rem;'
                'min-height:160px;">🎬</div>',
                unsafe_allow_html=True
            )
        short_title = title[:32] + "…" if len(title) > 32 else title
        short_genre = genres[:42] + "…" if len(genres) > 42 else genres
        st.markdown(
            f'<div style="font-size:0.82rem;font-weight:600;color:#e8e8f0;margin-top:6px;line-height:1.3">{short_title}</div>'
            f'<div style="font-size:0.72rem;color:#6b7280;margin-top:3px">{short_genre}</div>{badge}',
            unsafe_allow_html=True
        )


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #09090f; color: #e8e8f0; }
.block-container { padding: 2rem 3rem; }
.hero { background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0d1a0d 100%); border: 1px solid rgba(147,51,234,0.2); border-radius: 20px; padding: 3rem; text-align: center; margin-bottom: 2rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #34d399, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.hero-sub { color: #9ca3af; font-size: 1.1rem; margin-top: 0.5rem; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-card { flex: 1; background: linear-gradient(135deg, #111128, #1a1a35); border: 1px solid rgba(167,139,250,0.15); border-radius: 14px; padding: 1.25rem 1.5rem; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.score-badge { display: inline-block; background: linear-gradient(135deg, #7c3aed, #2563eb); color: white; border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; margin-top: 6px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #a78bfa; margin: 1.5rem 0 1rem; }
section[data-testid="stSidebar"] { background: #0d0d1a; border-right: 1px solid rgba(255,255,255,0.05); }
.stTabs [data-baseweb="tab-list"] { background: #111128; border-radius: 10px; padding: 4px; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7c3aed, #2563eb) !important; color: white !important; }
.stButton > button { background: linear-gradient(135deg, #7c3aed, #2563eb); color: white; border: none; border-radius: 10px; font-family: 'Syne', sans-serif; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="hero-title">🧠 RecoNeural</div>'
    '<div class="hero-sub">Neural Collaborative Filtering · GMF + MLP Fusion · MovieLens 1M</div></div>',
    unsafe_allow_html=True
)

try:
    engine = load_engine()
except Exception as e:
    st.error(f"⚠️ Model not found. Run `python train.py` first.\n\n`{e}`")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Model Config")
    cfg = engine.model_cfg
    st.markdown(
        f"- **Architecture**: GMF + MLP (NCF)\n"
        f"- **Embedding dim**: `{cfg['emb_dim']}`\n"
        f"- **MLP layers**: `{cfg['mlp_layers']}`\n"
        f"- **Users**: `{engine.num_users:,}`\n"
        f"- **Items**: `{engine.num_items:,}`"
    )
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics")
    for k, v in engine.metrics.items():
        st.metric(k, f"{v:.4f}")
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    top_k = st.slider("Top-K recommendations", 5, 20, 10)
    show_scores = st.toggle("Show confidence scores", True)
    exclude_seen = st.toggle("Exclude already-seen movies", True)

m = engine.metrics
st.markdown(f"""<div class="metric-row">
    <div class="metric-card"><div class="metric-value">{m.get("HR@10",0):.3f}</div><div class="metric-label">Hit Rate @ 10</div></div>
    <div class="metric-card"><div class="metric-value">{m.get("NDCG@10",0):.3f}</div><div class="metric-label">NDCG @ 10</div></div>
    <div class="metric-card"><div class="metric-value">{engine.num_users:,}</div><div class="metric-label">Total Users</div></div>
    <div class="metric-card"><div class="metric-value">{engine.num_items:,}</div><div class="metric-label">Total Movies</div></div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["👤 User Recommendations", "🎬 Similar Movies", "📈 Training Curves"])

with tab1:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        user_idx = st.selectbox("Select a User ID", options=engine.all_user_ids[:500], format_func=lambda x: f"User #{x}")
    with col_b:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_btn = st.button("🚀 Generate Recommendations", use_container_width=True)

    if run_btn:
        history_movies = engine.get_user_history(user_idx, n=5)
        if history_movies:
            st.markdown('<div class="section-title">📜 Watch History</div>', unsafe_allow_html=True)
            h_cols = st.columns(len(history_movies))
            for col, movie in zip(h_cols, history_movies):
                poster = fetch_poster_bytes(movie["title"])
                genres = movie["genres"].replace("|", " · ")
                render_movie_card(col, movie["title"], genres, poster)

        st.markdown('<div class="section-title">🎯 Top Recommendations</div>', unsafe_allow_html=True)
        with st.spinner("Neural model scoring all movies..."):
            recs = engine.recommend(user_idx, top_k=top_k, exclude_seen=exclude_seen)

        for row_start in range(0, min(top_k, len(recs)), 5):
            row_recs = recs[row_start:row_start + 5]
            cols = st.columns(5)
            for col, rec in zip(cols, row_recs):
                poster = fetch_poster_bytes(rec["title"])
                genres = rec["genres"].replace("|", " · ")
                badge = f'<div class="score-badge">⚡ {rec["score"]:.3f}</div>' if show_scores else ""
                render_movie_card(col, f'#{rec["rank"]} {rec["title"]}', genres, poster, badge)

with tab2:
    movie_query = st.selectbox("Search a movie to find similar ones", options=engine.all_movie_titles[:3000])
    sim_btn = st.button("🔍 Find Similar Movies")
    if sim_btn and movie_query:
        with st.spinner("Computing embedding similarities..."):
            similars = engine.get_similar_items(movie_query, top_k=top_k)
        if similars:
            st.markdown(f'<div class="section-title">🎬 Movies similar to "{movie_query}"</div>', unsafe_allow_html=True)
            for row_start in range(0, len(similars), 5):
                row = similars[row_start:row_start + 5]
                cols = st.columns(5)
                for col, movie in zip(cols, row):
                    poster = fetch_poster_bytes(movie["title"])
                    genres = movie["genres"].replace("|", " · ")
                    badge = f'<div class="score-badge">sim {movie["similarity"]:.3f}</div>' if show_scores else ""
                    render_movie_card(col, movie["title"], genres, poster, badge)

with tab3:
    hist = engine.history
    if hist and hist.get("train_loss"):
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"], name="Train Loss", line=dict(color="#ef4444", width=2)))
        fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="BCE Loss",
                          plot_bgcolor="#111128", paper_bgcolor="#09090f", font=dict(color="#e8e8f0"), height=350)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=epochs, y=hist["HR@10"], name="HR@10", line=dict(color="#34d399", width=2)))
        fig2.add_trace(go.Scatter(x=epochs, y=hist["NDCG@10"], name="NDCG@10", line=dict(color="#60a5fa", width=2)))
        fig2.update_layout(title="Ranking Metrics", xaxis_title="Epoch", yaxis_title="Score",
                           plot_bgcolor="#111128", paper_bgcolor="#09090f", font=dict(color="#e8e8f0"), height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Train the model to see curves here.")
