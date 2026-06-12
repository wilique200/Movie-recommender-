# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Title-based Movie Recommender", layout="wide")

# -------------------------
# Helpers
# -------------------------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = re.sub(r"\|", " ", str(s))
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    return s.lower().strip()

@st.cache_data
def load_movies_from_zip_path(zip_path, csv_ext=".csv"):
    """Load all CSVs from a zip file into a single DataFrame."""
    dfs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        files = [f for f in z.namelist() if f.lower().endswith(csv_ext)]
        for f in files:
            try:
                df = pd.read_csv(z.open(f))
                df["source_file"] = os.path.basename(f)
                dfs.append(df)
            except Exception:
                continue
    if not dfs:
        return pd.DataFrame()
    movies = pd.concat(dfs, ignore_index=True)
    return movies

@st.cache_data
def build_tfidf_matrix(movies):
    df = movies.copy()
    # prefer columns movie_name or title
    if "movie_name" in df.columns:
        df = df.rename(columns={"movie_name": "title"})
    if "title" not in df.columns:
        df["title"] = df.index.astype(str)
    df["genres"] = df.get("genre", df.get("genres", "")).fillna("")
    df["description"] = df.get("description", df.get("overview", "")).fillna("")
    df["content"] = (df["genres"].astype(str) + " " + df["description"].astype(str)).apply(clean_text)
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1,2))
    mat = tfidf.fit_transform(df["content"])
    return df.reset_index(drop=True), tfidf, mat

def recommend_by_title(title_query, df, mat, tfidf, topn=8):
    # fuzzy-ish substring match first
    matches = df[df["title"].str.lower().str.contains(title_query.lower(), na=False)]
    if matches.empty:
        # exact match fallback
        matches = df[df["title"].str.lower() == title_query.lower()]
    if matches.empty:
        return None, f"No movie found matching '{title_query}'. Try a different title or a substring."
    idx = matches.index[0]
    vec = mat[idx]
    sims = cosine_similarity(vec, mat).flatten()
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:topn]
    recs = df.iloc[top_idx].copy()
    recs["score"] = sims[top_idx]

    feature_names = np.array(tfidf.get_feature_names_out())
    q_row = mat[idx].toarray().flatten()
    top_terms_idx = q_row.argsort()[::-1][:8]
    top_terms = [feature_names[i] for i in top_terms_idx if q_row[i] > 0][:6]

    explanations = []
    for ridx, row in recs.iterrows():
        g1 = set(str(df.loc[idx, "genres"]).split())
        g2 = set(str(row["genres"]).split())
        shared_genres = sorted(list(g1.intersection(g2)))
        cand_row = mat[ridx].toarray().flatten()
        shared_strength = np.minimum(q_row, cand_row)
        top_shared_idx = shared_strength.argsort()[::-1][:5]
        shared_terms = [feature_names[i] for i in top_shared_idx if shared_strength[i] > 0][:4]
        reason_parts = []
        if shared_genres:
            reason_parts.append("Shared genres: " + ", ".join(shared_genres))
        if shared_terms:
            reason_parts.append("Matching keywords: " + ", ".join(shared_terms))
        if not reason_parts:
            reason_parts.append("Similar overall content and themes")
        explanations.append(" • ".join(reason_parts))

    recs["reason"] = explanations
    return recs.reset_index(drop=True), None

# -------------------------
# UI
# -------------------------
st.title("🎯 Title-based Movie Recommender")

st.sidebar.header("Data source")
st.sidebar.markdown("Place your downloaded ZIP in `data/` named `imdb_movies.zip` or upload a ZIP here.")
uploaded = st.sidebar.file_uploader("Upload ZIP with CSVs", type=["zip"])
use_local_zip = False
local_zip_path = "data/imdb_movies.zip"
if os.path.exists(local_zip_path):
    st.sidebar.success(f"Found local ZIP at {local_zip_path}")
    use_local_zip = True

if uploaded is None and not use_local_zip:
    st.info("No dataset found. Upload a ZIP or add `data/imdb_movies.zip` to the repo.")
    st.stop()

# Load movies
if uploaded is not None:
    bytes_io = io.BytesIO(uploaded.read())
    movies = load_movies_from_zip_path(bytes_io)
else:
    movies = load_movies_from_zip_path(local_zip_path)

if movies.empty:
    st.error("No CSV files found inside the ZIP or CSVs could not be read.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Recommendation settings")
topn = st.sidebar.slider("Top N recommendations", 3, 20, 8)
method_info = st.sidebar.selectbox("Explainability detail", ["Short reasons", "Show top terms"], index=0)

# Build TF-IDF
df, tfidf, mat = build_tfidf_matrix(movies)

st.subheader("Search by movie title")
query = st.text_input("Type a movie title or substring", value="Toy Story")
if st.button("Recommend"):
    if not query.strip():
        st.warning("Please type a movie title.")
    else:
        recs, err = recommend_by_title(query, df, mat, tfidf, topn=topn)
        if err:
            st.info(err)
        else:
            # show base movie
            base_title = df.loc[df['title'].str.lower().str.contains(query.lower()).idxmax(), 'title'] if any(df["title"].str.lower().str.contains(query.lower())) else query
            st.markdown(f"### Recommendations based on: **{base_title}**")
            for i, row in recs.iterrows():
                st.markdown(f"**{i+1}. {row['title']}**  —  **score:** {row['score']:.3f}")
                st.caption(row["reason"])
            st.markdown("---")
            st.info("Explanations show shared genres and matching TF‑IDF keywords from genres+description.")

st.markdown("### Dataset preview")
st.write(f"Total movies loaded: **{len(df)}**")
st.dataframe(df[["title", "genres", "rating", "year"]].head(50))
