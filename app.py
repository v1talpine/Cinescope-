
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="CineScope | Movie Analytics", page_icon="🎬", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "movies.csv"

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#111827,#312e81); color: white; margin-bottom: 1rem;}
    .hero p {color: #d1d5db; margin-bottom: 0;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 14px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

def demo_data(n=900, seed=42):
    """Create a clearly labelled synthetic dataset so the app works offline."""
    rng = np.random.default_rng(seed)
    genres = ["Drama", "Comedy", "Action", "Thriller", "Sci-Fi", "Animation", "Romance", "Horror", "Adventure", "Documentary"]
    years = rng.integers(1980, 2026, n)
    genre = rng.choice(genres, n)
    budget = np.exp(rng.uniform(np.log(0.5), np.log(250), n))  # millions USD
    runtime = np.clip(rng.normal(112, 22, n), 65, 210).round()
    votes = np.exp(rng.uniform(np.log(500), np.log(1_500_000), n)).astype(int)
    genre_effect = pd.Series(genre).map({
        "Drama": .35, "Comedy": .05, "Action": .1, "Thriller": .15, "Sci-Fi": .25,
        "Animation": .4, "Romance": .0, "Horror": -.15, "Adventure": .2, "Documentary": .3
    }).to_numpy()
    rating = 6.2 + genre_effect + (runtime - 110) * .004 + np.log10(votes) * .15 + rng.normal(0, .85, n)
    rating = np.clip(rating, 1.5, 9.5).round(1)
    revenue = budget * rng.uniform(.25, 3.2, n) * np.exp((rating - 6.2) * .18)
    titles = [f"Sample Film {i+1:04d}" for i in range(n)]
    return pd.DataFrame({
        "title": titles, "year": years, "genre": genre, "runtime_min": runtime.astype(int),
        "budget_m_usd": budget.round(2), "revenue_m_usd": revenue.round(2),
        "rating": rating, "vote_count": votes
    })

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "name": "title", "movie_title": "title", "release_year": "year",
        "genres": "genre", "runtime": "runtime_min", "duration": "runtime_min",
        "imdb_rating": "rating", "average_rating": "rating", "imdb_score": "rating",
        "budget": "budget_m_usd", "revenue": "revenue_m_usd", "votes": "vote_count",
        "num_votes": "vote_count"
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    required = ["title", "year", "genre", "runtime_min", "budget_m_usd", "revenue_m_usd", "rating", "vote_count"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df = df[required]
    for col in ["year", "runtime_min", "budget_m_usd", "revenue_m_usd", "rating", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["title"] = df["title"].fillna("Untitled").astype(str)
    df["genre"] = df["genre"].fillna("Unknown").astype(str).str.split(r"[,|]").str[0].str.strip()
    df = df.dropna(subset=["rating", "year", "runtime_min", "genre"])
    df = df[(df["rating"] >= 0) & (df["rating"] <= 10) & (df["runtime_min"] > 0)]
    df["year"] = df["year"].astype(int)
    df["runtime_min"] = df["runtime_min"].astype(int)
    return df.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

@st.cache_data
def load_demo():
    return demo_data()

st.markdown("""
<div class="hero">
  <h1 style="margin:0">🎬 CineScope</h1>
  <p>Explore what movie data can — and cannot — tell us about ratings, budgets and success.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Project controls")
    st.caption("Choose a dataset to explore.")
    upload = st.file_uploader("Upload a movie CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("**Expected columns**")
    st.code("title, year, genre, runtime_min,\nbudget_m_usd, revenue_m_usd,\nrating, vote_count", language="text")
    st.caption("Common alternatives like IMDb score, genres, runtime and votes are mapped automatically where possible.")
    st.markdown("---")
    st.caption("Demo mode uses synthetic data. It is for testing the app, not evidence about real films.")

if upload is not None:
    try:
        raw = pd.read_csv(upload)
        df = normalize_columns(raw)
        source_label = "Uploaded dataset"
        if len(df) < 10:
            st.warning("Fewer than 10 usable rows were found. Charts and modelling may be unreliable.")
    except Exception as exc:
        st.error(f"Couldn't read this CSV: {exc}")
        df = load_demo()
        source_label = "Synthetic demo dataset"
else:
    df = load_demo()
    source_label = "Synthetic demo dataset"

if df.empty:
    st.error("No usable movie rows found. Check your CSV columns and values.")
    st.stop()

st.caption(f"**Data source:** {source_label} · **Usable records:** {len(df):,}")
if source_label == "Synthetic demo dataset":
    st.info("You're viewing generated sample data. Upload a real dataset before making claims about the movie industry.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔎 Explore", "🧠 Rating model", "💡 Findings", "📘 Methodology"])

with tab1:
    st.subheader("The big picture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(df):,}")
    c2.metric("Average rating", f"{df['rating'].mean():.2f}/10")
    c3.metric("Median runtime", f"{df['runtime_min'].median():.0f} min")
    c4.metric("Genres", f"{df['genre'].nunique():,}")
    left, right = st.columns(2)
    with left:
        genre_counts = df["genre"].value_counts().head(12).rename_axis("Genre").reset_index(name="Movies")
        st.plotly_chart(px.bar(genre_counts, x="Genre", y="Movies", title="Most represented genres", color="Movies"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(df, x="rating", nbins=20, title="Distribution of movie ratings", labels={"rating":"Rating (out of 10)"}), use_container_width=True)
    yearly = df.groupby("year", as_index=False).agg(movie_count=("title","count"), average_rating=("rating","mean"))
    yearly = yearly[yearly["movie_count"] >= 1]
    st.plotly_chart(px.line(yearly, x="year", y="average_rating", markers=True, title="Average rating by release year", labels={"year":"Release year","average_rating":"Average rating"}), use_container_width=True)

with tab2:
    st.subheader("Explore the dataset")
    g1, g2, g3 = st.columns(3)
    genres = sorted(df["genre"].dropna().unique().tolist())
    selected_genres = g1.multiselect("Genres", genres, default=genres[:min(5, len(genres))])
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    if min_year < max_year:
        year_range = g2.slider("Release year", min_year, max_year, (min_year, max_year))
    else:
        year_range = (min_year, max_year)
    min_rating = g3.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
    filtered = df[df["genre"].isin(selected_genres) & df["year"].between(*year_range) & (df["rating"] >= min_rating)]
    st.write(f"**{len(filtered):,} movies match your filters.**")
    if not filtered.empty:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.scatter(filtered, x="runtime_min", y="rating", color="genre", hover_name="title", title="Runtime vs rating", labels={"runtime_min":"Runtime (minutes)","rating":"Rating"}), use_container_width=True)
        with right:
            budget_df = filtered.dropna(subset=["budget_m_usd"])
            if not budget_df.empty:
                st.plotly_chart(px.scatter(budget_df, x="budget_m_usd", y="rating", color="genre", hover_name="title", log_x=True, title="Budget vs rating", labels={"budget_m_usd":"Budget (USD millions)","rating":"Rating"}), use_container_width=True)
            else:
                st.info("No usable budget values in the selected rows.")
        st.dataframe(filtered.sort_values("rating", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download filtered data", filtered.to_csv(index=False).encode("utf-8"), "cinescope_filtered_movies.csv", "text/csv")
    else:
        st.warning("No movies match these filters. Try broadening the selection.")

with tab3:
    st.subheader("Can we estimate a movie's rating?")
    st.write("This baseline model uses available movie metadata to estimate ratings. It is a learning demonstration, not a reliable rating oracle.")
    feature_candidates = ["year", "runtime_min", "budget_m_usd", "revenue_m_usd", "vote_count", "genre"]
    available_features = [c for c in feature_candidates if df[c].notna().sum() > 0]
    default_features = [c for c in ["year", "runtime_min", "budget_m_usd", "vote_count", "genre"] if c in available_features]
    features = st.multiselect("Model inputs", available_features, default=default_features)
    model_df = df.dropna(subset=["rating"]).copy()
    if len(model_df) >= 30 and len(features) > 0:
        X = model_df[features].copy()
        y = model_df["rating"].copy()
        numeric_features = [c for c in features if c != "genre"]
        categorical_features = [c for c in features if c == "genre"]
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers = []
        if numeric_features:
            transformers.append(("num", numeric_pipe, numeric_features))
        if categorical_features:
            transformers.append(("cat", categorical_pipe, categorical_features))
        preprocessor = ColumnTransformer(transformers)
        model = Pipeline([("preprocess", preprocessor), ("regressor", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2))])
        test_size = 0.2 if len(model_df) >= 50 else 0.25
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, 10)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean absolute error", f"{mae:.2f} points")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("R² score", f"{r2:.2f}" if np.isfinite(r2) else "N/A")
        st.caption("Lower MAE/RMSE is better. R² closer to 1 generally indicates better fit, but a weak or negative score can be valid.")
        result = pd.DataFrame({"Actual rating": y_test.to_numpy(), "Predicted rating": preds})
        st.plotly_chart(px.scatter(result, x="Actual rating", y="Predicted rating", title="Actual vs predicted ratings", trendline=None), use_container_width=True)
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
    else:
        st.warning("The model needs at least 30 usable rows and one selected feature. Upload a larger dataset or use the demo.")

with tab4:
    st.subheader("Quick data-led observations")
    genre_summary = df.groupby("genre").agg(movies=("title","count"), mean_rating=("rating","mean"), median_runtime=("runtime_min","median")).reset_index()
    genre_summary = genre_summary[genre_summary["movies"] >= min(5, max(1, int(len(df)*0.01)))]
    if not genre_summary.empty:
        top = genre_summary.sort_values("mean_rating", ascending=False).iloc[0]
        bottom = genre_summary.sort_values("mean_rating", ascending=True).iloc[0]
        st.markdown(f"- **Highest average rating among represented genres:** {top['genre']} ({top['mean_rating']:.2f}/10; n={int(top['movies'])}).")
        st.markdown(f"- **Lowest average rating among represented genres:** {bottom['genre']} ({bottom['mean_rating']:.2f}/10; n={int(bottom['movies'])}).")
    corr_cols = ["rating", "runtime_min", "budget_m_usd", "revenue_m_usd", "vote_count"]
    corr = df[corr_cols].corr(numeric_only=True)
    if corr.shape[0] > 1:
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    st.warning("Correlation is not causation. Sample size, missing data, popularity and selection bias can change these patterns.")
    st.download_button("⬇️ Download cleaned dataset", df.to_csv(index=False).encode("utf-8"), "cinescope_cleaned_movies.csv", "text/csv")

with tab5:
    st.subheader("How the project works")
    st.markdown("""
    **Research question:** Which observable movie characteristics are associated with ratings, and how well can a simple model estimate ratings?

    **Workflow**
    1. Load a CSV or generate clearly labelled sample data.
    2. Standardize column names, convert numeric fields and remove unusable records.
    3. Use exploratory data analysis (EDA) to inspect distributions and relationships.
    4. Train a Random Forest regression model on a training split and evaluate it on held-out data.
    5. Report limitations instead of treating correlations or predictions as proof.

    **Core concepts**
    - *Mean / median:* summarize typical values.
    - *Correlation:* measures linear association, not cause and effect.
    - *Regression:* estimates a continuous target (rating).
    - *Train/test split:* checks performance on data not used for fitting.
    - *MAE:* average absolute prediction error in rating points.
    - *R²:* compares model predictions with a simple mean-based baseline.

    **Limitations**
    - The built-in dataset is synthetic and must not be presented as real-world evidence.
    - Real datasets may have different column names, missing values, currency units or multi-genre labels.
    - Votes, popularity and rating distributions can introduce bias.
    - Model quality depends on data quality and the features available.
    """)
    st.markdown("**Suggested report sections:** Abstract · Problem statement · Dataset & cleaning · EDA charts · Model & evaluation · Findings · Limitations · Conclusion.")
