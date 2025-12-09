import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="HNG Engagement Predictor", layout="wide")

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_PATH = BASE/ "messages_clean.csv"


# Utility and caching
@st.cache_resource
def load_meta_model():
    meta_model_path = MODEL_DIR / "xgb_meta.joblib"
    scaler_path = MODEL_DIR / "meta_scaler.joblib"

    model = joblib.load(meta_model_path) if meta_model_path.exists() else None
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return model, scaler

@st.cache_resource
def load_embedder():
    try:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        embedder = None
    return embedder

@st.cache_data(ttl=3600)
def load_data(nrows=None):
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    if nrows:
        return df.sample(nrows, random_state=42)
    return df


#artifacts
meta_model, meta_scaler = load_meta_model()
embedder = load_embedder()

st.title("HNG Slack Engagement Predictor")
st.write("Predict whether a Slack message will get reactions or replies.")

# Left column: Input & Prediction
left, right = st.columns([2,3])

with left:
    st.header("Quick Predict")
    user_text = st.text_area("Paste a Slack message", height=150)
    hour = st.slider("Hour of day", 0, 23, 12)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, 1)
    author_msg_count = st.number_input("Author historical message count", min_value=0, value=5)
    emoji_count = st.number_input("Emoji count estimate", min_value=0, value=0)

    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Paste some text to predict.")
        else:
            message_length = len(user_text)
            word_count = len(user_text.split())
            meta = np.array([[message_length, word_count, emoji_count, hour, weekday, author_msg_count]])
            pred_prob = None

            prod_model_path = MODEL_DIR / "xgb_sbert.joblib"
            if prod_model_path.exists() and embedder is not None:
                try:
                    prod = joblib.load(prod_model_path)
                    emb = embedder.encode([user_text])[0].reshape(1, -1)
                    if meta_scaler is not None:
                        meta_s = meta_scaler.transform(meta)
                    else:
                        meta_s = meta
                    X = np.hstack([emb, meta_s])
                    pred_prob = prod.predict_proba(X)[:,1][0]
                    st.success(f"Model (SBERT+XGB) predicted engagement probability: {pred_prob:.3f}")
                except Exception as e:
                    st.error(f"SBERT model error: {e}")
                    pred_prob = None
            if pred_prob is None and meta_model is not None and meta_scaler is not None:
                meta_s = meta_scaler.transform(meta)
                pred_prob = float(meta_model.predict_proba(meta_s)[:,1][0])
                st.success(f"Metadata model predicted engagement probability: {pred_prob:.3f}")
                st.info("This is a metadata-only prediction. For best performance use the SBERT+XGB model if available.")

            if pred_prob is None:
                st.error("No model available. Upload model artifacts in /models or deploy production model.")


# Right column: EDA quick tiles and charts
with right:
    st.header("Quick EDA snapshots")
    df = load_data(nrows=10000)
    total_messages = len(df)
    engaged_rate = df['engaged'].mean()
    st.metric("Total messages loaded", total_messages)
    st.metric("Overall engagement rate", f"{engaged_rate:.2%}")

    st.subheader("Top channels by messages")
    channel_counts = df['channel'].value_counts().head(10)
    st.bar_chart(channel_counts)

    st.subheader("Hourly activity heatmap")
    heatmap = df.pivot_table(index=df['weekday'], columns=df['hour'], values='message_id', aggfunc='count', fill_value=0)
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(heatmap, cmap="Blues", ax=ax)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Weekday")
    st.pyplot(fig)


# SHAP explainations

st.markdown("---")
st.header("Explainability (metadata SHAP)")

if meta_model is None or meta_scaler is None:
    st.warning("Metadata model or scaler missing. Place xgb_meta.joblib and meta_scaler.joblib into models/ to enable SHAP.")
else:
    df_full = load_data(nrows=2000)
    df_full['message_length'] = df_full['clean_text'].astype(str).apply(len)
    df_full['word_count'] = df_full['clean_text'].astype(str).apply(lambda x: len(x.split()))
    df_full['emoji_count'] = df_full.get('emoji_count', 0).fillna(0)
    weekday_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
    df_full['weekday'] = df_full['weekday'].map(weekday_map)

    auth_counts = df_full.groupby('user_id')['message_id'].count()
    df_full['author_message_count'] = df_full['user_id'].map(auth_counts).fillna(0)
    X_meta = df_full[['message_length','word_count','emoji_count','hour','weekday','author_message_count']].values
    X_meta_s = meta_scaler.transform(X_meta)

    with st.spinner("Computing SHAP on metadata model..."):
        explainer = shap.explainers.Tree(meta_model)
        shap_values = explainer(X_meta_s)

    fig_shap = shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(bbox_inches='tight')

    st.markdown("Feature effects on engagement. Use this to explain which metadata features push probability up or down.")


#deployment info

st.markdown("---")
st.write("Deployment notes")
#st.write("""
#- To enable SBERT predictions include xgb_sbert.joblib and meta_scaler.joblib in models/  
#- For smaller repo size, keep only metadata model in git and load SBERT model by name at runtime  
#- To run locally: `streamlit run app.py` in streamlit_app folder  
#""")
