import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pickle
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { padding-top: 1rem; }
.stTextArea textarea { font-size: 15px; }
.emotion-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 16px;
    font-weight: 600;
    margin-top: 4px;
}
.metric-box {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #e9ecef;
}
.pipeline-step {
    background: #f1f3f4;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    margin-bottom: 6px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ── NLTK setup ────────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk()

EMOTION_EMOJI = {
    "joy":      ("😄", "#FFF3CD", "#856404"),
    "sadness":  ("😢", "#D1ECF1", "#0C5460"),
    "anger":    ("😠", "#F8D7DA", "#721C24"),
    "fear":     ("😨", "#E2D9F3", "#432874"),
    "love":     ("❤️",  "#FDDDE6", "#842029"),
    "surprise": ("😲", "#D4EDDA", "#155724"),
}

# ── Data & model loading ──────────────────────────────────────────────────────
MODEL_FILE = "emotion_models.pkl"

@st.cache_resource
def load_or_train():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def train_models(data_path="train.txt"):
    if not os.path.exists(data_path):
        return None, None, None, None, None, None, None

    df = pd.read_csv(data_path, sep=";", header=None, names=["text", "emotion"])
    df["text"] = df["text"].apply(lambda x: x.lower())

    stop_words = set(stopwords.words("english"))

    def remove_stopwords(txt):
        return " ".join(w for w in txt.split() if w not in stop_words)

    df["text"] = df["text"].apply(remove_stopwords)

    label_map = {emo: i for i, emo in enumerate(df["emotion"].unique())}
    label_inv = {v: k for k, v in label_map.items()}
    df["emotion"] = df["emotion"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=0.20, random_state=42
    )

    bow = CountVectorizer()
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow  = bow.transform(X_test)
    nb_bow = MultinomialNB().fit(X_train_bow, y_train)
    acc_nb_bow = accuracy_score(y_test, nb_bow.predict(X_test_bow))

    tfidf = TfidfVectorizer()
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf  = tfidf.transform(X_test)
    nb_tfidf = MultinomialNB().fit(X_train_tf, y_train)
    acc_nb_tfidf = accuracy_score(y_test, nb_tfidf.predict(X_test_tf))

    lr = LogisticRegression(max_iter=1000).fit(X_train_tf, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test_tf))

    bundle = dict(
        bow=bow, tfidf=tfidf,
        nb_bow=nb_bow, nb_tfidf=nb_tfidf, lr=lr,
        label_inv=label_inv,
        acc={"nb_bow": acc_nb_bow, "nb_tfidf": acc_nb_tfidf, "lr": acc_lr},
        stop_words=stop_words,
    )
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

# ── Preprocess helper ─────────────────────────────────────────────────────────
def preprocess(text, stop_words):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

# ── Predict helper ────────────────────────────────────────────────────────────
def predict(text, model_key, bundle):
    clean = preprocess(text, bundle["stop_words"])
    if model_key in ("nb_bow",):
        vec = bundle["bow"].transform([clean])
        model = bundle["nb_bow"]
    elif model_key == "nb_tfidf":
        vec = bundle["tfidf"].transform([clean])
        model = bundle["nb_tfidf"]
    else:
        vec = bundle["tfidf"].transform([clean])
        model = bundle["lr"]

    pred_idx = model.predict(vec)[0]
    pred_label = bundle["label_inv"][pred_idx]
    proba = model.predict_proba(vec)[0]
    classes = [bundle["label_inv"][c] for c in model.classes_]
    return pred_label, dict(zip(classes, proba)), clean

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎭 Emotion Detector")
st.caption("Text emotion classification using Naive Bayes & Logistic Regression")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    data_path = st.text_input("Dataset path", value="train.txt")
    model_choice = st.radio(
        "Model",
        ["Naive Bayes (BoW)", "Naive Bayes (TF-IDF)", "Logistic Regression", "Compare all"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Model accuracies**")
    acc_placeholder = st.empty()
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Pipeline: lowercase → stopword removal → vectorize → classify")

MODEL_KEY_MAP = {
    "Naive Bayes (BoW)":   "nb_bow",
    "Naive Bayes (TF-IDF)": "nb_tfidf",
    "Logistic Regression":  "lr",
}

# Load / train
bundle = load_or_train() or train_models(data_path)

if bundle is None:
    st.warning(
        f"⚠️ `{data_path}` not found. Place your `train.txt` (format: `text;emotion`) "
        "in the same folder and restart."
    )
    st.stop()

# Show accuracies in sidebar
with acc_placeholder.container():
    for label, key in MODEL_KEY_MAP.items():
        st.metric(label.split("(")[0].strip(), f"{bundle['acc'][key]*100:.1f}%")

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter text")

examples = [
    "I can't believe how wonderful this day has been!",
    "I miss you so much, it hurts to even think about it.",
    "This is absolutely infuriating. I've had enough.",
    "Something feels wrong. I don't know what's going to happen.",
    "I love you more than words can say.",
    "I never expected that! What a surprise!",
]

with st.expander("💡 Try an example"):
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex[:45] + "…" if len(ex) > 45 else ex, key=f"ex_{i}"):
            st.session_state["input_text"] = ex

user_text = st.text_area(
    "Your text",
    value=st.session_state.get("input_text", ""),
    height=120,
    placeholder="Type something… e.g. I feel so happy today!",
    label_visibility="collapsed",
)

analyze_btn = st.button("🔍 Analyze emotion", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if analyze_btn and user_text.strip():
    st.markdown("---")

    if model_choice == "Compare all":
        st.markdown("### 📊 Model comparison")
        cols = st.columns(3)
        predictions = {}
        for col, (label, key) in zip(cols, MODEL_KEY_MAP.items()):
            pred, scores, _ = predict(user_text, key, bundle)
            predictions[key] = pred
            emoji, bg, fg = EMOTION_EMOJI.get(pred, ("🤔", "#f8f9fa", "#333"))
            with col:
                st.markdown(f"""
                <div class="metric-box">
                  <div style="font-size:12px;color:#666;margin-bottom:4px;">{label}</div>
                  <div style="font-size:28px;">{emoji}</div>
                  <div style="font-size:16px;font-weight:600;color:{fg};">{pred}</div>
                  <div style="font-size:12px;color:#888;margin-top:4px;">
                    {bundle['acc'][key]*100:.1f}% accuracy
                  </div>
                </div>
                """, unsafe_allow_html=True)

        all_agree = len(set(predictions.values())) == 1
        if all_agree:
            st.success(f"✅ All models agree: **{list(predictions.values())[0]}**")
        else:
            st.warning("⚠️ Models disagree — check the highest-accuracy result (Logistic Regression).")

        # Bar chart comparison
        st.markdown("#### Confidence scores — Logistic Regression")
        _, scores, _ = predict(user_text, "lr", bundle)
        fig, ax = plt.subplots(figsize=(7, 3))
        emotions = list(scores.keys())
        values = [scores[e] * 100 for e in emotions]
        colors = [EMOTION_EMOJI.get(e, ("", "#ccc", "#333"))[1] for e in emotions]
        bars = ax.barh(emotions, values, color=colors, edgecolor="#ccc", linewidth=0.5)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        key = MODEL_KEY_MAP[model_choice]
        pred, scores, clean_text = predict(user_text, key, bundle)
        emoji, bg, fg = EMOTION_EMOJI.get(pred, ("🤔", "#f8f9fa", "#333"))

        st.markdown(f"### Result — {model_choice}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:12px;color:#666;">Detected emotion</div>
              <div style="font-size:32px;">{emoji}</div>
              <div class="emotion-badge" style="background:{bg};color:{fg};">{pred}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            top_conf = scores[pred] * 100
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:12px;color:#666;">Confidence</div>
              <div style="font-size:32px;font-weight:700;color:{fg};">{top_conf:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:12px;color:#666;">Test accuracy</div>
              <div style="font-size:32px;font-weight:700;">{bundle['acc'][key]*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Bar chart
        st.markdown("#### Emotion distribution")
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = [EMOTION_EMOJI.get(e, ("", "#ccc", "#333"))[1] for e in sorted_scores]
        bars = ax.barh(list(sorted_scores.keys()), [v * 100 for v in sorted_scores.values()],
                       color=colors, edgecolor="#ccc", linewidth=0.5)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, sorted_scores.values()):
            ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

        # Pipeline
        with st.expander("🔬 NLP pipeline details"):
            st.markdown("**Step 1 — Lowercased**")
            st.markdown(f'<div class="pipeline-step">{user_text.lower()[:100]}</div>', unsafe_allow_html=True)
            st.markdown("**Step 2 — After stopword removal**")
            st.markdown(f'<div class="pipeline-step">{clean_text[:100]}</div>', unsafe_allow_html=True)
            st.markdown(f"**Step 3 — Vectorizer:** {'CountVectorizer (BoW)' if key == 'nb_bow' else 'TF-IDF'}")
            st.markdown(f"**Step 4 — Classifier:** {'Multinomial Naive Bayes' if 'nb' in key else 'Logistic Regression'}")

elif analyze_btn:
    st.warning("Please enter some text first.")