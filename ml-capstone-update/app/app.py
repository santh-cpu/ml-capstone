"""Streamlit app for the ML capstone: type in values, get predictions from the trained models.

Run from the repo root:   streamlit run app/app.py
Needs the models saved by the last cells of Review2_Final_ML_Capstone.ipynb (models/*.pkl).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

st.set_page_config(page_title="ML Capstone", layout="centered")
st.title("ML Capstone - Team 18")
st.caption("23CSE301 Machine Learning.")


@st.cache_resource
def load(name):
    path = MODELS / name
    return joblib.load(path) if path.exists() else None


def fields_of(pipe):
    """Input fields a fitted preprocessing pipeline actually uses: (numeric [(name, median)], categorical [(name, categories)])."""
    prep = pipe.named_steps["prep"]
    nums, cats = [], []
    for name, trans, cols in prep.transformers_:
        if name == "num":
            stats = trans.named_steps["imputer"].statistics_
            nums += [(c, float(s)) for c, s in zip(cols, stats) if not np.isnan(s)]
        elif name == "cat":
            enc = trans.named_steps["encoder"]
            cats += [
                (c, [str(x) for x in cats_]) for c, cats_ in zip(cols, enc.categories_)
            ]
    return nums, cats


def predict_form(key, pipe, examples=None, label_fn=None):
    """Draws the input widgets for a pipeline and returns a one-row DataFrame once the button is pressed."""
    nums, cats = fields_of(pipe)
    all_cols = list(pipe.named_steps["prep"].feature_names_in_)
    chosen = {}
    if examples is not None and len(examples):
        pick = st.selectbox(
            "Start from an example",
            ["(type my own values)"] + list(examples.index),
            key=f"{key}_ex",
        )
        if pick != "(type my own values)":
            chosen = examples.loc[pick].to_dict()
    vals = {}
    cols = st.columns(3)
    for i, (name, med) in enumerate(nums):
        start = chosen.get(name)
        start = med if start is None or pd.isna(start) else float(start)
        vals[name] = cols[i % 3].number_input(
            name.replace("_", " "),
            value=start,
            key=f"{key}_{name}_{pick if chosen else ''}",
        )
    for j, (name, options) in enumerate(cats):
        start = chosen.get(name)
        idx = (
            options.index(str(start))
            if start is not None and str(start) in options
            else 0
        )
        vals[name] = cols[(len(nums) + j) % 3].selectbox(
            name.replace("_", " "),
            options,
            index=idx,
            key=f"{key}_{name}_{pick if chosen else ''}",
        )
    row = {
        c: vals.get(c, np.nan) for c in all_cols
    }  # columns the model dropped stay empty
    return pd.DataFrame([row]), chosen


tab_reg, tab_cls, tab_clu = st.tabs(["Regression", "Classification", "Clustering"])

# ---------------------------------------------------------------- regression
with tab_reg:
    st.subheader("Roller coaster top speed")
    reg = load("coaster_speed_regressor.pkl")
    if reg is None:
        st.error(
            "models/coaster_speed_regressor.pkl not found. Run the notebook first."
        )
    else:
        st.write(
            "Gradient Boosting Regressor (tuned). On the test set its R² was about 0.12 and RMSE about 9.7 mph, so treat predictions as rough."
        )
        ex = None
        csv = ROOT / "coaster_db.csv"
        if csv.exists():
            raw = pd.read_csv(csv)
            sp = pd.to_numeric(
                raw["Speed"].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
                errors="coerce",
            )
            ht = (
                pd.to_numeric(
                    raw["Height"].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
                    errors="coerce",
                )
                if "Height" in raw
                else np.nan
            )
            ex = pd.DataFrame(
                {
                    "height_ft": ht,
                    "Inversions_clean": raw.get("Inversions"),
                    "Type_Main": raw.get("Type"),
                }
            )
            ex.index = raw["coaster_name"].astype(str)
            ex = ex[sp.notna().values].dropna(how="all").head(40)
        X, chosen = predict_form("reg", reg, ex)
        if st.button("Predict speed", key="go_reg"):
            st.metric("Predicted top speed", f"{reg.predict(X)[0]:.1f} mph")

# ---------------------------------------------------------------- classification
with tab_cls:
    st.subheader("Is this Pokémon legendary?")
    cls = load("pokemon_legendary_classifier.pkl")
    if cls is None:
        st.error(
            "models/pokemon_legendary_classifier.pkl not found. Run the notebook first."
        )
    else:
        st.write(f"Final tuned model: {type(cls.named_steps['clf']).__name__}.")
        ex = None
        csv = ROOT / "pokemon.csv"
        if csv.exists():
            p = pd.read_csv(csv)
            p["capture_rate"] = pd.to_numeric(p["capture_rate"], errors="coerce")
            ex = p.set_index("name")[
                [c for c in cls.named_steps["prep"].feature_names_in_ if c in p.columns]
            ].head(60)
        X, chosen = predict_form("cls", cls, ex)
        if st.button("Classify", key="go_cls"):
            prob = float(cls.predict_proba(X)[0, 1])
            st.metric("Prediction", "Legendary" if prob >= 0.5 else "Not legendary")
            st.progress(prob, text=f"Probability of legendary: {prob:.1%}")

# ---------------------------------------------------------------- clustering
with tab_clu:
    st.subheader("Which cluster does a drink belong to?")
    bundle = load("starbucks_clustering.pkl")
    if bundle is None:
        st.error("models/starbucks_clustering.pkl not found. Run the notebook first.")
    else:
        feats, items = bundle["features"], bundle["items"]
        km, sc = bundle["kmeans"], bundle["scaler"]
        st.write(
            f"K-Means with k = {km.n_clusters} on {len(feats)} standardised nutrition features."
        )
        names = items["label"].tolist()
        pick = st.selectbox("Start from a menu item", names, key="clu_pick")
        base = (
            items.set_index("label").loc[pick, feats]
            if names.count(pick) == 1
            else items[items["label"] == pick].iloc[0][feats]
        )
        cols = st.columns(3)
        vals = [
            cols[i % 3].number_input(
                f.replace("_", " "), value=float(base[f]), key=f"clu_{f}_{pick}"
            )
            for i, f in enumerate(feats)
        ]
        if st.button("Find cluster", key="go_clu"):
            z = sc.transform(pd.DataFrame([vals], columns=feats).values)
            c = int(km.predict(z)[0])
            dist = np.linalg.norm(km.cluster_centers_ - z, axis=1)
            st.metric("Assigned cluster", f"Cluster {c}")
            st.write("Distance to each cluster centre (standardised units):")
            st.dataframe(
                pd.DataFrame(
                    {
                        "cluster": range(len(dist)),
                        "distance": dist.round(2),
                        "items in cluster": np.bincount(km.labels_),
                    }
                ).set_index("cluster")
            )
            allz = sc.transform(items[feats].values)
            near = (
                items.assign(d=np.linalg.norm(allz - z, axis=1))
                .sort_values("d")
                .head(3)["label"]
                .tolist()
            )
            st.write("Most similar menu items: " + "; ".join(near))
