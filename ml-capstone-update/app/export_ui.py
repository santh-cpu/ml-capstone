"""Builds ui.html (the results dashboard + a 'Try it' tab that runs the trained models offline).

Called from the last cell of Review2_Final_ML_Capstone.ipynb:

    from app.export_ui import build_ui
    build_ui(globals())

It reads the trained objects and result tables from the notebook namespace, converts the fitted scikit-learn
pipelines to plain data (tree structures, scaler constants, K-Means centroids) and writes one self-contained
HTML file. The page needs no server and no internet connection.
"""
import base64, io, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _clean(o):
    """Make numpy/pandas values JSON-safe (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _clean(json.loads(o.reset_index().to_json(orient="records")))
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


# ---------------------------------------------------------------- model -> plain data
def _tree(t):
    tr = t.tree_ if hasattr(t, "tree_") else t
    val = tr.value[:, 0, :]
    if val.shape[1] > 1:                       # classifier: class probabilities
        value = (val / val.sum(axis=1, keepdims=True)).tolist()
    else:                                      # regressor / boosting stage: one number per node
        value = val[:, 0].tolist()
    return {"left": tr.children_left.tolist(), "right": tr.children_right.tolist(),
            "feature": tr.feature.tolist(), "threshold": tr.threshold.tolist(), "value": value}


def export_preprocessor(prep):
    """ColumnTransformer(num=[imputer, scaler], cat=[imputer, one-hot]) -> constants.
    Columns the imputer dropped (no observed values) do not reach the model, so they are skipped here too."""
    spec = {"numeric": [], "categorical": []}
    for name, trans, cols in prep.transformers_:
        if name == "remainder":
            continue
        if name == "num":
            imp, sc = trans.named_steps["imputer"], trans.named_steps["scaler"]
            kept = [c for c, s in zip(cols, imp.statistics_) if not np.isnan(s)]
            stats = [s for s in imp.statistics_ if not np.isnan(s)]
            for i, c in enumerate(kept):
                spec["numeric"].append({"name": c, "median": float(stats[i]), "mean": float(sc.mean_[i]),
                                        "scale": float(sc.scale_[i])})
        elif name == "cat":
            imp, enc = trans.named_steps["imputer"], trans.named_steps["encoder"]
            for i, c in enumerate(cols):
                spec["categorical"].append({"name": c, "fill": str(imp.statistics_[i]),
                                            "categories": [str(x) for x in enc.categories_[i]]})
        else:
            raise ValueError(f"unexpected transformer block: {name}")
    return spec


def export_estimator(est):
    kind = type(est).__name__
    if kind in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        return {"kind": "tree", "tree": _tree(est)}
    if kind in ("RandomForestClassifier", "RandomForestRegressor"):
        return {"kind": "forest", "trees": [_tree(t) for t in est.estimators_]}
    if kind == "GradientBoostingClassifier":
        p = float(est.init_.class_prior_[1])
        return {"kind": "gb", "init": float(np.log(p / (1 - p))), "lr": float(est.learning_rate),
                "trees": [_tree(est.estimators_[i, 0]) for i in range(est.estimators_.shape[0])]}
    if kind == "GradientBoostingRegressor":
        return {"kind": "gbr", "init": float(np.ravel(est.init_.constant_)[0]), "lr": float(est.learning_rate),
                "trees": [_tree(est.estimators_[i, 0]) for i in range(est.estimators_.shape[0])]}
    if kind == "LogisticRegression":
        return {"kind": "linear", "coef": est.coef_[0].tolist(), "intercept": float(est.intercept_[0])}
    raise ValueError(f"{kind} cannot be exported to the web UI (supported: decision tree, random forest, "
                     f"gradient boosting, logistic regression)")


def export_pipeline(pipe, step):
    return {"preprocessor": export_preprocessor(pipe.named_steps["prep"]),
            "model": export_estimator(pipe.named_steps[step]),
            "model_name": type(pipe.named_steps[step]).__name__}


def _examples(frame, name_col, feats, target=None, n_each=None, seed=7):
    cols = [c for c in feats if c in frame.columns]
    out = []
    groups = [(None, frame)] if n_each is None else [(v, frame[frame[target] == v]) for v in (0, 1)]
    for flag, g in groups:
        g = g.dropna(subset=[name_col])
        take = g.sample(min(len(g), n_each or 8), random_state=seed)
        for _, r in take.iterrows():
            out.append({"label": str(r[name_col]),
                        "values": {c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c])) for c in cols},
                        "actual": None if target is None or n_each is not None else (None if pd.isna(r[target]) else float(r[target])),
                        "legendary": None if n_each is None else bool(flag)})
    return out


def _dendrogram_png(X_scaled, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, m in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
        Z = linkage(X_scaled, method=m)
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, show_contracted=True,
                   color_threshold=0, above_threshold_color="#2a78d6", ax=ax)
        ax.axhline((Z[-k, 2] + Z[-(k - 1), 2]) / 2, color="red", ls="--", label=f"cut for k={k}")
        ax.set_title(f"dendrogram ({m} linkage, last 30 merges)")
        ax.set_xlabel("cluster (number of items in brackets)")
        ax.set_ylabel("merge distance")
        ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=70, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_ui(ns, out="ui.html"):
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    g = ns.__getitem__

    # ---------- regression
    reg_pipe = g("best_overall")
    reg_names = [n.split("__", 1)[1] for n in reg_pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.DataFrame({"Feature": reg_names, "Importance": reg_pipe.named_steps["reg"].feature_importances_})
    reg = {"table": g("results_df"), "n_train": len(g("X_train")), "n_test": len(g("X_test")),
           "features": g("num_features") + g("cat_features"), "target": g("speed_col"),
           "best_name": str(g("best_overall_name")), "y_test": np.asarray(g("y_test")), "pred": np.asarray(g("tuned_preds")),
           "importance": imp.sort_values("Importance", ascending=False).head(10),
           "rf_tuned": {"params": g("grid_rf").best_params_, "base": g("baseline_rf_r2"), "tuned": g("tuned_rf_r2")},
           "gb_tuned": {"params": g("grid_gb").best_params_, "base": g("baseline_gb_r2"), "tuned": g("tuned_gb_r2")}}

    # ---------- classification
    roc, cms = {}, {}
    for n, info in g("all_cls").items():
        s = g("get_scores")(info["pipe"], g("Xp_test"))
        f_, t_, _ = roc_curve(g("yp_test"), s)
        roc[n] = {"fpr": f_, "tpr": t_, "auc": roc_auc_score(g("yp_test"), s)}
        cms[n] = confusion_matrix(g("yp_test"), info["preds"])
    cls = {"table": g("final_cls_df"), "roc": roc, "cm": cms, "cv": pd.DataFrame(g("cv_rows")), "tune": g("tune_df"),
           "tune_params": g("grid_cls").best_params_, "tuned_name": g("final_name"), "rf_importance": g("rf_imp"),
           "n_train": len(g("Xp_train")), "n_test": len(g("Xp_test")), "legendary_share": float(g("yp").mean()), "n": len(g("yp"))}

    # ---------- clustering
    df, cols, km = g("df"), list(g("numeric_cols")), g("kmeans")
    Xc = df[cols].values
    names = [f"{b} ({p})" for b, p in zip(df["beverage"].astype(str), df["beverage_prep"].astype(str))]
    clu = {"ks": list(g("ks")), "inertias": g("inertias"), "k": g("K_CHOSEN"), "linkage": g("HIER_LINKAGE"),
           "eval": g("eval_df"), "link": g("link_rows"), "pca": g("X_pca"), "tsne": g("X_tsne"), "evr": g("evr"),
           "km": g("km_labels"), "agg": g("agg_labels"), "profile": df[cols].groupby(g("km_labels")).mean().round(1),
           "features": cols, "n": len(df), "names": names}

    # ---------- models for the "Try it" tab
    poke_feats = g("poke_num_features") + g("poke_cat_features")
    pdf = g("poke_df").copy()
    if "capture_rate" in pdf.columns:
        pdf["capture_rate"] = pd.to_numeric(pdf["capture_rate"], errors="coerce")
    coaster = g("df_clean")
    cname = next((c for c in coaster.columns if c.lower() == "coaster_name"), coaster.columns[0])
    models = {
        "reg": {**export_pipeline(reg_pipe, "reg"), "target": g("speed_col"),
                "examples": _examples(coaster.dropna(subset=[g("speed_col")]), cname,
                                      [c["name"] for c in export_pipeline(reg_pipe, "reg")["preprocessor"]["numeric"]] +
                                      [c["name"] for c in export_pipeline(reg_pipe, "reg")["preprocessor"]["categorical"]],
                                      target=g("speed_col"))},
        "cls": {**export_pipeline(g("best_cls"), "clf"),
                "examples": _examples(pdf, "name", poke_feats, target="is_legendary", n_each=4)},
        "clu": {"features": cols, "mean": Xc.mean(axis=0).tolist(), "std": Xc.std(axis=0).tolist(),
                "median": np.median(Xc, axis=0).tolist(), "centers": km.cluster_centers_.tolist(),
                "sizes": np.bincount(km.labels_).tolist(), "profile": df[cols].groupby(km.labels_).mean().values.tolist(),
                "items": [{"label": n, "values": [None if pd.isna(v) else float(v) for v in row]} for n, row in zip(names, Xc)]},
    }
    reg_rmse = float(np.sqrt(np.mean((np.asarray(g("y_test")) - np.asarray(g("tuned_preds"))) ** 2)))
    yt, pt = np.asarray(g("y_test")), np.asarray(g("tuned_preds"))
    models["reg"]["test_r2"] = float(1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    models["reg"]["test_rmse"] = reg_rmse
    models["reg"]["model_label"] = str(g("best_overall_name"))

    data = _clean({"reg": reg, "cls": cls, "clu": clu, "models": models})
    html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__DENDRO__", _dendrogram_png(g("X_scaled"), g("K_CHOSEN")))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KB)")
