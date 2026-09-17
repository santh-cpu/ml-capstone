import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer

# --- IMPORTS ---
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning, module=r".*impute.*")
warnings.filterwarnings(
    "ignore", message=r".*Skipping features without any observed values.*"
)

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
SEED = 42

# --- LOAD DATA ---
path = None
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".csv"):
            path = os.path.join(root, f)
            break
    if path:
        break

if not path:
    csvs = glob.glob("*.csv")
    if csvs:
        path = csvs[0]

if not path:
    raise FileNotFoundError("No CSV found.")

df = pd.read_csv(path)
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "_", regex=False)
)

df["speed_mph"] = pd.to_numeric(df["speed_mph"], errors="coerce")
df = df.dropna(subset=["speed_mph"])


# --- CLEAN FEATURES ---
def get_num(s):
    return (
        s.astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0].astype(float)
    )


def pick_col(data, opts):
    for c in opts:
        if c in data.columns:
            return data[c]
    return pd.Series(dtype=object)


df["speed_mph_clean"] = get_num(pick_col(df, ["speed_mph", "speed"]))

h1 = get_num(pick_col(df, ["height_ft", "height_value_ft"]))
h2 = get_num(pick_col(df, ["height"]))
hm = get_num(pick_col(df, ["height_m", "height_value_m"])) * 3.28084
df["height_ft_clean"] = h1.fillna(h2).fillna(hm)

l1 = get_num(pick_col(df, ["length_ft", "length_value_ft", "length"]))
lm = get_num(pick_col(df, ["length_m", "length_value_m"])) * 3.28084
df["length_ft_clean"] = l1.fillna(lm)

df["inversions_clean"] = get_num(
    pick_col(df, ["inversions_clean", "inversions", "num_inversions"])
).fillna(0)
df["year_introduced"] = get_num(
    pick_col(df, ["year_introduced", "opening_date", "year", "opened"])
)

target = "speed_mph_clean"
df_clean = (
    df.dropna(subset=[target])
    .dropna(subset=["height_ft_clean", "length_ft_clean"], how="all")
    .copy()
)

num_cols = [
    "height_ft_clean",
    "length_ft_clean",
    "inversions_clean",
    "year_introduced",
]
for c in num_cols:
    if df_clean[c].isna().all():
        df_clean[c] = 0.0

t_col = pick_col(df_clean, ["type_main", "type"])
m_col = pick_col(df_clean, ["manufacturer"])
df_clean["type_clean"] = t_col.astype(str).replace(["nan", "None", ""], "Unknown")

if not m_col.empty:
    rare = m_col.value_counts()[m_col.value_counts() < 3].index
    df_clean["manufacturer_clean"] = m_col.replace(rare, "Other").astype(str)
    cat_cols = ["type_clean", "manufacturer_clean"]
else:
    cat_cols = ["type_clean"]

for c in cat_cols:
    df_clean[c] = df_clean[c].fillna("Unknown")

X = df_clean[num_cols + cat_cols]
y = df_clean[target]

# --- PIPELINE & SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Standard preprocessing
num_pipe = Pipeline(
    [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
)

cat_pipe = Pipeline(
    [
        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

prep = ColumnTransformer(
    [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

# Special numerical preprocessing for Polynomial Regression (applies degree scaling)
num_poly_pipe = Pipeline(
    [
        ("imp", SimpleImputer(strategy="median")),
        (
            "poly",
            PolynomialFeatures(degree=2, include_bias=False),
        ),  # default degree 2
        ("scale", StandardScaler()),
    ]
)

prep_poly = ColumnTransformer(
    [
        ("num", num_poly_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

# --- BASE EVALUATION ---
models = {
    # Standard Linear Models
    "Linear Regression": (Pipeline([("prep", prep), ("reg", LinearRegression())])),
    "Ridge Regression": (Pipeline([("prep", prep), ("reg", Ridge(random_state=SEED))])),
    "Lasso Regression": (Pipeline([("prep", prep), ("reg", Lasso(random_state=SEED))])),
    "ElasticNet Regression": (
        Pipeline([("prep", prep), ("reg", ElasticNet(random_state=SEED))])
    ),
    "Polynomial Regression": Pipeline(
        [("prep", prep_poly), ("reg", LinearRegression())]
    ),
    # Non-linear Models
    "SVR": Pipeline([("prep", prep), ("reg", SVR(C=10.0, kernel="rbf"))]),
    "KNN": Pipeline(
        [("prep", prep), ("reg", KNeighborsRegressor(n_neighbors=5, n_jobs=-1))]
    ),
    "Random Forest": Pipeline(
        [
            ("prep", prep),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1
                ),
            ),
        ]
    ),
    "Gradient Boosting": Pipeline(
        [
            ("prep", prep),
            (
                "reg",
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "Decision Tree": Pipeline(
        [
            ("prep", prep),
            ("reg", DecisionTreeRegressor(max_depth=8, random_state=SEED)),
        ]
    ),
}

res = []
for name, p in models.items():
    p.fit(X_train, y_train)
    preds = p.predict(X_test)

    res.append(
        {
            "Model": name,
            "R2 Score": round(r2_score(y_test, preds), 4),
            "RMSE (mph)": round(root_mean_squared_error(y_test, preds), 2),
            "MAE (mph)": round(mean_absolute_error(y_test, preds), 2),
        }
    )

summary = (
    pd.DataFrame(res).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
)
print(summary.to_string(index=False))

# --- TUNING ---
top2 = summary["Model"].head(2).tolist()

grids = {
    "Linear Regression": {},
    "Ridge Regression": {"reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression": {"reg__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet Regression": {
        "reg__alpha": [0.01, 0.1, 1.0, 10.0],
        "reg__l1_ratio": [0.2, 0.5, 0.7, 0.9],
    },
    "Polynomial Regression": {
        "prep__num__poly__degree": [2, 3],
        "prep__num__poly__interaction_only": [True, False],
    },
    "SVR": {
        "reg__C": [0.1, 1.0, 10.0, 100.0],
        "reg__gamma": ["scale", "auto"],
        "reg__epsilon": [0.01, 0.1, 0.2],
    },
    "KNN": {
        "reg__n_neighbors": [3, 5, 7, 9, 11],
        "reg__weights": ["uniform", "distance"],
        "reg__metric": ["minkowski", "manhattan"],
    },
    "Random Forest": {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [6, 10, 14, None],
    },
    "Gradient Boosting": {
        "reg__n_estimators": [100, 200],
        "reg__learning_rate": [0.03, 0.1, 0.2],
        "reg__max_depth": [3, 5, 7],
    },
    "Decision Tree": {
        "reg__max_depth": [4, 6, 8, 12, None],
        "reg__min_samples_leaf": [1, 3, 5],
    },
}

tuned_fits = {}
scores = {}

for name in top2:
    p = models[name]
    grid = grids[name]

    if grid:
        gs = GridSearchCV(p, grid, cv=5, scoring="r2", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_
    else:
        p.fit(X_train, y_train)
        best_est = p

    tuned_fits[name] = best_est
    scores[name] = r2_score(y_test, best_est.predict(X_test))

best_name = max(scores, key=scores.get)
best_p = tuned_fits[best_name]
best_preds = best_p.predict(X_test)

# --- PLOTS ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(
    x=best_preds, y=y_test - best_preds, alpha=0.5, ax=ax[0], color="purple"
)
ax[0].axhline(y=0, color="red", linestyle="--")
ax[0].set(
    title=f"Residuals ({best_name})",
    xlabel="Predicted (mph)",
    ylabel="Residuals (mph)",
)

sns.scatterplot(x=y_test, y=best_preds, alpha=0.5, ax=ax[1], color="teal")
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax[1].set(
    title=f"Pred vs Actual ({best_name})",
    xlabel="Actual (mph)",
    ylabel="Predicted (mph)",
)

plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches="tight")
plt.show()

# --- FEATURE IMPORTANCE / COEFFICIENTS ---
if best_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
    ohe_cols = list(
        best_p.named_steps["prep"]
        .named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(cat_cols)
    )
    all_cols = num_cols + ohe_cols

    imp = pd.DataFrame(
        {
            "Feature": all_cols,
            "Importance": best_p.named_steps["reg"].feature_importances_,
        }
    )
    imp = imp.sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(9, 5))
    sns.barplot(x="Importance", y="Feature", data=imp, palette="viridis")
    plt.title(f"Feature Importance ({best_name})")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

elif best_name in [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet Regression",
]:
    ohe_cols = list(
        best_p.named_steps["prep"]
        .named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(cat_cols)
    )
    all_cols = num_cols + ohe_cols

    coefs = pd.DataFrame(
        {"Feature": all_cols, "Coefficient": best_p.named_steps["reg"].coef_}
    )
    coefs["Abs_Coef"] = coefs["Coefficient"].abs()
    coefs = coefs.sort_values(by="Abs_Coef", ascending=False).head(10)

    plt.figure(figsize=(9, 5))
    sns.barplot(x="Coefficient", y="Feature", data=coefs, palette="coolwarm")
    plt.title(f"Top Model Coefficients ({best_name})")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
