import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

# Standardize column headers
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "_", regex=False)
)


# --- DYNAMIC FEATURE EXTRACTION ---
def get_num(s):
  return (
      s.astype(str)
      .str.replace(",", "")
      .str.extract(r"(\d+\.?\d*)")[0]
      .astype(float)
  )


def pick_col(data, opts):
  for c in opts:
    if c in data.columns:
      return data[c]
  # Fallback search for partial string matches
  for opt in opts:
    matches = [c for c in data.columns if opt in c]
    if matches:
      return data[matches[0]]
  return pd.Series(dtype=object)


# Find speed column dynamically to avoid KeyError
speed_series = pick_col(df, ["speed_mph", "speed", "mph"])
if speed_series.empty:
  # Fallback to first numeric column if no speed column is found
  speed_series = df.select_dtypes(include=[np.number]).iloc[:, 0]

df["speed_mph_clean"] = get_num(speed_series)
df = df.dropna(subset=["speed_mph_clean"])

# Extract height, length, inversions, and year introduced
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

# --- FEATURE ENGINEERING FOR CLASSIFICATION ---
# Create binary target: 1 if speed >= median speed, else 0
median_speed = df["speed_mph_clean"].median()
df["is_fast_coaster"] = (df["speed_mph_clean"] >= median_speed).astype(int)
target = "is_fast_coaster"

df_clean = df.dropna(subset=[target]).copy()

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

df_clean["type_clean"] = (
    t_col.astype(str).replace(["nan", "None", ""], "Unknown")
    if not t_col.empty
    else "Unknown"
)

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

# --- PIPELINE & STRATIFIED SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

num_pipe = Pipeline(
    [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
)

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# --- BASE EVALUATION (CLASSIFICATION PART A) ---
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=SEED
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "Support Vector Machine (SVC)": SVC(
        C=1.0, kernel="rbf", probability=True, random_state=SEED
    ),
}

res = []
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, clf) in enumerate(models.items()):
  p = Pipeline([("prep", prep), ("clf", clf)])
  p.fit(X_train, y_train)
  preds = p.predict(X_test)

  acc = accuracy_score(y_test, preds)
  prec = precision_score(y_test, preds, average="weighted", zero_division=0)
  rec = recall_score(y_test, preds, average="weighted", zero_division=0)
  f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

  res.append({
      "Model": name,
      "Accuracy": round(acc, 4),
      "Precision": round(prec, 4),
      "Recall": round(rec, 4),
      "Weighted F1": round(f1, 4),
  })

  # Render Confusion Matrix Plot
  cm = confusion_matrix(y_test, preds)
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      ax=axes[idx],
      cbar=False,
      annot_kws={"size": 12},
  )
  axes[idx].set_title(f"{name}", fontsize=11, fontweight="bold")
  axes[idx].set_xlabel("Predicted Label")
  axes[idx].set_ylabel("True Label")

# Remove extra 6th subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig("part_a_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- SUMMARY TABLE ---
summary = (
    pd.DataFrame(res)
    .sort_values(by="Weighted F1", ascending=False)
    .reset_index(drop=True)
)
print("--- REVIEW 1: CLASSIFICATION PART A RESULTS ---")
print(summary.to_string(index=False))