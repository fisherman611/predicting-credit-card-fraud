"""EDA of competition train, stratified validation, and unlabeled test."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train", type=Path, required=True)
parser.add_argument("--test", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--oof", type=Path)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
palette = {"Train": "#247ba0", "Validation": "#ed7d31", "Test": "#666e7e"}

train_all = pd.read_csv(args.train)
test = pd.read_csv(args.test)
features = [c for c in test.columns if c != "id"]
assert "is_fraud" in train_all and set(features) == set(train_all.columns) - {"id", "is_fraud"}
assert train_all.id.is_unique and test.id.is_unique and not train_all.id.isin(test.id).any()
fit_idx, val_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train_all, train_all.is_fraud))
parts = {"Train": train_all.iloc[fit_idx], "Validation": train_all.iloc[val_idx], "Test": test}
numeric = [c for c in features if pd.api.types.is_numeric_dtype(test[c]) and not pd.api.types.is_bool_dtype(test[c])]
categorical = [c for c in features if c not in numeric]
summary = []
for name, frame in parts.items():
    row = {"split": name, "rows": len(frame), "columns": len(frame.columns),
           "missing_cells": int(frame.isna().sum().sum()),
           "duplicate_feature_rows": int(frame.duplicated(features).sum()),
           "id_min": int(frame.id.min()), "id_max": int(frame.id.max())}
    if "is_fraud" in frame:
        row.update(fraud_count=int(frame.is_fraud.sum()), fraud_rate=float(frame.is_fraud.mean()))
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(out / "split_summary.csv", index=False)

profile = []
for name, frame in parts.items():
    for col in features:
        s = frame[col]
        row = {"split": name, "feature": col, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
               "unique": int(s.nunique()), "kind": "numeric" if col in numeric else "categorical"}
        if col in numeric:
            q = s.quantile([0, .01, .25, .5, .75, .99, 1])
            row.update(mean=s.mean(), std=s.std(), min=q[0], q01=q[.01], q25=q[.25], median=q[.5],
                       q75=q[.75], q99=q[.99], max=q[1])
        profile.append(row)
pd.DataFrame(profile).to_csv(out / "feature_profile.csv", index=False)

drift = []
for col in features:
    ref = parts["Train"][col].dropna()
    for group in ("Validation", "Test"):
        other = parts[group][col].dropna()
        if col in numeric:
            score = float(ks_2samp(ref, other).statistic)
            metric = "KS"
        else:
            a = ref.astype(str).value_counts(normalize=True)
            b = other.astype(str).value_counts(normalize=True)
            score = float(a.sub(b, fill_value=0).abs().sum() / 2)
            metric = "TV"
        drift.append({"feature": col, "comparison": group, "metric": metric, "effect": score})
drift = pd.DataFrame(drift)
drift.to_csv(out / "distribution_effects.csv", index=False)

# Shared bins come from train, with values outside the 0.1–99.9 percentile
# clipped to the endpoints so every observation contributes to each histogram.
fig, axes = plt.subplots(5, 3, figsize=(16, 17), constrained_layout=True)
for ax, col in zip(axes.flat, numeric):
    lo, hi = parts["Train"][col].quantile([.001, .999])
    if lo == hi:
        lo, hi = parts["Train"][col].min(), parts["Train"][col].max()
    bins = np.linspace(lo, hi, 42) if lo < hi else 10
    for name, frame in parts.items():
        values = frame[col].clip(lo, hi).dropna()
        ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), histtype="step",
                linewidth=1.4, label=name, color=palette[name])
    ax.set(title=col, ylabel="Proportion per bin")
for ax in axes.flat[len(numeric):]:
    ax.remove()
axes.flat[0].legend(fontsize=8)
fig.suptitle("Numeric distributions | shared train bins; tails clipped at 0.1% and 99.9%")
fig.savefig(out / "01_numeric_splits.png", dpi=145)
plt.close(fig)

fig, axes = plt.subplots(4, 3, figsize=(16, 16), constrained_layout=True)
for ax, col in zip(axes.flat, categorical):
    levels = train_all[col].astype(str).value_counts().head(10).index.tolist()
    rows = []
    for name, frame in parts.items():
        counts = frame[col].astype(str).value_counts(normalize=True)
        rows.extend({"value": level, "split": name, "share": counts.get(level, 0)} for level in levels)
    sns.barplot(data=pd.DataFrame(rows), x="share", y="value", hue="split", palette=palette, ax=ax)
    ax.set(title=col, xlabel="Proportion", ylabel="")
    if col != categorical[0]:
        ax.legend_.remove()
for ax in axes.flat[len(categorical):]:
    ax.remove()
fig.suptitle("Categorical and binary distributions | top 10 train values")
fig.savefig(out / "02_categories_splits.png", dpi=145)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
key = ["amount_usd", "account_balance_usd", "velocity_score", "merchant_risk_score",
       "hours_since_last_txn", "distance_from_home_km"]
for ax, col in zip(axes.flat, key):
    frame = parts["Train"]
    lo, hi = frame[col].quantile([.001, .999])
    bins = np.linspace(lo, hi, 45)
    for y, color, name in [(0, "#247ba0", "Legitimate"), (1, "#db5b39", "Fraud")]:
        values = frame.loc[frame.is_fraud == y, col].clip(lo, hi).dropna()
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=f"{name} (n={len(values):,})", color=color)
    ax.set(title=col, ylabel="Density")
    ax.legend(fontsize=7)
fig.suptitle("Training features by target | each class normalized separately")
fig.savefig(out / "03_features_by_target.png", dpi=145)
plt.close(fig)

# Wilson intervals keep small groups visible and avoid assuming symmetry.
risk_rows = []
for col in ["cvv_retry_count", "merchant_category", "channel", "is_new_merchant",
            "is_foreign_transaction", "ip_country_mismatch", "used_vpn"]:
    stats = parts["Train"].groupby(col, observed=True, dropna=False).is_fraud.agg(["size", "sum"])
    for level, row in stats.iterrows():
        n, k = int(row["size"]), int(row["sum"])
        phat, z = k / n, 1.96
        center = (phat + z*z/(2*n))/(1+z*z/n)
        width = z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
        risk_rows.append({"feature": col, "value": str(level), "n": n, "fraud": k,
                          "rate": phat, "lo": center-width, "hi": center+width})
risk = pd.DataFrame(risk_rows)
risk.to_csv(out / "fraud_rates.csv", index=False)
fig, axes = plt.subplots(4, 2, figsize=(15, 16), constrained_layout=True)
for ax, col in zip(axes.flat, risk.feature.unique()):
    d = risk[risk.feature == col].sort_values("rate")
    pos = np.arange(len(d))
    ax.errorbar(d.rate*100, pos, xerr=np.vstack([(d.rate-d.lo)*100, (d.hi-d.rate)*100]), fmt="o", color="#247ba0", capsize=2)
    ax.set_yticks(pos, [f"{v} (n={n:,}, fraud={k:,})" for v, n, k in zip(d.value, d.n, d.fraud)], fontsize=8)
    ax.set(title=col, xlabel="Fraud rate % (95% Wilson CI)")
axes.flat[-1].remove()
fig.suptitle("Fraud rates in the fit partition | group sizes and uncertainty")
fig.savefig(out / "04_fraud_rates.png", dpi=145)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
corr = parts["Train"][numeric].corr(method="spearman")
sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, square=True,
            cbar_kws={"label": "Spearman correlation"})
ax.set_title("Rank correlation between numeric features | training partition")
fig.savefig(out / "05_numeric_correlations.png", dpi=145)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
sns.barplot(data=drift, x="effect", y="feature", hue="comparison",
            order=drift.groupby("feature").effect.max().sort_values(ascending=False).index, ax=ax,
            palette={"Validation": palette["Validation"], "Test": palette["Test"]})
ax.set(title="Distribution difference from training", xlabel="KS for numeric; total variation for categorical", ylabel="")
fig.savefig(out / "06_distribution_effects.png", dpi=145)
plt.close(fig)

errors = None
if args.oof:
    archive = np.load(args.oof, allow_pickle=False)
    assert np.array_equal(archive["id"], train_all.id.to_numpy())
    assert np.array_equal(archive["is_fraud"], train_all.is_fraud.to_numpy())
    assert np.array_equal(archive["fold"][val_idx], np.ones(len(val_idx), dtype=int))
    scores = archive["blend_80_20"][val_idx]
    val = train_all.iloc[val_idx].copy()
    cutoff_count = int(len(val)*.05)
    selected = np.zeros(len(val), dtype=bool)
    selected[np.argpartition(scores, -cutoff_count)[-cutoff_count:]] = True
    val["outcome"] = np.where(val.is_fraud.eq(1) & selected, "Fraud captured",
                       np.where(val.is_fraud.eq(1), "Fraud missed",
                       np.where(selected, "Legitimate flagged", "Legitimate below top 5%")))
    errors = val.groupby("outcome", observed=True).size().rename("rows").reset_index()
    errors.to_csv(out / "validation_outcomes.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fraud = val[val.is_fraud == 1]
    sns.countplot(data=fraud, x="cvv_retry_count", hue="outcome", ax=axes[0],
                  hue_order=["Fraud captured", "Fraud missed"], palette=["#247ba0", "#db5b39"])
    axes[0].set(title="Validation fraud by CVV retries", xlabel="CVV retries", ylabel="Number of fraud rows")
    bars = val[val.outcome.isin(["Fraud captured", "Fraud missed"])].groupby("outcome", observed=True)[
        ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch"]].mean().T
    bars.plot.bar(ax=axes[1], color=["#247ba0", "#db5b39"], rot=25)
    axes[1].set(title="Flag rates among captured and missed fraud", ylabel="Proportion")
    fig.suptitle("V1 OOF scores | validation fold 0 | top 5% flagged")
    fig.savefig(out / "07_validation_errors.png", dpi=145)
    plt.close(fig)

sections = [
    ("Tập dữ liệu", "Validation là fold 0 của StratifiedKFold 5 fold seed 42 trên train.csv. Train là 4 fold còn lại. Test là toàn bộ test.csv, không có nhãn. Validation này đã tham gia các phân tích trước và không là holdout mới độc lập."),
    ("Phân phối biến số", "Histogram dùng cùng cạnh bin của train, mỗi tập được chuẩn hóa theo số dòng. Đuôi ngoài phân vị 0,1%–99,9% được gộp tại hai mép. Xem feature_profile.csv để đọc min, max và các phân vị đầy đủ."),
    ("Biến phân loại", "Biểu đồ thể hiện tỷ lệ theo từng tập. Các bảng feature_profile.csv và distribution_effects.csv chứa toàn bộ biến."),
    ("Phân phối theo nhãn", "Mỗi lớp được chuẩn hóa riêng để so sánh hình dạng. Điều này không thể hiện tỷ lệ fraud thực tế."),
    ("Tỷ lệ fraud theo nhóm", "Điểm là tỷ lệ fraud trên tập train, thanh ngang là khoảng tin cậy Wilson 95%, nhãn kèm số dòng và số fraud."),
    ("Tương quan", "Tương quan Spearman của các biến số trên train. Tương quan không chứng minh quan hệ nhân quả."),
    ("Khác biệt phân phối", "Hiệu ứng KS dùng cho biến số và total variation dùng cho biến phân loại. Không so sánh trực tiếp độ lớn giữa hai thước đo."),
]
images = ["01_numeric_splits.png", "02_categories_splits.png", "03_features_by_target.png",
          "04_fraud_rates.png", "05_numeric_correlations.png", "06_distribution_effects.png"]
header = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>EDA dự đoán gian lận</title>
<style>body{font:16px system-ui;line-height:1.55;color:#202b37;max-width:1300px;margin:3em auto;padding:0 1.5em}
img{width:100%;height:auto} table{border-collapse:collapse}td,th{padding:.45em;border:1px solid #d8dee4;text-align:right}h2{margin-top:2em}</style>
<h1>EDA gian lận thẻ tín dụng</h1><p>Competition Playground S6E11 · Full train 699.960 dòng · StratifiedKFold seed 42</p>"""
body = header + summary.to_html(index=False, na_rep="—")
body += "<p>Tệp thống kê: " + " · ".join(f'<a href="{f}">{f}</a>' for f in
        ["split_summary.csv", "feature_profile.csv", "distribution_effects.csv", "fraud_rates.csv"]) + "</p>"
for (title, desc), image in zip(sections, images):
    body += f"<h2>{html.escape(title)}</h2><p>{html.escape(desc)}</p><img src='{image}' alt='{html.escape(title)}'>"
if errors is not None:
    body += "<h2>Lỗi của v1 trên validation</h2><p>Phân tích top 5% theo dự đoán OOF của v1 trên fold validation, kèm số lượng và tỷ lệ thực tế.</p>"
    body += errors.to_html(index=False) + '<img src="07_validation_errors.png" alt="Validation error analysis">'
body += "</html>"
(out / "report.html").write_text(body, encoding="utf-8")
print(summary.to_string(index=False))
print("Strongest within-type effects")
print(drift.sort_values("effect", ascending=False).head(8).to_string(index=False))
print("Report", out / "report.html")
