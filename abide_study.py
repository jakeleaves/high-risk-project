"""Leave-one-site-out autism classification on ABIDE I.

The pre-specified question is whether a resting-state connectivity
classifier trained on other hospitals still separates autism from
control at a held-out acquisition site. A pooled cross-validation
score is saved only as the optimistic comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nilearn_data"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "report" / "figures"
CACHE = DATA_DIR / "features.npz"
SEED = 22
N_COMPONENTS = 100
MIN_TIMEPOINTS = 50
N_REGIONS = 200
MIN_SITE_N = 15


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    features = load_or_build_features()
    summary, per_site = evaluate(features)
    summary_path = RESULT_DIR / "summary.json"
    per_site_path = RESULT_DIR / "per_site.csv"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2))
    per_site.to_csv(per_site_path, index=False)
    make_figures(per_site, summary)
    print(json.dumps(jsonable(summary), indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {per_site_path}")


def load_or_build_features() -> dict[str, np.ndarray]:
    if CACHE.exists():
        print(f"loading cached features from {CACHE}")
        cached = np.load(CACHE, allow_pickle=True)
        return {key: cached[key] for key in cached.files}
    bundle = fetch_cohort()
    features = build_features(bundle)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **features)
    print(f"cached features at {CACHE}")
    return features


def fetch_cohort():
    from nilearn.datasets import fetch_abide_pcp

    print("downloading ABIDE I CPAC Craddock-200 time series")
    # Band-pass 0.01-0.1 Hz, no global signal regression.
    # QC keeps subjects who passed every PCP rater.
    return fetch_abide_pcp(
        data_dir=str(DATA_DIR),
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        quality_checked=True,
        verbose=1,
    )


def build_features(bundle) -> dict[str, np.ndarray]:
    pheno = bundle.phenotypic.copy().reset_index(drop=True)
    series_list = list(bundle.rois_cc200)
    if len(pheno) != len(series_list):
        raise RuntimeError(
            f"phenotype rows ({len(pheno)}) and time series ({len(series_list)}) differ"
        )
    dx = pd.to_numeric(pheno["DX_GROUP"], errors="coerce")
    if not set(dx.dropna().unique()).issubset({1, 2}):
        raise RuntimeError(f"unexpected DX_GROUP values: {sorted(dx.dropna().unique())}")

    rows = []
    kept_x = []
    dropped = {"missing_series": 0, "bad_shape": 0, "short_scan": 0, "bad_dx": 0}
    for i, raw in enumerate(series_list):
        if raw is None:
            dropped["missing_series"] += 1
            continue
        if dx.iloc[i] not in (1, 2):
            dropped["bad_dx"] += 1
            continue
        series = np.asarray(raw, dtype=float)
        if series.ndim != 2:
            dropped["bad_shape"] += 1
            continue
        if series.shape[1] != N_REGIONS and series.shape[0] == N_REGIONS:
            series = series.T
        if series.shape[0] < MIN_TIMEPOINTS or series.shape[1] != N_REGIONS:
            dropped["short_scan"] += 1
            continue
        vector = connectivity_vector(series)
        if vector is None:
            dropped["bad_shape"] += 1
            continue
        row = pheno.iloc[i]
        age = pd.to_numeric(row["AGE_AT_SCAN"], errors="coerce")
        sex = pd.to_numeric(row["SEX"], errors="coerce")
        kept_x.append(vector)
        rows.append(
            {
                "sub_id": str(row["SUB_ID"]),
                "site": str(row["SITE_ID"]),
                "autism": int(dx.iloc[i] == 1),
                "age": float(age) if pd.notna(age) and age > 0 else np.nan,
                "sex": float(sex) if pd.notna(sex) and sex in (1, 2) else np.nan,
            }
        )
        if len(rows) % 100 == 0:
            print(f"featurized {len(rows)} subjects")

    meta = pd.DataFrame(rows)
    print(f"kept {len(meta)} subjects; dropped {dropped}")
    if meta["site"].nunique() < 2 or meta["autism"].nunique() < 2:
        raise RuntimeError("feature table does not have two sites and two classes")
    return {
        "X": np.vstack(kept_x).astype(np.float32),
        "y": meta["autism"].to_numpy(dtype=np.int8),
        "site": meta["site"].to_numpy(dtype=str),
        "sub_id": meta["sub_id"].to_numpy(dtype=str),
        "age": meta["age"].to_numpy(dtype=np.float32),
        "sex": meta["sex"].to_numpy(dtype=np.float32),
        "dropped_json": np.array(json.dumps(dropped)),
    }


def connectivity_vector(series: np.ndarray) -> np.ndarray | None:
    # A flat region makes the correlation undefined. Drop it rather than invent a value.
    if np.any(np.nanstd(series, axis=0) == 0):
        return None
    corr = np.corrcoef(series, rowvar=False)
    if not np.isfinite(corr).all():
        return None
    corr = np.clip(corr, -0.999, 0.999)
    z = np.arctanh(corr)
    idx = np.triu_indices(N_REGIONS, k=1)
    return z[idx].astype(np.float32)


def make_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=N_COMPONENTS,
                    svd_solver="randomized",
                    random_state=SEED,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_predict(pipe: Pipeline, x_train, y_train, x_test):
    n_comp = min(N_COMPONENTS, x_train.shape[0] - 1, x_train.shape[1])
    pipe.set_params(pca__n_components=max(1, n_comp))
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    proba = pipe.predict_proba(x_test)
    classes = list(pipe.named_steps["clf"].classes_)
    if proba.shape[1] == 2 and 1 in classes:
        positive = proba[:, classes.index(1)]
    else:
        positive = np.full(len(x_test), np.nan)
    return pred, positive


def score_split(y_true, pred, proba) -> dict[str, float]:
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auroc": np.nan,
        "auprc": np.nan,
        "brier": np.nan,
    }
    if np.isfinite(proba).all():
        out["brier"] = float(brier_score_loss(y_true, proba))
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, proba))
        out["auprc"] = float(average_precision_score(y_true, proba))
    return out


def evaluate(features: dict[str, np.ndarray]):
    x = features["X"]
    y = features["y"].astype(int)
    site = features["site"].astype(str)
    age = features["age"].astype(float)
    sex = features["sex"].astype(float)
    sites = sorted(pd.unique(site))
    print(f"evaluating {len(y)} subjects across {len(sites)} sites")

    per_site_rows = []
    loso_y = []
    loso_pred = []
    loso_proba = []
    age_y = []
    age_pred = []
    age_proba = []

    for held_out in sites:
        test = site == held_out
        train = ~test
        n = int(test.sum())
        n_autism = int(y[test].sum())
        row = {
            "site": held_out,
            "n": n,
            "n_autism": n_autism,
            "n_control": n - n_autism,
            "prevalence": n_autism / n,
            "small_site": n < MIN_SITE_N,
        }
        if y[train].min() == y[train].max() or n == 0:
            row.update(empty_scores("connectivity"))
            row.update(empty_scores("age_sex"))
            per_site_rows.append(row)
            continue
        pred, proba = fit_predict(make_pipe(), x[train], y[train], x[test])
        scores = score_split(y[test], pred, proba)
        row.update({f"connectivity_{key}": value for key, value in scores.items()})
        loso_y.append(y[test])
        loso_pred.append(pred)
        loso_proba.append(proba)

        age_ok = np.isfinite(age) & np.isfinite(sex)
        if (
            age_ok[train].sum() > 10
            and len(np.unique(y[train & age_ok])) > 1
            and age_ok[test].any()
        ):
            x_age = np.column_stack([age, sex])
            age_pred_s, age_proba_s = fit_age_sex(
                x_age[train & age_ok],
                y[train & age_ok],
                x_age[test & age_ok],
            )
            age_scores = score_split(y[test & age_ok], age_pred_s, age_proba_s)
            row.update({f"age_sex_{key}": value for key, value in age_scores.items()})
            age_y.append(y[test & age_ok])
            age_pred.append(age_pred_s)
            age_proba.append(age_proba_s)
        else:
            row.update(empty_scores("age_sex"))
        per_site_rows.append(row)
        print(
            f"{held_out:12} n={n:3}  ba={row['connectivity_balanced_accuracy']:.3f}"
        )

    per_site = pd.DataFrame(per_site_rows).sort_values("n", ascending=False)
    pooled = pooled_cv(x, y, seed=SEED)
    shuffled = pooled_cv(x, y, seed=SEED, shuffle_labels=True)
    site_accuracy, site_baseline, site_balanced = site_classifier(x, site)
    motion = motion_loso(features["sub_id"].astype(str), y, site)
    loso_y_all = np.concatenate(loso_y)
    loso_pred_all = np.concatenate(loso_pred)
    loso_proba_all = np.concatenate(loso_proba)
    pooled_loso = score_split(loso_y_all, loso_pred_all, loso_proba_all)
    usable = per_site.loc[~per_site["small_site"]]
    macro_ba = float(usable["connectivity_balanced_accuracy"].mean())
    macro_auroc = float(usable["connectivity_auroc"].mean())
    macro_brier = float(usable["connectivity_brier"].mean())
    age_macro_ba = float(usable["age_sex_balanced_accuracy"].mean())
    rho_n, p_n = spearmanr(
        usable["n"],
        usable["connectivity_balanced_accuracy"],
        nan_policy="omit",
    )
    rho_prev, p_prev = spearmanr(
        usable["prevalence"],
        usable["connectivity_balanced_accuracy"],
        nan_policy="omit",
    )

    summary = {
        "n_subjects": int(len(y)),
        "n_autism": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "n_sites": int(len(sites)),
        "n_sites_reported": int((~per_site["small_site"]).sum()),
        "min_site_n": MIN_SITE_N,
        "pca_components": N_COMPONENTS,
        "pipeline": "cpac",
        "strategy": "filt_noglobal",
        "atlas": "cc200",
        "quality_checked": True,
        "dropped": json.loads(str(features["dropped_json"])),
        "connectivity_loso_macro_balanced_accuracy": macro_ba,
        "connectivity_loso_macro_auroc": macro_auroc,
        "connectivity_loso_macro_brier": macro_brier,
        "connectivity_loso_pooled_balanced_accuracy": pooled_loso["balanced_accuracy"],
        "connectivity_loso_pooled_auroc": pooled_loso["auroc"],
        "connectivity_loso_pooled_auprc": pooled_loso["auprc"],
        "connectivity_loso_pooled_brier": pooled_loso["brier"],
        "age_sex_loso_macro_balanced_accuracy": age_macro_ba,
        "pooled_cv_balanced_accuracy_mean": pooled["balanced_accuracy_mean"],
        "pooled_cv_balanced_accuracy_std": pooled["balanced_accuracy_std"],
        "pooled_cv_auroc_mean": pooled["auroc_mean"],
        "pooled_cv_auroc_std": pooled["auroc_std"],
        "shuffled_pooled_cv_auroc_mean": shuffled["auroc_mean"],
        "site_classifier_accuracy": site_accuracy,
        "site_classifier_balanced_accuracy": site_balanced,
        "site_majority_baseline_accuracy": site_baseline,
        "motion_loso_macro_balanced_accuracy": motion["balanced_accuracy"],
        "motion_loso_macro_auroc": motion["auroc"],
        "motion_median_fd_control": motion["median_fd_control"],
        "motion_median_fd_autism": motion["median_fd_autism"],
        "spearman_loso_ba_vs_n": float(rho_n),
        "spearman_loso_ba_vs_n_p": float(p_n),
        "spearman_loso_ba_vs_prevalence": float(rho_prev),
        "spearman_loso_ba_vs_prevalence_p": float(p_prev),
        "gap_pooled_minus_loso_ba": float(
            pooled["balanced_accuracy_mean"] - macro_ba
        ),
    }
    if age_y:
        age_scores = score_split(
            np.concatenate(age_y),
            np.concatenate(age_pred),
            np.concatenate(age_proba),
        )
        summary["age_sex_loso_pooled_balanced_accuracy"] = age_scores["balanced_accuracy"]
        summary["age_sex_loso_pooled_auroc"] = age_scores["auroc"]
    return summary, per_site


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def empty_scores(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_balanced_accuracy": np.nan,
        f"{prefix}_auroc": np.nan,
        f"{prefix}_auprc": np.nan,
        f"{prefix}_brier": np.nan,
    }


def fit_age_sex(x_train, y_train, x_test):
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def pooled_cv(x, y, seed: int, shuffle_labels: bool = False) -> dict[str, float]:
    labels = y.copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        labels = rng.permutation(labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    bas, aucs = [], []
    for train, test in folds.split(x, labels):
        pred, proba = fit_predict(make_pipe(), x[train], labels[train], x[test])
        scored = score_split(labels[test], pred, proba)
        bas.append(scored["balanced_accuracy"])
        aucs.append(scored["auroc"])
    return {
        "balanced_accuracy_mean": float(np.mean(bas)),
        "balanced_accuracy_std": float(np.std(bas)),
        "auroc_mean": float(np.nanmean(aucs)),
        "auroc_std": float(np.nanstd(aucs)),
    }


def motion_loso(sub_id: np.ndarray, y: np.ndarray, site: np.ndarray) -> dict[str, float]:
    """Leave-one-site-out using only mean framewise displacement."""
    path = DATA_DIR / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    empty = {
        "balanced_accuracy": np.nan,
        "auroc": np.nan,
        "median_fd_control": np.nan,
        "median_fd_autism": np.nan,
    }
    if not path.exists():
        return empty
    pheno = pd.read_csv(path)
    fd_map = pd.to_numeric(
        pheno.set_index(pheno["SUB_ID"].astype(str))["func_mean_fd"],
        errors="coerce",
    )
    fd = np.array([fd_map.get(sub, np.nan) for sub in sub_id], dtype=float)
    fd = np.where(fd < 0, np.nan, fd)
    empty["median_fd_control"] = float(np.nanmedian(fd[y == 0]))
    empty["median_fd_autism"] = float(np.nanmedian(fd[y == 1]))
    scores_ba, scores_auc = [], []
    for held_out in sorted(pd.unique(site)):
        test = site == held_out
        train = ~test
        if int(test.sum()) < MIN_SITE_N or len(np.unique(y[train])) < 2:
            continue
        if not np.isfinite(fd[train]).all() or not np.isfinite(fd[test]).all():
            continue
        pred, proba = fit_age_sex(fd[train].reshape(-1, 1), y[train], fd[test].reshape(-1, 1))
        scored = score_split(y[test], pred, proba)
        scores_ba.append(scored["balanced_accuracy"])
        scores_auc.append(scored["auroc"])
    if scores_ba:
        empty["balanced_accuracy"] = float(np.mean(scores_ba))
        empty["auroc"] = float(np.nanmean(scores_auc))
    return empty


def site_classifier(x, site) -> tuple[float, float, float]:
    values, counts = np.unique(site, return_counts=True)
    keep_sites = set(values[counts >= 5])
    mask = np.array([item in keep_sites for item in site])
    x_use = x[mask]
    site_use = site[mask]
    majority = float(counts.max() / counts[counts >= 5].sum())
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, bals = [], []
    for train, test in folds.split(x_use, site_use):
        pred, _ = fit_predict(make_pipe(), x_use[train], site_use[train], x_use[test])
        accs.append(float(np.mean(pred == site_use[test])))
        bals.append(float(balanced_accuracy_score(site_use[test], pred)))
    return float(np.mean(accs)), majority, float(np.mean(bals))


def make_figures(per_site: pd.DataFrame, summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    draw_workflow()
    draw_cohort(per_site)
    draw_loso(per_site, summary)
    draw_comparison(summary)


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, "ABIDE I\nCPAC, QC passed"),
        (2.15, "Craddock-200\ntime series"),
        (4.1, "Fisher-z\nconnectivity"),
        (6.05, "Leave one\nsite out"),
        (8.0, "Balanced accuracy\nand AUROC"),
    ]
    for x, text in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, 0.85),
                1.8,
                1.3,
                facecolor="#E6F1F8",
                edgecolor="#145C8A",
                linewidth=1.2,
            )
        )
        ax.text(x + 0.9, 1.5, text, ha="center", va="center", fontsize=8.5)
    for x in (2.0, 3.95, 5.9, 7.85):
        ax.annotate(
            "",
            xy=(x + 0.15, 1.5),
            xytext=(x - 0.05, 1.5),
            arrowprops={"arrowstyle": "->", "color": "#145C8A", "lw": 1.2},
        )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "workflow.png", bbox_inches="tight")
    fig.savefig(RESULT_DIR / "workflow.png", bbox_inches="tight")
    plt.close(fig)


def draw_cohort(per_site: pd.DataFrame) -> None:
    plot_df = per_site.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["site"], plot_df["n_control"], color="#0072B2", label="Control")
    ax.barh(
        plot_df["site"],
        plot_df["n_autism"],
        left=plot_df["n_control"],
        color="#D55E00",
        label="Autism",
    )
    ax.set_xlabel("Subjects passing QC with a usable time series")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cohort_by_site.png", bbox_inches="tight")
    fig.savefig(RESULT_DIR / "cohort_by_site.png", bbox_inches="tight")
    plt.close(fig)


def draw_loso(per_site: pd.DataFrame, summary: dict) -> None:
    plot_df = per_site.sort_values("connectivity_balanced_accuracy", ascending=True)
    colors = ["#999999" if flag else "#145C8A" for flag in plot_df["small_site"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["site"], plot_df["connectivity_balanced_accuracy"], color=colors)
    ax.axvline(0.5, color="#333333", lw=1, ls="--", label="Chance")
    ax.axvline(
        summary["pooled_cv_balanced_accuracy_mean"],
        color="#D55E00",
        lw=1.2,
        label="Pooled 5-fold CV",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Leave-one-site-out balanced accuracy")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "loso_balanced_accuracy.png", bbox_inches="tight")
    fig.savefig(RESULT_DIR / "loso_balanced_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def draw_comparison(summary: dict) -> None:
    labels = [
        "Connectivity\nleave-one-site-out",
        "Head motion\nleave-one-site-out",
        "Age and sex\nleave-one-site-out",
        "Connectivity\npooled CV",
    ]
    values = [
        summary["connectivity_loso_macro_balanced_accuracy"],
        summary["motion_loso_macro_balanced_accuracy"],
        summary["age_sex_loso_macro_balanced_accuracy"],
        summary["pooled_cv_balanced_accuracy_mean"],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color=["#145C8A", "#E69F00", "#009E73", "#D55E00"])
    ax.axhline(0.5, color="#333333", lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro balanced accuracy")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", bbox_inches="tight")
    fig.savefig(RESULT_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
