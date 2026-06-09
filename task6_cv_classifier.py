"""
task6_cv_classifier.py
----------------------
Classical CV baseline for Task 6 (fluid composition classification).

Uses extracted physical features from feature_table.json to train a simple
classifier that predicts fluid class from visual impact dynamics.

Features used:
  - contact_time_ms      : duration droplet touches surface
  - max_spread_factor    : β_max = D_max / D0 (spreading factor)
  - max_spread_width_mm  : absolute maximum spread width
  - impact_velocity_mm_s : impact velocity (U0)

Classifier: Random Forest + Logistic Regression (compared)
Evaluation: Leave-One-Out cross-validation (small dataset)

Output:
  benchmark/results/cv_classifier_metrics.json
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path("/home/ubuntu/materials")

# ── Load data ─────────────────────────────────────────────────────────────────

def get_fluid_class(video: str, folder: str):
    n = video.lower()
    stem = n.replace(".mp4", "").strip()
    if stem.startswith("scale"):
        return None
    if folder == "02182026":
        if re.match(r"water\d*$", stem):          return "pure_water"
        if stem == "tx":                           return "surfactant_only"
        if stem.startswith("cainh") or stem.startswith("cainl"):
            return "CA_with_surfactant"
        if stem.startswith("caonly"):              return "CA_washed"
    elif folder == "03242026":
        if "only ca" in stem or stem.startswith("caonly"):
            return "CA_washed"
        if stem == "ca+tr":                        return "CA_washed"
        if re.match(r"[\d.]+p", stem) or re.match(r"[\d.]+\s", stem):
            return "surfactant_only"
        if stem.startswith("0.028p"):              return "surfactant_only"
    elif folder == "05052026":
        if stem.startswith("cainh"):               return "CA_with_surfactant"
        if re.match(r"[\d.]+[a-z]", stem):         return "surfactant_only"
    return None


CLASS_TO_INT = {
    "pure_water":         0,
    "surfactant_only":    1,
    "CA_with_surfactant": 2,
    "CA_washed":          3,
}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CHOICE_MAP   = {"pure_water": "A", "surfactant_only": "B",
                "CA_with_surfactant": "C", "CA_washed": "D"}


def load_dataset():
    with open(BASE / "feature_table.json") as f:
        ft = json.load(f)

    rows = []
    for rec in ft:
        folder = rec["folder"]
        video  = rec["video"]
        cls    = get_fluid_class(video, folder)
        if cls is None:
            continue

        row = {
            "video":               video,
            "folder":              folder,
            "fluid_class":         cls,
            "contact_time_ms":     rec.get("contact_time_ms"),
            "max_spread_factor":   rec.get("max_spread_factor"),
            "max_spread_width_mm": rec.get("max_spread_width_mm"),
            "impact_velocity":     rec.get("impact_velocity_mm_per_s"),
            "D0_mm":               rec.get("pre_impact_diameter_mm"),
        }
        rows.append(row)

    return rows


# ── Feature matrix ────────────────────────────────────────────────────────────

FEATURE_SETS = {
    "all_features": [
        "contact_time_ms", "max_spread_factor",
        "max_spread_width_mm", "impact_velocity", "D0_mm",
    ],
    "best_complete": [
        "contact_time_ms", "max_spread_width_mm",
    ],
    "physics_core": [
        "contact_time_ms", "max_spread_factor", "impact_velocity",
    ],
}


def make_matrix(rows, feature_names):
    """Build X, y arrays keeping only rows where all features are present."""
    valid = [r for r in rows
             if all(r.get(f) is not None for f in feature_names)]
    X = np.array([[r[f] for f in feature_names] for r in valid],
                 dtype=np.float32)
    y = np.array([CLASS_TO_INT[r["fluid_class"]] for r in valid])
    labels = [r["fluid_class"] for r in valid]
    return X, y, labels, valid


# ── LOO cross-validation ──────────────────────────────────────────────────────

def loo_evaluate(X, y, labels, clf_name="rf"):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    n = len(y)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros((n, 4))

    for i in range(n):
        tr_idx = [j for j in range(n) if j != i]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[i:i+1]

        if clf_name == "rf":
            clf = RandomForestClassifier(n_estimators=50, random_state=42,
                                         class_weight="balanced")
        elif clf_name == "lr":
            clf = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, C=1.0,
                                          class_weight="balanced")),
            ])

        clf.fit(X_tr, y_tr)
        preds[i] = clf.predict(X_te)[0]
        try:
            probs[i] = clf.predict_proba(X_te)[0]
        except Exception:
            pass

    accuracy = float((preds == y).mean())

    # Per-class accuracy
    per_class = {}
    for c in range(4):
        mask = y == c
        if mask.sum() > 0:
            per_class[INT_TO_CLASS[c]] = round(
                100 * float((preds[mask] == c).mean()), 1)

    # Confusion matrix
    choices = ["A", "B", "C", "D"]
    conf = {gt: {pr: 0 for pr in choices} for gt in choices}
    for true_c, pred_c in zip(y, preds):
        gt_ch  = CHOICE_MAP[INT_TO_CLASS[true_c]]
        pr_ch  = CHOICE_MAP[INT_TO_CLASS[pred_c]]
        conf[gt_ch][pr_ch] += 1

    return {
        "accuracy_pct":       round(100 * accuracy, 1),
        "per_class_accuracy": per_class,
        "confusion_matrix":   conf,
        "n_samples":          n,
        "class_distribution": {INT_TO_CLASS[c]: int((y == c).sum())
                                for c in range(4)},
    }


# ── Feature importance ────────────────────────────────────────────────────────

def feature_importance(X, y, feature_names):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                  class_weight="balanced")
    clf.fit(X, y)
    return {f: round(float(imp), 4)
            for f, imp in zip(feature_names, clf.feature_importances_)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading feature table...")
    rows = load_dataset()
    print(f"  {len(rows)} videos with fluid labels")
    print("  Distribution:", Counter(r["fluid_class"] for r in rows))

    results = {}

    for feat_set_name, feat_names in FEATURE_SETS.items():
        X, y, labels, valid_rows = make_matrix(rows, feat_names)
        print(f"\n── Feature set: {feat_set_name}  ({len(feat_names)} features, {len(y)} samples) ──")
        print(f"   Features: {feat_names}")
        print(f"   Class dist: {Counter(labels)}")

        if len(y) < 10:
            print("  [skip] too few samples")
            continue

        set_results = {"features": feat_names, "n_samples": int(len(y))}

        for clf_name in ["rf", "lr"]:
            res = loo_evaluate(X, y, labels, clf_name)
            set_results[clf_name] = res
            print(f"   {clf_name.upper():2s} LOO accuracy: {res['accuracy_pct']}%"
                  f"  per-class: {res['per_class_accuracy']}")

        # Feature importance from RF on full data
        imp = feature_importance(X, y, feat_names)
        set_results["feature_importance"] = imp
        print(f"   Importance: {imp}")

        results[feat_set_name] = set_results

    # Best result summary
    print(f"\n{'='*60}")
    print("Classical CV Task 6 Classifier — Summary")
    print(f"{'='*60}")
    print(f"{'Feature set':<20} {'Model':<4} {'LOO Acc':>8}  Per-class")
    print("-" * 70)
    for fs, res in results.items():
        for clf_name in ["rf", "lr"]:
            if clf_name in res:
                r = res[clf_name]
                print(f"{fs:<20} {clf_name.upper():<4} {r['accuracy_pct']:>7}%  {r['per_class_accuracy']}")

    print(f"\nVLM zero-shot baselines (for comparison):")
    print(f"  Gemini 2.0 Flash    : 21.3%  (below 25% chance)")
    print(f"  GPT-4o              : 10.6%  (below 25% chance)")
    print(f"  Claude Sonnet 4.5   : 18.1%  (below 25% chance)")
    print(f"  Chance baseline     : 25.0%")

    # Save
    out = {
        "description": "Classical CV Task 6 classifier using LOO cross-validation",
        "evaluation": "leave-one-out",
        "feature_sets": results,
        "vlm_comparison": {
            "gemini_2_0_flash": 21.3,
            "gpt_4o_mini":      11.7,
            "gpt_4o":           10.6,
            "claude_sonnet_45": 18.1,
            "chance_baseline":  25.0,
        },
    }
    out_path = BASE / "benchmark/results/cv_classifier_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
