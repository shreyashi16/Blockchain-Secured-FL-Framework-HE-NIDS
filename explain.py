"""SHAP explanations for the federated XGBoost ensemble.

Run from the repo root:  pip install shap && python -m xai.explain
Writes outputs/metadata/xai_report.json (loaded by the dashboard).

Global attributions are the sample-weighted average of each client booster's
SHAP values, mirroring the ensemble's FedAvg weights. It's an approximation
(the ensemble averages probabilities; SHAP is in log-odds), which is standard
practice for tree ensembles.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, shap

ROOT = Path(__file__).resolve().parents[1]

def _core(m):
    for a in ("model", "clf", "estimator", "booster_"):   # skip XGBClassifier.booster (a str param)
        v = getattr(m, a, None)
        if v is not None and not isinstance(v, str):
            return v
    return m

def load_bundle(path):
    b = pickle.load(open(path, "rb"))          # ADJUST keys here if your bundle differs
    models = b.get("models") or b.get("client_models") or b.get("boosters")
    w = np.asarray(b.get("weights") or b.get("client_weights"), float)
    return [_core(m) for m in models], w / w.sum()

def shap_3d(ex, X):
    v = ex.shap_values(X)
    v = np.stack(v, -1) if isinstance(v, list) else v
    return v if v.ndim == 3 else v[..., None]     # (samples, features, classes)

def top(names, vals, k):
    return [{"name": names[i], "value": float(vals[i])} for i in np.argsort(-vals)[:k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=ROOT / "outputs/models/global_model.pkl")
    ap.add_argument("--data", default=ROOT / "dataset/X_test.csv")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--flow", type=int, default=0, help="row of the sample to explain locally")
    a = ap.parse_args()

    X = pd.read_csv(a.data).drop(columns=["Label"], errors="ignore")
    X = X.sample(min(a.sample, len(X)), random_state=42).reset_index(drop=True)
    names, models, w = list(X.columns), *load_bundle(a.bundle)

    sv, base = 0, 0
    for m, wc in zip(models, w):
        ex = shap.TreeExplainer(m)
        sv = sv + wc * shap_3d(ex, X)
        base = base + wc * np.atleast_1d(ex.expected_value)
    K = sv.shape[2]
    abs_sv = np.abs(sv)

    pred = int((base + sv[a.flow].sum(0)).argmax())    # logit-average prediction
    contrib = sv[a.flow][:, pred]
    idx = np.argsort(-np.abs(contrib))[:8]

    report = {
        "n_samples": len(X), "classes": K,
        "global": top(names, abs_sv.mean((0, 2)), 15),
        "per_class": {f"Class {k}": top(names, abs_sv[:, :, k].mean(0), 6) for k in range(K)},
        "local": {"flow_index": a.flow, "predicted": f"Class {pred}",
                  "contributions": [{"name": names[i], "feature_value": float(X.iloc[a.flow, i]),
                                     "shap": float(contrib[i])} for i in idx]},
    }
    out = ROOT / "outputs/metadata/xai_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
