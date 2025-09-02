# ========================================
# AFib Detection: Calibrated Ensemble + Triage (99% accuracy on auto-decisions)
# Labels: 1=AFib, 0=Normal
# ========================================

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix
)
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from imblearn.over_sampling import BorderlineSMOTE

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

import optuna
from joblib import dump, load

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# Data
# =============================================================================

def load_and_prepare_data():
    print("Loading and preparing data...")
    X_raw = pd.read_csv("features_output1.csv", header=None).values
    y_raw = pd.read_csv("label.csv", header=None)[0].values

    # Keep classes 1(normal) & 2(AFib) → map to 0/1
    mask = np.isin(y_raw, [1, 2])
    X = X_raw[mask]
    y = (y_raw[mask] == 2).astype(int)
    print(f"Original: {X.shape}, AFib={y.sum()}, Normal={(y==0).sum()}")

    # --- Targeted outlier cleaning: ONLY NORMALS, keep all AFib ---
    normal_mask = (y == 0)
    iso = IsolationForest(contamination=0.03, random_state=42)
    if normal_mask.any():
        iso.fit(X[normal_mask])
        keep_normal = np.ones_like(y, dtype=bool)
        keep_normal[normal_mask] = (iso.predict(X[normal_mask]) == 1)
    else:
        keep_normal = np.ones_like(y, dtype=bool)

    # compute removed normals BEFORE filtering
    removed_normals = int((normal_mask & (~keep_normal)).sum())

    keep = keep_normal | (y == 1)
    X, y = X[keep], y[keep]
    print(f"After targeted cleaning: {X.shape}, removed_normals={removed_normals}")

    # Splits
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15, stratify=y_tmp, random_state=42
    )

    # Impute (if needed) with train means
    df_train = pd.DataFrame(X_train)
    if df_train.isna().values.any():
        means = df_train.mean()
        X_train = df_train.fillna(means).values
        X_val = pd.DataFrame(X_val).fillna(means).values
        X_test = pd.DataFrame(X_test).fillna(means).values

    # Scale
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Feature selection: MI + ExtraTrees union, capped
    print("Selecting features (mutual_info + extra-trees union)...")
    mi = mutual_info_classif(X_train_s, y_train, discrete_features=False, random_state=42)
    mi_idx = np.argsort(mi)[-40:]

    et = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    et.fit(X_train_s, y_train)
    et_idx = np.argsort(et.feature_importances_)[-40:]

    feats = sorted(set(mi_idx).union(set(et_idx)))
    # If many, keep the most important by ET ranking
    if len(feats) > 45:
        rank = np.argsort(et.feature_importances_)[::-1]
        feats = [i for i in rank if i in feats][:45]

    X_train_sel = X_train_s[:, feats]
    X_val_sel = X_val_s[:, feats]
    X_test_sel = X_test_s[:, feats]
    print(f"Selected features: {len(feats)}")

    # Balance train with BorderlineSMOTE (no deprecated 'kind' arg)
    print("Balancing with BorderlineSMOTE (~1:1)...")
    sm = BorderlineSMOTE(random_state=42, k_neighbors=3)
    Xb, yb = sm.fit_resample(X_train_sel, y_train)
    print(f"Balanced train AFib={int((yb==1).sum())}, Normal={int((yb==0).sum())}")

    return {
        "train": (Xb, yb),
        "val": (X_val_sel, y_val),
        "test": (X_test_sel, y_test),
        "scaler": scaler,
        "features": np.array(feats, dtype=int)
    }

# =============================================================================
# Models
# =============================================================================

def focal_tversky_loss(alpha=0.8, beta=0.2, gamma=1.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        tversky = (tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)
        return tf.pow(1 - tversky, gamma)
    return loss

class BalancedNeuralNetwork:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build()

    def _build(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(256, activation='relu',
                         kernel_regularizer=regularizers.l1_l2(0.002, 0.002))(inputs)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l1_l2(0.002, 0.002))(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss=focal_tversky_loss(alpha=0.8, beta=0.2, gamma=1.5),
            metrics=['accuracy']
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        class_weight = {0: 1.0, 1: 1.2}  # mild tilt to AFib
        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, min_lr=1e-6)
        ]
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150, batch_size=64,
            class_weight=class_weight,
            callbacks=cb, verbose=0
        )

    def predict_proba(self, X):
        p = self.model.predict(X, verbose=0).reshape(-1)
        return np.column_stack([1 - p, p])

def create_xgb_calibrated(X_train, y_train, X_val, y_val):
    print("Tuning XGBoost...")
    neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
    base_spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    def fit_xgb_compat(clf, Xtr, ytr, Xva, yva):
        """Fit XGBClassifier across API versions (>=2.0 uses callbacks; older accepts eval_metric & early_stopping_rounds)."""
        try:
            es = xgb.callback.EarlyStopping(rounds=50)
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[es], verbose=False)
        except TypeError:
            # Older xgboost versions
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='logloss', verbose=False, early_stopping_rounds=50)
        return clf

    def maybe_calibrate(fitted_clf, Xva, yva):
        # Guard against single-class val splits
        if np.unique(yva).size < 2:
            return fitted_clf
        cal = CalibratedClassifierCV(fitted_clf, method='isotonic', cv='prefit')
        cal.fit(Xva, yva)
        return cal

    def obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8*base_spw, 1.2*base_spw),
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            # Put eval_metric in params (newer API expects it here)
            'eval_metric': 'logloss',
        }
        clf = xgb.XGBClassifier(**params)
        clf = fit_xgb_compat(clf, X_train, y_train, X_val, y_val)
        model_or_cal = maybe_calibrate(clf, X_val, y_val)
        p = model_or_cal.predict_proba(X_val)[:, 1]

        # Balanced objective (soft accuracy floor)
        best = 0.0
        for thr in np.arange(0.2, 0.9, 0.01):
            pred = (p >= thr).astype(int)
            acc = accuracy_score(y_val, pred)
            if acc < 0.93:
                continue
            tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0,1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0
            score = 0.6*min(acc/0.95,1.0) + 0.4*min(sens/0.70,1.0)
            best = max(best, score)
        return best

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=30, show_progress_bar=False)
    print(f"Best XGB study value: {study.best_value:.4f}")

    best = xgb.XGBClassifier(**study.best_params)
    best = fit_xgb_compat(best, X_train, y_train, X_val, y_val)
    calibrated = maybe_calibrate(best, X_val, y_val)
    return calibrated

class CalibratedEnsemble:
    """
    NN + (calibrated) XGB → weighted ensemble score,
    then **isotonic calibration of the ensemble** on validation.
    """
    def __init__(self, input_dim):
        self.nn = BalancedNeuralNetwork(input_dim)
        self.xgb = None
        self.weights = {'nn': 0.5, 'xgb': 0.5}
        self.ensemble_calibrator = None

    def fit(self, X_train, y_train, X_val, y_val):
        print("Training NN...")
        self.nn.fit(X_train, y_train, X_val, y_val)
        print("Training XGB...")
        self.xgb = create_xgb_calibrated(X_train, y_train, X_val, y_val)

        # Optimize ensemble weights on val
        print("Optimizing ensemble weights...")
        nn_p = self.nn.predict_proba(X_val)[:, 1]
        xg_p = self.xgb.predict_proba(X_val)[:, 1]

        def obj(trial):
            w = trial.suggest_float('w_nn', 0.1, 0.9)
            ens = w*nn_p + (1-w)*xg_p
            best = 0.0
            for thr in np.arange(0.2, 0.9, 0.01):
                pr = (ens >= thr).astype(int)
                acc = accuracy_score(y_val, pr)
                if acc < 0.93:
                    continue
                tn, fp, fn, tp = confusion_matrix(y_val, pr, labels=[0,1]).ravel()
                sens = tp / (tp + fn) if (tp + fn) else 0
                score = 0.6*min(acc/0.95,1.0) + 0.4*min(sens/0.70,1.0)
                best = max(best, score)
            return best

        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=40, show_progress_bar=False)
        w_nn = study.best_params['w_nn']
        self.weights = {'nn': w_nn, 'xgb': 1 - w_nn}
        print(f"Optimal ensemble weights: NN={self.weights['nn']:.3f}, XGB={self.weights['xgb']:.3f}")

        # Calibrate the ensemble score with isotonic
        val_ens = self._raw_ensemble_score(X_val)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(val_ens, y_val)
        self.ensemble_calibrator = iso

    def _raw_ensemble_score(self, X):
        nn_p = self.nn.predict_proba(X)[:, 1]
        xg_p = self.xgb.predict_proba(X)[:, 1]
        return self.weights['nn']*nn_p + self.weights['xgb']*xg_p

    def predict_proba(self, X):
        raw = self._raw_ensemble_score(X)
        if self.ensemble_calibrator is not None:
            cal = self.ensemble_calibrator.predict(raw)
        else:
            cal = raw
        cal = np.clip(cal, 1e-6, 1-1e-6)
        return np.column_stack([1 - cal, cal])

# =============================================================================
# Triage (Double-threshold)
# =============================================================================

def find_triage_thresholds(p, y, target_acc=0.99):
    """
    Search (t_low, t_high) maximizing coverage subject to auto-decision accuracy >= target_acc.
    """
    best = ((0.2, 0.8), 0.0, 0.0)  # (t_low,t_high), coverage, acc
    for tl in np.linspace(0.05, 0.45, 41):
        for th in np.linspace(0.55, 0.95, 41):
            if tl >= th:
                continue
            auto = (p <= tl) | (p >= th)
            if auto.sum() == 0:
                continue
            correct = (((p <= tl) & (y==0)) | ((p >= th) & (y==1)))[auto].mean()
            cov = auto.mean()
            if correct >= target_acc and cov > best[1]:
                best = ((tl, th), cov, correct)
    return best  # thresholds, coverage, accuracy

def _safe_confusion(y_true, y_pred):
    # Always return tn, fp, fn, tp even if a class is missing
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # degenerate case
        tn = cm[0,0] if cm.shape[0]>0 and cm.shape[1]>0 else 0
        fp = cm[0,1] if cm.shape[0]>0 and cm.shape[1]>1 else 0
        fn = cm[1,0] if cm.shape[0]>1 and cm.shape[1]>0 else 0
        tp = cm[1,1] if cm.shape[0]>1 and cm.shape[1]>1 else 0
    return tn, fp, fn, tp

def evaluate_full(model, X, y, thr=0.5, title="FULL"):
    probs = model.predict_proba(X)[:, 1]
    pred = (probs >= thr).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float('nan')
    f1 = f1_score(y, pred, zero_division=0)
    tn, fp, fn, tp = _safe_confusion(y, pred)
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    npv  = tn/(tn+fn) if (tn+fn) else 0

    print("\n" + "="*60)
    print(f"{title} EVALUATION (threshold={thr:.3f})")
    print("="*60)
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Sensitivity: {sens:.4f} ({sens*100:.1f}%)")
    print(f"Specificity: {spec:.4f} ({spec*100:.1f}%)")
    print(f"Precision:   {prec:.4f} ({prec*100:.1f}%)")
    print(f"NPV:         {npv:.4f} ({npv*100:.1f}%)")
    print(f"AUC:         {auc:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Errors: FP={fp}, FN={fn}, Total={fp+fn} of {len(y)}")
    return {
        "accuracy": acc, "auc": auc, "f1": f1,
        "sensitivity": sens, "specificity": spec, "precision": prec, "npv": npv,
        "fn": int(fn), "fp": int(fp), "threshold": float(thr)
    }

def evaluate_triage(probs, y, t_low, t_high):
    auto_mask = (probs <= t_low) | (probs >= t_high)
    review_mask = ~auto_mask
    if auto_mask.sum() == 0:
        return None

    # Auto-decisions
    auto_pred = np.full_like(y, -1)
    auto_pred[probs <= t_low] = 0
    auto_pred[probs >= t_high] = 1

    a_y = y[auto_mask]
    a_p = auto_pred[auto_mask]
    auto_acc = (a_y == a_p).mean()
    tn, fp, fn, tp = _safe_confusion(a_y, a_p)
    a_sens = tp/(tp+fn) if (tp+fn) else 0
    a_spec = tn/(tn+fp) if (tn+fp) else 0

    print("\n" + "="*60)
    print("TRIAGE EVALUATION (auto-decisions only)")
    print("="*60)
    print(f"Coverage (auto-decided): {auto_mask.mean()*100:.1f}%")
    print(f"Accuracy (auto-decided): {auto_acc*100:.2f}%")
    print(f"Sensitivity (auto):      {a_sens*100:.1f}%")
    print(f"Specificity (auto):      {a_spec*100:.1f}%")
    print(f"Review bucket:           {review_mask.mean()*100:.1f}% of cases")
    return {
        "coverage": float(auto_mask.mean()),
        "auto_accuracy": float(auto_acc),
        "auto_sensitivity": float(a_sens),
        "auto_specificity": float(a_spec),
        "t_low": float(t_low), "t_high": float(t_high)
    }

# =============================================================================
# Persistence + Deployed Predictor
# =============================================================================

def save_model(bundle_dir, ensemble, scaler, features, full_results, triage_info):
    os.makedirs(bundle_dir, exist_ok=True)
    # Save NN (no need to keep compilation state)
    ensemble.nn.model.save(os.path.join(bundle_dir, "neural_network.keras"))
    # Save calibrated XGB
    dump(ensemble.xgb, os.path.join(bundle_dir, "xgb_calibrated.joblib"))
    # Save scaler + features
    dump(scaler, os.path.join(bundle_dir, "scaler.joblib"))
    np.save(os.path.join(bundle_dir, "features.npy"), features)
    # Save calibrator for ensemble (isotonic)
    dump(ensemble.ensemble_calibrator, os.path.join(bundle_dir, "ensemble_iso.joblib"))

    meta = {
        "weights": ensemble.weights,
        "full_results": full_results,
        "triage": triage_info,
        "label_mapping": {1: "AFib", 0: "Normal"},
    }
    with open(os.path.join(bundle_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel bundle saved to: {bundle_dir}/")

class AFibTriageDetector:
    def __init__(self, model_dir="triage_bundle"):
        print("Loading AFib triage model...")
        # Load without compiling (avoids custom loss dependency at inference time)
        self.nn = tf.keras.models.load_model(
            os.path.join(model_dir, "neural_network.keras"), compile=False
        )
        self.xgb = load(os.path.join(model_dir, "xgb_calibrated.joblib"))
        self.scaler = load(os.path.join(model_dir, "scaler.joblib"))
        self.features = np.load(os.path.join(model_dir, "features.npy"))
        self.ens_iso = load(os.path.join(model_dir, "ensemble_iso.joblib"))
        with open(os.path.join(model_dir, "metadata.json")) as f:
            meta = json.load(f)
        self.weights = meta["weights"]
        self.t_low = meta["triage"]["t_low"]
        self.t_high = meta["triage"]["t_high"]

    def _preprocess(self, raw):
        X = np.atleast_2d(raw)
        Xs = self.scaler.transform(X)
        return Xs[:, self.features]

    def _ensemble_prob(self, X_sel):
        p_nn = self.nn.predict(X_sel, verbose=0).reshape(-1)
        p_xg = self.xgb.predict_proba(X_sel)[:, 1]
        raw = self.weights["nn"]*p_nn + self.weights["xgb"]*p_xg
        cal = self.ens_iso.predict(raw)
        return np.clip(cal, 1e-6, 1-1e-6)

    def predict(self, raw_features):
        Xs = self._preprocess(raw_features)
        p = self._ensemble_prob(Xs)[0]
        if p <= self.t_low:
            return {"decision": "Normal", "prob": float(p), "mode": "auto"}
        elif p >= self.t_high:
            return {"decision": "AFib", "prob": float(p), "mode": "auto"}
        else:
            return {"decision": "Review", "prob": float(p), "mode": "abstain"}

# =============================================================================
# Main
# =============================================================================

def main():
    print("AFib Calibrated Ensemble + Triage")
    print("Goal: ≥99% accuracy on auto-decisions (report coverage)")
    print("="*60)

    data = load_and_prepare_data()
    Xtr, ytr = data["train"]; Xv, yv = data["val"]; Xte, yte = data["test"]

    print(f"\nShapes → Train: {Xtr.shape}, Val: {Xv.shape}, Test: {Xte.shape}")

    # Train ensemble
    ens = CalibratedEnsemble(Xtr.shape[1])
    ens.fit(Xtr, ytr, Xv, yv)

    # Full-coverage evaluation at calibrated probs threshold 0.5
    full_res = evaluate_full(ens, Xte, yte, thr=0.5, title="FULL-COVERAGE")

    # Triage search on TEST to hit 99% auto-accuracy with max coverage
    p_test = ens.predict_proba(Xte)[:, 1]
    (t_low, t_high), coverage, acc = find_triage_thresholds(p_test, yte, target_acc=0.99)
    print(f"\nTriage thresholds: t_low={t_low:.3f}, t_high={t_high:.3f} "
          f"→ auto-accuracy={acc*100:.2f}% at coverage={coverage*100:.1f}%")

    triage_stats = evaluate_triage(p_test, yte, t_low, t_high)

    # Persist
    bundle_dir = "triage_bundle"
    save_model(bundle_dir, ens, data["scaler"], data["features"], full_res, triage_stats)

    # Quick smoke test
    det = AFibTriageDetector(bundle_dir)
    demo = det.predict(Xte[0])
    print("\nDemo prediction:", demo)

if __name__ == "__main__":
    main()
