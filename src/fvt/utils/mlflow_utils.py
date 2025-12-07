"""
mlflow_utils.py — VERSION FINALE (safe_start_run + vectorisation restaurée)
----------------------------------------------------------------------------

Cette version gère :
- SKLEARN baseline + tuning nested (tuned=true/false)
- KERAS baseline + tuning nested
- vectorisation automatique (TF-IDF) comme avant
- compatibilité totale avec NB01 (où X_train/X_test sont du texte brut)
- sécurité anti-run-actif via safe_start_run()
- tagging et métriques cohérents
"""

import os
import pickle
import mlflow
import pandas as pd
from mlflow.models import infer_signature, Model
from mlflow.models.signature import ModelSignature
from mlflow.exceptions import MlflowException
try:
    from mlflow.tracking import MlflowClient  # facultatif pour tagger après coup
except Exception:  # pragma: no cover - environnement sans mlflow complet
    MlflowClient = None
from mlflow.types.schema import Schema, TensorSpec
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# SAFE START RUN
# ---------------------------------------------------------------------------

def _safe_start_run(run_name=None, **kwargs):
    """
    Démarre un run MLflow de manière robuste :
    - si un run est actif → nested=True automatiquement
    - sinon → run normal
    """
    active = mlflow.active_run()
    nested = active is not None
    kwargs = {k: v for k, v in kwargs.items() if k != "nested"}
    return mlflow.start_run(run_name=run_name, nested=nested, **kwargs)


# ---------------------------------------------------------------------------
# UTILS MLflow
# ---------------------------------------------------------------------------

def save_item_to_new_run(item, filename, run_id):
    local_path = f"/tmp/{filename}"
    with open(local_path, "wb") as f:
        pickle.dump(item, f)
    mlflow.log_artifact(local_path, artifact_path=f"run_{run_id}")


def load_item_from_run(run_id, filename):
    # Essaye d'abord à la racine, puis dans le sous-dossier run_<id> (utilisé par save_item_to_new_run)
    artifact_candidates = [
        filename,
        f"run_{run_id}/{filename}",
    ]

    local_path = None
    last_exc = None
    for artifact_path in artifact_candidates:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact_path
            )
            break
        except Exception as exc:  # fallback si le fichier n'est pas à cet emplacement
            last_exc = exc

    if local_path is None:
        raise RuntimeError(
            f"Impossible de télécharger l'artefact {filename} pour le run {run_id}"
        ) from last_exc

    # Charger selon l'extension
    if filename.endswith(".pkl"):
        with open(local_path, "rb") as f:
            return pickle.load(f)

    elif filename.endswith(".npy"):
        return np.load(local_path, allow_pickle=False)

    else:
        raise ValueError(f"Extension non supportée pour : {filename}")


def run_exists(run_id):
    try:
        mlflow.get_run(run_id)
        return True
    except Exception:
        return False


def save_embeddings_and_vocab_to_new_run(embeddings, vocab, run_id):
    save_item_to_new_run(embeddings, "embeddings.pkl", run_id)
    save_item_to_new_run(vocab, "vocab.pkl", run_id)


# ---------------------------------------------------------------------------
# F1 THRESHOLD OPTIMISATION
# ---------------------------------------------------------------------------

def optimize_threshold_f1(y_true, y_proba):
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.05, 0.95, 40):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_pr_auc(y_true, y_proba):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    return auc(r, p)


# ---------------------------------------------------------------------------
# SKLEARN : BASELINE + TUNING
# ---------------------------------------------------------------------------

def _vectorize_data(vectorizer, classifier, X_train, X_test):
    """Retourne les versions vectorisées pour baseline."""
    if vectorizer is None:
        return X_train, X_test

    # Si classifier est un pipeline → il gère déjà le TF-IDF
    if isinstance(classifier, Pipeline):
        return X_train, X_test

    # Sinon TF-IDF externe
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def _vectorize_for_tuning(vectorizer, classifier, X_train, X_test):
    """
    Pour le tuning : on NE REFIT PAS le vectorizer !
    On réutilise celui du baseline.
    """
    if vectorizer is None:
        return X_train, X_test

    if isinstance(classifier, Pipeline):
        return X_train, X_test

    return vectorizer.transform(X_train), vectorizer.transform(X_test)


def _log_sklearn_params(prefix, estimator):
    """Logue les params simples d'un estimator sklearn (évite les objets complexes)."""
    if estimator is None or not hasattr(estimator, "get_params"):
        return
    for k, v in estimator.get_params().items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            try:
                mlflow.log_param(f"{prefix}{k}", v)
            except Exception:
                pass


def _log_dataset_input(dataset_tag, split_run_id=None, n_train=None, n_test=None, context="train"):
    """Logue un mini-dataset de métadonnées dans MLflow (colonne Data/Inputs)."""
    if not dataset_tag:
        return
    # mlflow.data n'existe que sur MLflow >= 2.4
    if not hasattr(mlflow, "data"):
        return
    meta = {"dataset": dataset_tag}
    if split_run_id:
        meta["split_run_id"] = split_run_id
    if n_train is not None:
        meta["n_train"] = int(n_train)
    if n_test is not None:
        meta["n_test"] = int(n_test)
    try:
        ds = mlflow.data.from_pandas(pd.DataFrame([meta]), source=str(dataset_tag))
        mlflow.log_input(ds, context=context)
    except Exception:
        # On ne bloque pas le run si le log_input échoue
        pass


def load_split_from_mlflow(
    experiment_name,
    *,
    dataset_name=None,
    artifact_prefix="clean_text_split",
    artifact_name=None,
    run_name_prefix="tweets_split",
):
    """
    Retrouve et charge un split train/test stocké dans MLflow.
    - dataset_name : nom logique (ex: "clean_tweets_201k"). Si None, essaie DATASET_NAME env.
    - artifact_name : nom exact du fichier pkl (optionnel). Si absent, construit à partir de dataset_name
      avec artifact_prefix, ex: clean_text_split_201k.pkl.
    - run_name_prefix : filtre supplémentaire sur le runName (ex: "tweets_split").
    Retourne le dict du split et le run_id sélectionné.
    """
    ds = dataset_name or os.getenv("DATASET_NAME")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Expérience introuvable : {experiment_name}")

    runs = mlflow.search_runs([exp.experiment_id], max_results=5000)
    if runs.empty:
        raise RuntimeError("Aucun run trouvé dans cette expérimentation.")

    run_names = runs["tags.mlflow.runName"].fillna("")
    mask = False
    if ds:
        mask |= runs["params.dataset"].fillna("") == ds
        mask |= run_names.str.contains(ds)
    if run_name_prefix:
        mask |= run_names.str.startswith(run_name_prefix)

    filtered = runs[mask].sort_values(by="start_time", ascending=False)
    if filtered.empty:
        raise RuntimeError(f"Aucun run de split trouvé pour dataset={ds or 'unknown'}")

    run_id = filtered.iloc[0]["run_id"]

    if artifact_name is None:
        suffix = ""
        if ds and "_" in ds:
            suffix = ds.split("_")[-1]  # ex: 201k
        artifact_name = f"{artifact_prefix}_{suffix}.pkl" if suffix else f"{artifact_prefix}.pkl"

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
    with open(local_path, "rb") as f:
        split_dict = pickle.load(f)

    return split_dict, run_id


def find_use_embeddings_in_mlflow(
    experiment_name,
    *,
    embedding_name="USE",
    dataset_name=None,
    artifact_name=None,
    run_name_prefix=None,
):
    """Retrouve des embeddings USE déjà logués dans MLflow et les télécharge.

    - embedding_name : nom logique (ex: "USE").
    - dataset_name   : dataset associé (ex: "clean_tweets_201k"). Si None, essaie DATASET_NAME env.
    - artifact_name  : nom exact du pkl. Si None, construit par défaut (use_embeddings_<suffix>.pkl).
    - run_name_prefix: filtre optionnel sur le runName.
    Retourne le chemin local de l'artefact ou None si non trouvé.
    """
    ds = dataset_name or os.getenv("DATASET_NAME")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    runs = mlflow.search_runs([exp.experiment_id], max_results=5000)
    if runs.empty:
        return None

    run_names = runs["tags.mlflow.runName"].fillna("")
    embed_col = runs["params.embedding"] if "params.embedding" in runs else pd.Series("", index=runs.index)
    mask = (embed_col.fillna("") == embedding_name)
    if ds:
        dataset_col = runs["params.dataset"] if "params.dataset" in runs else pd.Series("", index=runs.index)
        mask &= dataset_col.fillna("") == ds
        mask |= run_names.str.contains(ds)
    if run_name_prefix:
        mask |= run_names.str.startswith(run_name_prefix)

    filtered = runs[mask].sort_values(by="start_time", ascending=False)
    if filtered.empty:
        return None

    run_id = filtered.iloc[0]["run_id"]

    if artifact_name is None:
        suffix = ""
        if ds and "_" in ds:
            suffix = ds.split("_")[-1]
        artifact_name = f"use_embeddings_{suffix}.pkl" if suffix else "use_embeddings.pkl"

    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
    except Exception:
        return None


def _log_sklearn_params(prefix, estimator):
    """Logue les params simples d'un estimator sklearn (évite les objets complexes)."""
    if estimator is None or not hasattr(estimator, "get_params"):
        return
    for k, v in estimator.get_params().items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            try:
                mlflow.log_param(f"{prefix}{k}", v)
            except Exception:
                pass


def train_and_log_classifier(
    model_name,
    X_train, X_test,
    y_train, y_test,
    vectorizer=None,
    classifier=None,
    params=None,
    register_model=None,
    do_tuning=False,
    tuner=None,
    dataset_name=None,
    split_run_id=None,
):
    """
    Entraîne un modèle sklearn + log MLflow.
    Gère la vectorisation automatique comme dans l’ancienne version.
    """

    # ---------------------------------------------------
    # BASELINE
    # ---------------------------------------------------
    with _safe_start_run(run_name=model_name):
        mlflow.set_tag("tuned", "false")

        # Dataset : tag + param si fourni
        dataset_tag = dataset_name or (params or {}).get("dataset") or (params or {}).get("dataset_name") or os.getenv("DATASET_NAME")
        split_ref = split_run_id or (params or {}).get("split_run_id") or os.getenv("SPLIT_RUN_ID")
        if dataset_tag:
            mlflow.set_tag("dataset", dataset_tag)
            mlflow.log_param("dataset", dataset_tag)

        # Params explicites + params des objets sklearn (vectorizer / classifier)
        if params:
            mlflow.log_params(params)
        _log_sklearn_params("vec_", vectorizer)
        _log_sklearn_params("clf_", classifier)

        # Log input dataset (métadonnées uniquement)
        _log_dataset_input(dataset_tag, split_run_id=split_ref, n_train=len(X_train) if X_train is not None else None, n_test=len(X_test) if X_test is not None else None)

        # Vectorisation baseline
        X_train_vec, X_test_vec = _vectorize_data(vectorizer, classifier, X_train, X_test)

        classifier.fit(X_train_vec, y_train)

        try:
            y_proba = classifier.predict_proba(X_test_vec)[:, 1]
        except Exception:
            raw = classifier.decision_function(X_test_vec)
            y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

        best_t = optimize_threshold_f1(y_test, y_proba)
        y_pred = (y_proba >= best_t).astype(int)

        metrics_base = {
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": compute_pr_auc(y_test, y_proba),
            "best_threshold": best_t,
        }

        mlflow.log_metrics(metrics_base)

        # Log du modèle : pour TF-IDF on logue un pipeline complet (vectorizer + classifier)
        # de façon à accepter du texte brut en prédiction.
        if vectorizer is not None and not isinstance(classifier, Pipeline):
            model_to_log = Pipeline([("vec", vectorizer), ("clf", classifier)])
            sample_df = pd.DataFrame({"text": X_train[:1]})
            sample_out = model_to_log.predict_proba(sample_df["text"])[:, 1] if hasattr(model_to_log, "predict_proba") else model_to_log.decision_function(sample_df["text"])
        else:
            model_to_log = classifier
            sample_in = X_train_vec[:1]
            if hasattr(sample_in, "toarray"):
                sample_in = sample_in.toarray()
            sample_df = pd.DataFrame(sample_in)
            sample_out = model_to_log.predict_proba(sample_in)[:, 1] if hasattr(model_to_log, "predict_proba") else model_to_log.decision_function(sample_in)

        signature = infer_signature(sample_df, sample_out)

        try:
            mlflow.sklearn.log_model(
                model_to_log,
                name="model",
                signature=signature,
                input_example=sample_df,
            )
        except Exception:
            # fallback sans input_example si la sérialisation échoue
            mlflow.sklearn.log_model(
                model_to_log,
                name="model",
                signature=signature,
            )

        tuned_model = None
        metrics_tuned = None

        # ---------------------------------------------------
        # TUNING nested
        # ---------------------------------------------------
        if do_tuning and tuner and tuner.get("type") == "sklearn":

            grid = list(ParameterGrid(tuner["params"]))
            best_f1_global = -1

            X_train_tune, X_test_tune = _vectorize_for_tuning(
                vectorizer, classifier, X_train, X_test
            )

            for cfg in grid:

                with _safe_start_run(run_name=f"{model_name}_tuned"):
                    mlflow.set_tag("tuned", "true")
                    if dataset_tag:
                        mlflow.set_tag("dataset", dataset_tag)
                        mlflow.log_param("dataset", dataset_tag)

                    clf = clone(classifier)
                    clf.set_params(**cfg)
                    clf.fit(X_train_tune, y_train)

                    try:
                        y_proba_t = clf.predict_proba(X_test_tune)[:, 1]
                    except Exception:
                        raw = clf.decision_function(X_test_tune)
                        y_proba_t = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

                    t = optimize_threshold_f1(y_test, y_proba_t)
                    y_pred_t = (y_proba_t >= t).astype(int)

                    f1_t = f1_score(y_test, y_pred_t)
                    roc_t = roc_auc_score(y_test, y_proba_t)
                    pr_t = compute_pr_auc(y_test, y_proba_t)

                    mlflow.log_metrics({
                        "f1": f1_t,
                        "roc_auc": roc_t,
                        "pr_auc": pr_t,
                        "best_threshold": t
                    })
                    mlflow.log_params(cfg)

                    # Log du modèle tuné (pipeline complet si vectorizer fourni)
                    if vectorizer is not None and not isinstance(classifier, Pipeline):
                        model_to_log_t = Pipeline([("vec", vectorizer), ("clf", clf)])
                        sample_df_t = pd.DataFrame({"text": X_train[:1]})
                        sample_out_t = model_to_log_t.predict_proba(sample_df_t["text"])[:, 1] if hasattr(model_to_log_t, "predict_proba") else model_to_log_t.decision_function(sample_df_t["text"])
                    else:
                        model_to_log_t = clf
                        sample_in_t = X_train_tune[:1]
                        if hasattr(sample_in_t, "toarray"):
                            sample_in_t = sample_in_t.toarray()
                        sample_df_t = pd.DataFrame(sample_in_t)
                        sample_out_t = model_to_log_t.predict_proba(sample_in_t)[:, 1] if hasattr(model_to_log_t, "predict_proba") else model_to_log_t.decision_function(sample_in_t)

                    sig_t = infer_signature(sample_df_t, sample_out_t)
                    try:
                        mlflow.sklearn.log_model(
                            model_to_log_t,
                            name="model",
                            signature=sig_t,
                            input_example=sample_df_t,
                        )
                    except Exception:
                        mlflow.sklearn.log_model(
                            model_to_log_t,
                            name="model",
                            signature=sig_t,
                        )

                    if f1_t > best_f1_global:
                        best_f1_global = f1_t
                        tuned_model = clf
                        metrics_tuned = {
                            "f1": f1_t,
                            "roc_auc": roc_t,
                            "pr_auc": pr_t,
                            "best_threshold": t
                        }

        return classifier, tuned_model, metrics_base, metrics_tuned


# ---------------------------------------------------------------------------
# KERAS : TUNING + BASELINE
# ---------------------------------------------------------------------------

from tensorflow import keras

def tune_model_keras(model_name, X_train, X_test, y_train, y_test,
                     tuner_configs, tuner_build_fn):
    best_f1 = -1
    best_model = None
    best_cfg = None
    best_metrics = None

    for cfg in tuner_configs:

        with _safe_start_run(run_name=f"{model_name}_tuned"):
            mlflow.set_tag("tuned", "true")

            model = tuner_build_fn(cfg)
            model.fit(X_train, y_train,
                      epochs=cfg["epochs"],
                      batch_size=cfg["batch_size"],
                      verbose=0)

            y_proba = model.predict(X_test).ravel()
            t = optimize_threshold_f1(y_test, y_proba)
            y_pred = (y_proba >= t).astype(int)

            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)
            pr = compute_pr_auc(y_test, y_proba)

            mlflow.log_metrics({
                "f1": f1,
                "roc_auc": roc,
                "pr_auc": pr,
                "best_threshold": t
            })
            mlflow.log_params(cfg)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_cfg = cfg
                best_metrics = {
                    "f1": f1,
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "best_threshold": t
                }

    return best_model, best_cfg, best_metrics


def train_and_log_keras_model(
    model_name,
    model,
    X_train, X_test,
    y_train, y_test,
    params,
    epochs=3,
    batch_size=256,
    do_tuning=False,
    tuner_configs=None,
    tuner_build_fn=None,
    dataset_name=None,
    split_run_id=None,
):

    # BASELINE
    with _safe_start_run(run_name=model_name):
        mlflow.set_tag("tuned", "false")

        dataset_tag = dataset_name or (params or {}).get("dataset") or (params or {}).get("dataset_name") or os.getenv("DATASET_NAME")
        split_ref = split_run_id or (params or {}).get("split_run_id") or os.getenv("SPLIT_RUN_ID")
        if dataset_tag:
            mlflow.set_tag("dataset", dataset_tag)
            mlflow.log_param("dataset", dataset_tag)

        if params:
            mlflow.log_params(params)

        # Log input dataset (métadonnées uniquement)
        _log_dataset_input(dataset_tag, split_run_id=split_ref, n_train=len(X_train) if X_train is not None else None, n_test=len(X_test) if X_test is not None else None)

        model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0)

        y_proba = model.predict(X_test).ravel()
        t = optimize_threshold_f1(y_test, y_proba)
        y_pred = (y_proba >= t).astype(int)

        metrics_base = {
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": compute_pr_auc(y_test, y_proba),
            "best_threshold": t
        }

        mlflow.log_metrics(metrics_base)
        mlflow.log_params(params)

        # Log modèle Keras avec signature TensorSpec (évite l'erreur MLflow)
        input_dim = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, input_dim))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1,))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        mlflow.keras.log_model(
            model,
            name="model",
            signature=signature,
        )

        tuned_model = None
        metrics_tuned = None

        # TUNING
        if do_tuning and tuner_configs and tuner_build_fn:
            tuned_model, best_cfg, metrics_tuned = tune_model_keras(
                model_name,
                X_train, X_test,
                y_train, y_test,
                tuner_configs,
                tuner_build_fn
            )

        return model, tuned_model, metrics_base, metrics_tuned


# ---------------------------------------------------------------------------
# RUNS COMPARISON HELPERS
# ---------------------------------------------------------------------------

def get_runs_table(
    experiment_name,
    *,
    prefix=None,
    metrics=("f1", "roc_auc"),
    top_n=10,
    order_by_metric=None,
    max_results=5000,
    human_duration=True,
    include_params=True,
    exclude_param_keywords=(),
    drop_constant_params=True,
):
    """
    Retourne un DataFrame des runs MLflow filtrés et triés.

    - prefix : filtre sur le nom du run (startswith)
    - metrics : liste des métriques à récupérer (colonne metrics.<name>)
    - top_n : nombre de runs retournés (None pour tous)
    - order_by_metric : métrique utilisée pour le tri décroissant (défaut = premier de metrics)
    Inclut automatiquement la durée du run (en secondes) si start_time / end_time sont présents.
    - human_duration : ajoute duration_display (format lisible) + duration_hms (hh:mm:ss).
    - Tague les runs sans fin connue comme "non fini" via duration_status.
    - include_params : ajoute toutes les colonnes params.* (préfixées en param_) pour inspecter les hyperparams.
    - exclude_param_keywords : tuple de mots-clés pour filtrer des params de contexte (embedding, model, dataset...).
    - drop_constant_params : supprime les colonnes param_* vides ou constantes sur les runs retournés.
    """

    def _format_duration(sec):
        if sec is None or pd.isna(sec):
            return "non fini"
        try:
            sec = float(sec)
        except Exception:
            return "non fini"
        if sec < 60:
            return f"{int(round(sec))} s"
        if sec < 3600:
            m = int(sec // 60)
            s = int(round(sec - m * 60))
            return f"{m} min {s:02d} s"
        h = int(sec // 3600)
        rem = sec - h * 3600
        m = int(rem // 60)
        s = int(round(rem - m * 60))
        return f"{h} h {m:02d} min {s:02d} s"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Expérience introuvable : {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=max_results,
    )

    if prefix is not None:
        pref = str(prefix)
        if pref != "":
            rn = runs["tags.mlflow.runName"].fillna("")
            mask = rn.str.lower().str.startswith(pref.lower())
            runs = runs[mask]
        # si pref == "" on ne filtre pas

    if runs.empty:
        return pd.DataFrame(columns=["model", *metrics])

    # Durée du run en secondes (start_time et end_time sont en ms).
    # Fallback : si end_time manquant, on tente mlflow.get_run ; sinon tag "non fini".
    runs = runs.copy()
    if "start_time" in runs.columns and "end_time" in runs.columns:
        delta = runs["end_time"] - runs["start_time"]
        if hasattr(delta, "dt"):
            runs["duration_sec"] = delta.dt.total_seconds()
        else:
            runs["duration_sec"] = delta / 1000.0
    else:
        runs["duration_sec"] = np.nan

    missing_duration = runs["duration_sec"].isna()
    durations = []
    statuses = []
    client = None
    if MlflowClient is not None:
        try:
            client = MlflowClient()
        except Exception:
            client = None
    if missing_duration.any():
        for rid, is_missing in zip(runs["run_id"], missing_duration):
            if not is_missing:
                continue
            try:
                run_info = mlflow.get_run(rid).info
                st = getattr(run_info, "start_time", None)
                en = getattr(run_info, "end_time", None)
                if st is not None and en is not None:
                    durations.append((en - st) / 1000.0)
                    statuses.append("fini")
                else:
                    durations.append(np.nan)
                    statuses.append("non fini")
            except Exception:
                durations.append(np.nan)
                statuses.append("non fini")

            if statuses[-1] == "non fini" and client is not None:
                try:
                    client.set_tag(rid, "non_fini", "true")
                except Exception:
                    pass

        runs.loc[missing_duration, "duration_sec"] = durations
        runs.loc[missing_duration, "duration_status"] = statuses

    if "duration_status" not in runs.columns:
        runs["duration_status"] = "fini"
    else:
        runs["duration_status"] = runs["duration_status"].fillna("fini")

    # Si duration_sec est un Timedelta, convertir en secondes
    if "duration_sec" in runs.columns:
        try:
            if pd.api.types.is_timedelta64_dtype(runs["duration_sec"]):
                runs["duration_sec"] = runs["duration_sec"].dt.total_seconds()
        except Exception:
            pass


    col_map = {"tags.mlflow.runName": "model"}
    for m in metrics:
        col = f"metrics.{m}"
        if col in runs.columns:
            col_map[col] = m

    # Ajouter la durée si calculée
    if "duration_sec" in runs.columns:
        col_map["duration_sec"] = "duration_sec"
        col_map["duration_status"] = "duration_status"

    present_cols = list(col_map.keys())
    df = runs[present_cols].copy().rename(columns=col_map)

    # Optionnel : ajouter toutes les hyperparams loguées (params.*) en les préfixant pour éviter les collisions.
    if include_params:
        param_cols = [c for c in runs.columns if c.startswith("params.")]
        if exclude_param_keywords:
            lowered = tuple(k.lower() for k in exclude_param_keywords)
            param_cols = [c for c in param_cols if not any(k in c.lower() for k in lowered)]
        if param_cols:
            params_df = runs[param_cols].copy()
            params_df.columns = [c.replace("params.", "param_") for c in params_df.columns]
            # reset_index pour éviter des mismatches d'index en cas de filtrage/tri
            df = pd.concat([df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)

    if df.empty:
        return pd.DataFrame(columns=["model", *metrics])

    # Optionnel : supprimer les colonnes param_* vides ou constantes
    if drop_constant_params:
        param_cols_prefixed = [c for c in df.columns if c.startswith("param_")]
        to_drop = []
        for col in param_cols_prefixed:
            series = df[col]
            if not series.notna().any() or series.nunique(dropna=True) <= 1:
                to_drop.append(col)
        if to_drop:
            df = df.drop(columns=to_drop)

    if human_duration and "duration_sec" in df.columns:
        df["duration_hms"] = pd.to_timedelta(df["duration_sec"], unit="s").astype(str).str.split(".").str[0]
        df["duration_display"] = df["duration_sec"].apply(_format_duration)

    sort_metric = order_by_metric or (metrics[0] if metrics else None)
    if sort_metric in df.columns:
        df = df.sort_values(by=sort_metric, ascending=False)

    if top_n:
        df = df.head(top_n)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LOADER SERVING : TEXTE BRUT → MODELE MLflow
# ---------------------------------------------------------------------------

def load_for_serving(run_id, dst_path=None):
    """
    Charge un artefact MLflow et retourne un modèle prêt pour l'inférence.
    - Si l'artefact est un pipeline complet (ex: TF-IDF sklearn), il est renvoyé tel quel.
    - Si l'artefact attend des embeddings USE/BERT, on ajoute un prétraitement texte->embedding.
    - dst_path permet de préciser un dossier de cache pour download_artifacts (optionnel).
    """
    client = MlflowClient()

    # Télécharge le dossier d'artéfact du modèle (au besoin, essaie de deviner le sous-dossier)
    def _download_model_artifact():
        # Chemin par défaut
        try:
            return client.download_artifacts(run_id, "model", dst_path=dst_path)
        except Exception:
            # Si "model" n'existe pas, on inspecte la racine et on prend un dossier/fichier plausible
            entries = client.list_artifacts(run_id, path="")
            if not entries:
                raise MlflowException(
                    "Aucun artefact trouvé pour ce run (pas de dossier 'model' ni d'autres artefacts)"
                )
            # priorité : un dossier nommé "model"
            for art in entries:
                if getattr(art, "is_dir", False) and art.path.rstrip("/").lower() == "model":
                    return client.download_artifacts(run_id, art.path, dst_path=dst_path)
            # sinon premier dossier
            dirs = [art.path for art in entries if getattr(art, "is_dir", False)]
            if dirs:
                return client.download_artifacts(run_id, dirs[0], dst_path=dst_path)
            # sinon premier fichier (ex: model.pkl directement à la racine)
            files = [art.path for art in entries if not getattr(art, "is_dir", False)]
            if files:
                return client.download_artifacts(run_id, files[0], dst_path=dst_path)
            # Rien d'exploitable
            raise MlflowException(
                "Aucun artefact exploitable pour ce run (ni dossier, ni fichier)"
            )

    local_path = _download_model_artifact()

    mlmodel = Model.load(f"{local_path}/MLmodel")
    flavors = set(mlmodel.flavors.keys())

    # Récup des params pour savoir quel embedding est attendu
    run = mlflow.get_run(run_id)
    params = run.data.params
    embedding_type = params.get("embedding_type", "").lower()
    embedding_model = params.get("embedding_model", "")

    # Choix du loader selon le flavor
    if "sklearn" in flavors:
        base_model = mlflow.sklearn.load_model(local_path)
    elif "keras" in flavors:
        base_model = mlflow.keras.load_model(local_path)
    elif "pytorch" in flavors:
        from mlflow import pytorch as mlflow_pytorch
        base_model = mlflow_pytorch.load_model(local_path)
    elif "lightgbm" in flavors:
        from mlflow import lightgbm as mlflow_lgbm
        base_model = mlflow_lgbm.load_model(local_path)
    else:
        base_model = mlflow.pyfunc.load_model(local_path)

    # Ajout du prétraitement si l'artefact n'embarque pas l'encodeur
    if embedding_type == "use":
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ImportError("tensorflow_hub manquant pour encoder USE") from exc
        encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        def encode_use(texts):
            return encoder(texts).numpy()

        return Pipeline([
            ("encode", FunctionTransformer(encode_use, validate=False)),
            ("model", base_model),
        ])

    if embedding_type in {"bert", "distilbert", "hf", "sbert"}:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError("torch/transformers manquants pour encoder BERT/HF") from exc

        model_name = embedding_model or "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(model_name)
        enc_model = AutoModel.from_pretrained(model_name)

        def encode_hf(texts):
            if isinstance(texts, str):
                texts = [texts]
            tokens = tok(
                list(texts),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = enc_model(**tokens)
            # vecteur CLS
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return cls

        return Pipeline([
            ("encode", FunctionTransformer(encode_hf, validate=False)),
            ("model", base_model),
        ])

    # Par défaut : pipeline déjà complet (TF-IDF, CountVectorizer, etc.)
    return base_model
