import requests
import pushover
import sys
import os
PUSHOVER_USER = pushover.user
PUSHOVER_TOKEN = pushover.api

def send_notification(message, title=""):
    title = title + " " + os.path.basename(sys.argv[0]) + " " + " ".join(sys.argv[1:])
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_TOKEN,
            "user": PUSHOVER_USER,
            "title": title,
            "message": message
        }
    )

def get_compact_traceback():
    import traceback
    tb = traceback.extract_tb(sys.exc_info()[2])
    last = tb[-1]  # ultima chiamata

    return f"{last.filename}:{last.lineno} in {last.name} → {last.line}"
# Vanilla A.I.D.

#%pip install -r ../requirements.txt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import PrototypeRepresentationLearner
from fairlearn.reductions import ExponentiatedGradient
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from time import time
import os
import json
import pickle
import config
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data import label_encoding
import warnings
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity
import shap
from shap.explainers import KernelExplainer
from utils.test import test_accuracy, test_fairness, test_model, combined_score, explain_fav_unfav_mapping, print_metrics_table, keys_of_dict
import re
import argparse
warnings.filterwarnings("ignore", category=FutureWarning)

class ManualFairSearch:
    def __init__(self, best_estimator_, best_params_, best_score_, cv_results_):
        self.best_estimator_ = best_estimator_
        self.best_params_ = best_params_
        self.best_score_ = best_score_
        self.cv_results_ = cv_results_

from joblib import Parallel, delayed, Memory
memory = Memory(location="../out/tmp", verbose=0)


@memory.cache
def evaluate_params(params, estimator, splitter, X_train, y_train,
                    sensitive_col, random_state, custom_sensitives):

    from sklearn.base import clone
    import numpy as np

    fold_scores = []

    for train_idx, val_idx in splitter.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx].reset_index(drop=True)
        X_va = X_train.iloc[val_idx].reset_index(drop=True)
        y_tr = y_train.iloc[train_idx].reset_index(drop=True)
        y_va = y_train.iloc[val_idx].reset_index(drop=True)

        model = clone(estimator)
        model.set_params(**params)

        # fit
        if custom_sensitives is not None:
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr, sensitive_features=X_tr[sensitive_col])

        # predict
        if isinstance(model, ThresholdOptimizer):
            preds = model.predict(X_va, sensitive_features=X_va[sensitive_col],
                          random_state=random_state)
        else:
            preds = model.predict(X_va)

        if len(set(preds)) == 1:
            print("Only one class predicted.")
            #send_notification("Only one class predicted.", title="⚠️")

        # eval
        if custom_sensitives is None:
            accuracy_metrics, fairness_metrics = test_model(
                model, sensitive_col, X_va, y_va, predictions_prob=False)
        else:
            accuracy_metrics, fairness_metrics = test_model(
                model, sensitive_col, X_va, y_va,
                predictions_prob=False,
                custom_sensitives=custom_sensitives.iloc[val_idx])

        score = combined_score(accuracy_metrics, fairness_metrics)
        fold_scores.append(score)

    return {
        "params": params,
        "score": np.mean(fold_scores),
        "fold_scores": fold_scores
    }

def train_best_fair_model_parallel(estimator, param_grid, X_train, y_train, sensitive_col, model_path, cv=5, random_state=42, verbose=False, custom_sensitives=None):
    from sklearn.model_selection import StratifiedKFold, ParameterGrid
    from sklearn.base import clone
    import numpy as np

    search_path = model_path + "_search"
    # check if search already cached to disk and load it
    if os.path.exists(search_path):
        with open(search_path, "rb") as f:
            search = pickle.load(f)
        if verbose:
            print("\tFair model search loaded from disk ({}).".format(search_path))
        return search.best_estimator_, search.best_params_, search

    # if not exists, then search
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []
    best_score = -np.inf
    best_params = None

    if hasattr(estimator, "n_jobs"):
        estimator.set_params(n_jobs=1)

    # for each config of params
    from sklearn.model_selection import ParameterGrid

    param_list = list(ParameterGrid(param_grid))

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_params)(
            params,
            estimator,
            splitter,
            X_train,
            y_train,
            sensitive_col,
            random_state,
            custom_sensitives
        )
        for params in param_list
    )

    best_result = max(results, key=lambda x: x["score"])
    best_params = best_result["params"]
    best_score = best_result["score"]

     # refit
    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    fit_kwargs = {}
    if custom_sensitives is not None: best_estimator.fit(X_train, y_train)
    else: best_estimator.fit(X_train, y_train, sensitive_features=X_train[sensitive_col])
    cv_results_ = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

    search = ManualFairSearch(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        cv_results_=cv_results_
    )

    # Salva modello e search
    with open(model_path, "wb") as f:
        pickle.dump(best_estimator, f)
    with open(search_path, "wb") as f:
        pickle.dump(search, f)

    if verbose:
        print("\tFair model saved to disk ({}).".format(model_path))
        print("\tBest params:", best_params)
        print("\tBest score:", best_score)

    return best_estimator, best_params, search


def train_best_fair_model(estimator, param_grid, X_train, y_train, sensitive_col, model_path, cv=5, random_state=42, verbose=False, custom_sensitives=None):
    from sklearn.model_selection import StratifiedKFold, ParameterGrid
    from sklearn.base import clone
    import numpy as np

    search_path = model_path + "_search"
    # check if search already cached to disk and load it
    if os.path.exists(search_path):
        with open(search_path, "rb") as f:
            search = pickle.load(f)
        if verbose:
            print("\tFair model search loaded from disk ({}).".format(search_path))
        return search.best_estimator_, search.best_params_, search

    # if not exists, then search
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []
    best_score = -np.inf
    best_params = None

    # for each config of params
    for params in ParameterGrid(param_grid):
        fold_scores = []

        if verbose:
            print("\t>>> Testing params:", params)

        # split train/validation
        for train_idx, val_idx in splitter.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx].reset_index(drop=True)
            X_va = X_train.iloc[val_idx].reset_index(drop=True)
            y_tr = y_train.iloc[train_idx].reset_index(drop=True)
            y_va = y_train.iloc[val_idx].reset_index(drop=True)

            # clone estimator with params
            model = clone(estimator)
            model.set_params(**params)

            # training
            if custom_sensitives is not None: model.fit(X_tr, y_tr)
            else: model.fit(X_tr, y_tr, sensitive_features=X_tr[sensitive_col])

            # testing
            model.predict(X_va, sensitive_features=X_va[sensitive_col], random_state=random_state) if isinstance(model, ThresholdOptimizer) else model.predict(X_va)

            if custom_sensitives is None:
                accuracy_metrics, fairness_metrics = test_model(model, sensitive_col , X_va, y_va, predictions_prob=False)
            else:
                accuracy_metrics, fairness_metrics = test_model(model, sensitive_col , X_va, y_va, predictions_prob=False, custom_sensitives=custom_sensitives.iloc[val_idx])

            # add to results
            score = combined_score(accuracy_metrics, fairness_metrics)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        results.append({
            "params": params,
            "score": mean_score,
            "fold_scores": fold_scores
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

     # refit
    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    fit_kwargs = {}
    if custom_sensitives is not None: best_estimator.fit(X_train, y_train)
    else: best_estimator.fit(X_train, y_train, sensitive_features=X_train[sensitive_col])
    cv_results_ = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

    search = ManualFairSearch(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        cv_results_=cv_results_
    )

    # Salva modello e search
    with open(model_path, "wb") as f:
        pickle.dump(best_estimator, f)
    with open(search_path, "wb") as f:
        pickle.dump(search, f)

    if verbose:
        print("\tFair model saved to disk ({}).".format(model_path))
        print("\tBest params:", best_params)
        print("\tBest score:", best_score)

    return best_estimator, best_params, search

def train_best_model(params, X_train, y_train, model_path, verbose=False, sensitive_features=None):
    if os.path.exists(model_path):
        # if model/search already exists load it
        with open(model_path + "_search", "rb") as f:
            search = pickle.load(f)
        if verbose: print("\tModel loaded from disk ({}).".format(model_path))
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        # otherwise train the model and save it
        search = HalvingGridSearchCV(**params)
        if sensitive_features is not None: search.fit(X_train, y_train, sensitive_features=sensitive_features)
        else: search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        if verbose: print("\tModel saved to disk ({}).".format(model_path))
        with open(model_path + "_search", "wb") as f:
            pickle.dump(search, f)
    return model, best_params

def preprocess_debiasing(model_type, param_grid,  X_train, y_train, X_test, verbose=False, pickle_path=None):
    from itertools import product
    combinations = []
    if model_type==PrototypeRepresentationLearner:
        keys = list(param_grid.keys())
        combinations = [
            dict(zip(keys, values))
            for values in product(*(param_grid[k] for k in keys))
        ]
    data_versions = []
    models_versions = []
    for params in combinations:
        if verbose: print(params)
        model = model_type(**params)
        model.fit(X_train, y_train)
        X_train_transformed = model.transform(X_train)
        X_test_transformed = model.transform(X_test)

        models_versions.append(model)
        data_versions.append((X_train_transformed, X_test_transformed))

    if pickle_path is not None:
        with open(pickle_path, "wb") as f:
            pickle.dump(data_versions, f)
        if verbose: print("Transformed Train/Test sets saved to {}".format(pickle_path))
        with open(pickle_path + "_models", "wb") as f:
            pickle.dump(models_versions, f)
        if verbose: print("Models saved to {}".format(pickle_path + "_models"))

    return data_versions, models_versions

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing.")
    parser.add_argument("d", type=str, help="Name of one of the available datasets in the repo")
    parser.add_argument("-ms", "--max_size", type=int, help="Limit row numbers for the specified dataset", default=None)
    args = parser.parse_args()

    d = args.d
    max_size = args.max_size

    #data_pool = ['adult', 'bank_marketing']
    unm_model_pool = ['lr']
    debiasing_pool = ['to', 'eg', 'prl']
    debiasing_prep = ['prl']
    discr_model_pool = ['lr', 'xgb', 'mlp']
    test_size = 0.2

    model_types = {
        "lr": LogisticRegression,
        "mlp": MLPClassifier,
        "to": ThresholdOptimizer,
        "xgb": XGBClassifier,
        "prl": PrototypeRepresentationLearner,
        "eg": ExponentiatedGradient
    }
    # Cols for exps results
    res_cols = config.res_cols

    preprocessing_debiasing = [PrototypeRepresentationLearner]

    model_grids = config.model_grids
    fairness_levels = config.fairness_lv

    from_exp = 1
    exp_no = 1
    results_df = pd.DataFrame(columns=res_cols)
    checkpoint_exists = False

    # check if a checkpoint already exists
    if os.path.exists(config.checkpoint_path.format(d)):
        checkpoint_exists = True
        results_df = pd.read_pickle(config.checkpoint_path.format(d))
        from_exp = len(results_df)
        print("Checkpoint containing {} tests loaded.".format(from_exp))

    f = open(config.datasets_info_path, "r")
    datasets_info = json.load(f)
    f.close()

    data_info = datasets_info[d]
    sensitives = list(data_info['fairness']['sensitives'].keys())  # names of the columns
    sensitives_maps = data_info['fairness']['sensitives']  # mapping for multiple values subst. for sensitives
    target_encoding = data_info['preprocessing']['target_encoding']

    padding = 2
    width = len(d) + padding * 2

    print("+" + "-" * width + "+")
    print("|" + d.center(width) + "|")
    print("+" + "-" * width + "+")

    print(">>> Dataset Info")
    print("\tsensitives:", sensitives)

    print("\t\t", sensitives_maps)
    print("\ttarget:", target_encoding)

    if os.path.exists(config.datasets_path + "/{}.preprocessed".format(d)):
        with open(config.datasets_path + "/{}.preprocessed".format(d), "rb") as f:
            X, Y = pickle.load(f)
        print("\tDataset loaded from disk.")
    else:
        raise Exception("Dataset not found. Run data_preprocess.py first.")

    print(">>> Shuffling & Splitting")

    if max_size is not None:
        print("\tmax_size:", max_size)
        X, _, Y, _ = train_test_split(
            X, Y,
            train_size=max_size,
            stratify=Y,
            random_state=42
        )

    X_train_unm, X_probing, y_train_unm, y_probing = train_test_split(X, Y, test_size=0.5, random_state=42,
                                                                      shuffle=True, stratify=Y)
    # index reset
    # X_train, y_train --> for training un/mitigated model
    X_train_unm = X_train_unm.reset_index(drop=True)
    y_train_unm = y_train_unm.reset_index(drop=True)
    # X_probing, y_probing --> for probing the unmitigated model, label subst. etc...
    X_probing = X_probing.reset_index(drop=True)
    y_probing = y_probing.reset_index(drop=True)

    with open("../out/data/{}_probing.data".format(d), "wb") as f:
        pickle.dump((X_probing, y_probing), f)

    for s in sensitives:
        X_train, X_test, y_train, y_test = train_test_split(X_train_unm, y_train_unm, test_size=test_size,
                                                            random_state=42, shuffle=True, stratify=y_train_unm)

        # index reset
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        for unm_model in unm_model_pool:
            start_time_unmitigated = time()
            print(">>> (1) Unmitigated model ({}) training...".format(model_types[unm_model].__name__))
            unmitigated_model_path = "../out/models/unmitigated_{}_{}.model".format(unm_model, d)
            # Search for best model
            unmitigated_halving_params = {
                "estimator": model_types[unm_model](random_state=42),
                "param_grid": model_grids[unm_model],
                "scoring": "f1_macro",
                "random_state": 42,
                "cv": 5,
                "factor": 2,
                "n_jobs": -1,
                "min_resources": int(len(X_train) / 2)
                # usa tutti i campioni nel fold per evitare problemi con classi uniche
            }
            unmitigated, unmitigated_params = train_best_model(unmitigated_halving_params, X_train, y_train,
                                                               unmitigated_model_path, verbose=True)
            unmitigated_accuracy_dict, unmitigated_fairness_dict = test_model(unmitigated, s, X_test, y_test)

            end_time_unmitigated = time()

            for flv in fairness_levels.keys():
                for debiasing in debiasing_pool:
                    start_time_debiasing = time()
                    print(">>> (2) Debiasing ({}) model on {}. Fairness lv. {}".format(model_types[debiasing].__name__, s, flv))
                    mitigated_model_path = "../out/models/mitigated_{}_{}_{}-{}.model".format(debiasing, flv, s, d)
                    prep_path = "../out/data/{}_{}_{}_{}".format(debiasing, flv, s, d)

                    trans_model = None

                    # add fairness lv to debiasing model grid
                    for k, v in fairness_levels[flv][debiasing].items():
                        model_grids[debiasing][k] = v

                    if debiasing in debiasing_prep:

                        # if preprocessing algo then
                        # if preprocessing is cached then
                        if os.path.exists(prep_path + ".data_versions") and os.path.exists(
                                prep_path + ".data_versions_models"):

                            with open(prep_path + ".data_versions", "rb") as f:
                                data_versions = pickle.load(f)
                            print("Data versions loaded from disk.")

                            with open(prep_path + ".data_versions_models", "rb") as f:
                                model_versions = pickle.load(f)
                            print("Model versions loaded from disk.")
                        else:
                            data_versions, model_versions = preprocess_debiasing(model_types[debiasing],
                                                                                 model_grids[debiasing], X_train, y_train,
                                                                                 X_test, verbose=True,
                                                                                 pickle_path="../out/data/{}_{}_{}_{}.data_versions".format(
                                                                                     debiasing, flv, s, d))

                        pattern = re.compile(rf"{debiasing}_{flv}_{s}_{d}_best_model\((\d+)\)$")
                        full_path = None

                        for filename in os.listdir("../out/models/"):
                            match = pattern.match(filename)
                            if match:
                                best_index = int(match.group(1))
                                print("Best data version", best_index)
                                full_path = os.path.join("../out/models/", filename)

                        if full_path is not None:
                            with open(full_path, "rb") as f:
                                mitigated = pickle.load(f)
                            print("Best model loaded from disk.")
                        else:
                            lr_grid = {
                                "penalty": ["l2"],
                                "C": [0.1, 1.0],
                                "solver": ["lbfgs"],
                                "max_iter": [1000],
                                "tol": [1e-4, 1e-2],
                                "fit_intercept": [True],
                                "class_weight": ["balanced", None],
                            }

                            best_search_score = -1
                            best_search = None
                            best_index = None

                            for i, (X_train_transformed, X_test_transformed) in enumerate(data_versions):
                                search_params = {
                                    "estimator": LogisticRegression(),
                                    "param_grid": lr_grid,
                                    "X_train": pd.DataFrame(X_train_transformed),
                                    "y_train": y_train,
                                    "sensitive_col": s,
                                    "model_path": "../out/models/{}_{}_{}_{}_v{}".format(debiasing, flv, s, d, i),
                                    "custom_sensitives": X_train[s],
                                }
                                mit, mit_params, search = train_best_fair_model_parallel(**search_params)
                                if search.best_score_ > best_search_score:
                                    best_search = search
                                    best_search_score = search.best_score_
                                    best_index = i

                            mitigated = best_search.best_estimator_
                            mitigated_params = best_search.best_params_

                            with open("../out/models/{}_{}_{}_{}_best_model({})".format(debiasing, flv, s, d, best_index),
                                      "wb") as f:
                                pickle.dump(mitigated, f)
                            print("Best model cached to disk.")

                        mitigated_accuracy_dict, mitigated_fairness_dict = test_model(mitigated, s,
                                                                                      data_versions[best_index][1], y_test,
                                                                                      custom_sensitives=X_test[s])
                    else:
                        if debiasing == 'to':
                            #for param_dict in model_grids[debiasing]:
                            model_grids[debiasing]["estimator"] = [unmitigated]

                        mitigated, mitigated_params, mitigated_search = train_best_fair_model_parallel(
                            estimator=model_types[debiasing]() if debiasing == 'to' else model_types[debiasing](
                                estimator=LogisticRegression(), constraints=DemographicParity()),
                            param_grid=model_grids[debiasing],
                            X_train=X_train,
                            y_train=y_train,
                            sensitive_col=s,
                            model_path=mitigated_model_path,
                            cv=5,
                            random_state=42,
                            verbose=True
                        )

                        mitigated_accuracy_dict, mitigated_fairness_dict = test_model(mitigated, s, X_test, y_test)

                    # for key in unmitigated_accuracy_dict.keys(): print(key,unmitigated_accuracy_dict[key], mitigated_accuracy_dict[key])
                    # for key in unmitigated_fairness_dict.keys(): print(key,unmitigated_fairness_dict[key], mitigated_fairness_dict[key])
                    print("\tDP-diff", unmitigated_fairness_dict['DP-diff'], "-->", mitigated_fairness_dict['DP-diff'])
                    print("\tDP-ratio", unmitigated_fairness_dict['DP-ratio'], "-->", mitigated_fairness_dict['DP-ratio'])

                    # PROBING mitigated
                    print(">>> Probing mitigated model...")

                    if isinstance(mitigated, ExponentiatedGradient):
                        y_u = mitigated.predict(X_probing, random_state=42)
                    elif debiasing == 'prl':
                        trans_model = model_versions[best_index]
                        mitigated_params = [trans_model.get_params(), mitigated_params]
                        y_u = mitigated.predict(trans_model.transform(X_probing))
                    else:
                        y_u = mitigated.predict(X_probing, sensitive_features=X_probing[s], random_state=42)

                    # y_u = mitigated.predict(X_probing, random_state=42) if isinstance(mitigated, ExponentiatedGradient)  else mitigated.predict(X_probing, sensitive_features=X_probing[s], random_state=42)

                    end_time_debiasing = time()

                    # FAV/UNFAV MAPPING
                    fav_unfav_mapping = data_info['fairness']['fav_labels_mapping']
                    print("\tFAV-UNFAV mapping", fav_unfav_mapping)
                    print("\tFAV-UNFAV meaning", explain_fav_unfav_mapping(fav_unfav_mapping, target_encoding, cat=True))

                    # ENCODE y_u
                    y_u_encoded = label_encoding(pd.Series(y_u).astype(str), map=fav_unfav_mapping)
                    # ENCODE     y_probing
                    y_probing_encoded = label_encoding(pd.Series(y_probing).astype(str), map=fav_unfav_mapping)
                    # NEW LABELS: Y'
                    y_prime = [1 if y_i_u != y_i else 0 for y_i_u, y_i in zip(y_u_encoded, y_probing_encoded)]

                    # SPLITTING FOR DISCR. TRAINING
                    print(">>> Splitting for discriminator training...")
                    from imblearn.over_sampling import RandomOverSampler
                    X_prob_train, X_prob_test, y_prime_train, y_prime_test = train_test_split(X_probing, y_prime,
                                                                                              test_size=test_size,
                                                                                              random_state=42, shuffle=True,
                                                                                              stratify=y_prime)

                    X_prob_train = X_prob_train.reset_index(drop=True)
                    X_prob_test = X_prob_test.reset_index(drop=True)

                    ros = RandomOverSampler(random_state=42)  # inizializza l"oversampler
                    X_prob_train, y_prime_train = ros.fit_resample(X_prob_train, y_prime_train)

                    X_prob_train = X_prob_train.reset_index(drop=True)
                    X_prob_test = X_prob_test.reset_index(drop=True)

                    for discr_model in discr_model_pool:
                        if checkpoint_exists:
                            if exp_no <= from_exp:
                                # FIXME: add check
                                exp_no += 1
                                continue
                        print(">>> (3) Training discriminator ({}) on new labels...".format(
                            model_types[discr_model].__name__))
                        discr_model_path = "../out/models/discr_{}_{}_{}-{}-{}.model".format(discr_model, debiasing, flv, s, d)
                        start_time_discr = time()

                        discriminator_halving_params = {
                            "estimator": model_types[discr_model](),
                            "param_grid": model_grids[discr_model],
                            "scoring": "f1_macro",
                            "random_state": 42,
                            "cv": 5,
                            "factor": 2,
                            "n_jobs": -1,
                            "min_resources": int(len(X_train) / 2)
                        }

                        discriminator, discriminator_params = train_best_model(discriminator_halving_params, X_prob_train,
                                                                               y_prime_train, discr_model_path,
                                                                               verbose=True)

                        discr_accuracy_dict, discr_fairness_dict = test_model(discriminator, s, X_prob_test, y_prime_test)

                        print_metrics_table(keys_of_dict(discr_accuracy_dict,
                                                         ['accuracy', 'auc', 'f1_score', 'precision', 'recall',
                                                          'matthew_coeff']))
                        print_metrics_table(discr_fairness_dict)

                        end_time_discr = time()

                        # GETTINKG KNOWLEDGE ABOUT DEBIASING
                        explainer_path = "../out/models/shap_{}_{}-{}-{}-{}-{}.model".format(discr_model, unm_model, debiasing, flv,
                                                                                          s, d)

                        if os.path.exists(explainer_path):
                            # with open(explainer_path, "rb") as f:
                            #   artifact = pickle.load(f)
                            print("Shap values are already computed.", explainer_path)


                        else:
                            background = shap.sample(X_probing, 100)  # also for preprocessing should be valid

                            k = int(1.0 * X_probing.shape[1])

                            def model_wrapper(X):
                                # for MLP
                                if not isinstance(X, pd.DataFrame):
                                    X = pd.DataFrame(X, columns=X_probing.columns)

                                return discriminator.predict_proba(X)

                            explainer = KernelExplainer(
                                model_wrapper,
                                data=background,
                                link="logit"
                            )

                            shap_values = explainer.shap_values(X_probing, nsamples=1000, l1_reg=f"num_features({k})")

                            artifact = {
                                "shap_values": shap_values,
                                #"X": X_probing,
                                "background": background,
                                "expected_value": explainer.expected_value,
                                "feature_names": X_probing.columns.tolist()
                            }

                            with open(explainer_path, "wb") as f:
                                pickle.dump(artifact, f)
                            print(">>> Shap explainer saved to disk", explainer_path)

                        end_time = time()
                        results_df.loc[exp_no] = {
                            "total_time": end_time - start_time_unmitigated,  # for one whole run of exp #
                            # Data Info
                            "dataset": d,
                            "sensitive": s,
                            "test_size": test_size,
                            # Unmitigated Model
                            "unm_model_type": model_types[unm_model].__name__,
                            "unm_model_accuracy": unmitigated_accuracy_dict,  # JSON with objs
                            "unm_model_fairness": unmitigated_fairness_dict,  # JSON with objs
                            "unm_model_params": unmitigated_params,
                            "unm_model_exp_time": end_time_unmitigated - start_time_unmitigated,
                            # Mitigated Model (Debiasing)
                            "debiasing_lv": flv,
                            "debiasing_type": model_types[debiasing].__name__,
                            "debiasing_accuracy": mitigated_accuracy_dict,  # JSON with objs
                            "debiasing_fairness": mitigated_fairness_dict,  # JSON with objs
                            "debiasing_params": mitigated_params,
                            "debiasing_combined_score": combined_score(mitigated_accuracy_dict, mitigated_fairness_dict),
                            "debiasing_exp_time": end_time_debiasing - start_time_debiasing,
                            # Discriminator Model
                            "discr_model_type": model_types[discr_model].__name__,
                            "discr_model_accuracy": discr_accuracy_dict,  # JSON with objs
                            "discr_model_fairness": discr_fairness_dict,  # JSON with objs,
                            "discr_model_params": discriminator_params,
                            "discr_model_exp_time": end_time_discr - start_time_discr,
                            "y_u": y_u_encoded,  # predictions of discriminator
                            "y_probing": y_probing_encoded,  # ground_truths
                            "y_prime": y_prime  # new labels
                            # XAI
                        }
                        exp_no += 1
                        results_df.to_pickle(config.checkpoint_path.format(d))
    # save results to disk and delete checkpoints
    results_df.to_pickle(config.results_path.format(d))
    if os.path.exists(config.checkpoint_path.format(d)): os.remove(config.checkpoint_path.format(d))

if __name__ == "__main__":
    try:
        main()
        send_notification("Completed", title="✅")
        memory.clear()
    except Exception as e:
        loc = get_compact_traceback()
        send_notification(f"{type(e).__name__}: {e}\n{loc}", title="❌")
        raise

import atexit

def on_exit():
    send_notification("Script terminated (possibily due to crashes).", title="💥")

atexit.register(on_exit)

