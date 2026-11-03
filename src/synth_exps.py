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
    print()

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
from src.config import probing_sets_dict, debiasing_seeds
import warnings
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity
import shap
from shap.explainers import KernelExplainer
from utils.test import test_accuracy, test_fairness, test_model, combined_score, explain_fav_unfav_mapping, print_metrics_table, keys_of_dict
import re
import argparse
warnings.filterwarnings("ignore", category=FutureWarning)
from src.exps import train_best_model, preprocess_debiasing, train_best_fair_model_parallel, keys_of_dict, print_metrics_table, combined_score
from src.utils.test import test_model
from src.utils.data import label_encoding
import numpy as np

from fairlearn.reductions import DemographicParity

from joblib import Parallel, delayed, Memory
memory = Memory(location="../out/tmp", verbose=0)

# settings
unm_model_pool = ['lr']
debiasing_pool = ['to', 'eg', 'prl']
debiasing_prep = ['prl']
discr_model_pool = ['xgb']
test_size = 0.3

model_types = {
    "lr": LogisticRegression,
    "mlp": MLPClassifier,
    "to": ThresholdOptimizer,
    "xgb": XGBClassifier,
    "prl": PrototypeRepresentationLearner,
    "eg": ExponentiatedGradient
}

# Cols for exps results
# FIXME: da cambiare
res_cols = config.res_cols

preprocessing_debiasing = [PrototypeRepresentationLearner]

model_grids = config.model_grids
fairness_levels = config.fairness_lv

def data_loading(d, max_size=None):
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
    return X, Y, data_info, sensitives, sensitives_maps, target_encoding

def unmitigated_training(unm_model, s, X_train, y_train, X_test, y_test, d, n_samples):
    print(">>> (1) Unmitigated model ({}) training...".format(model_types[unm_model].__name__))
    unmitigated_model_path = f"../out/models/unmitigated_{unm_model}_{d}_{n_samples}.model"

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
    return unmitigated, unmitigated_params, unmitigated_accuracy_dict, unmitigated_fairness_dict

def mitigated_training(X_train, y_train, X_test, y_test, unmitigated, unmitigated_fairness_dict, debiasing, flv, s, d, n_samples, seed=42):
    print(">>> (2) Debiasing ({}) model on {}. Fairness lv. {}".format(model_types[debiasing].__name__, s, flv))
    mitigated_model_path = f"../out/models/mitigated_{debiasing}_{flv}_{s}-{d}_{n_samples}.model_seed{seed}"
    prep_path = f"../out/data/{debiasing}_{flv}_{s}_{d}_{n_samples}_seed{seed}"

    trans_model = None
    model_versions = None

    # TODO: add diff_bound for EG
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
            # "../out/data/{}_{}_{}_{}.data_versions".format(debiasing, flv, s, d)
            data_versions, model_versions = preprocess_debiasing(model_types[debiasing],
                                                                 model_grids[debiasing], X_train, y_train,
                                                                 X_test, s, verbose=True,
                                                                 pickle_path=prep_path + ".data_versions", seed=seed)

        pattern = re.compile(rf"mitigated_{debiasing}_{flv}_{s}_{d}_{n_samples}_best_model\((\d+)\)_seed{seed}$")
        full_path = None

        for filename in os.listdir("../out/models/"):
            match = pattern.match(filename)
            if match:
                best_index = int(match.group(1))
                trans_model = model_versions[best_index]
                print("Best data version", best_index)
                full_path = os.path.join("../out/models/", filename)

        if full_path is not None:
            with open(full_path, "rb") as f:
                mitigated = pickle.load(f)
                mitigated_params = mitigated.get_params()
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
                    "model_path": f"../out/models/{debiasing}_{flv}_{s}_{d}_{n_samples}_v{i}_seed{seed}",
                    "custom_sensitives": X_train[s],
                    "random_state": seed,
                }
                mit, mit_params, search = train_best_fair_model_parallel(**search_params)
                if search.best_score_ > best_search_score:
                    best_search = search
                    best_search_score = search.best_score_
                    best_index = i
                    trans_model = model_versions[best_index]

            mitigated = best_search.best_estimator_
            mitigated_params = best_search.best_params_

            with open(f"../out/models/mitigated_{debiasing}_{flv}_{s}_{d}_{n_samples}_best_model({best_index})_seed{seed}", "wb") as f:
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
                estimator=LogisticRegression(), constraints=DemographicParity()), #TODO: add diff_bound
            param_grid=model_grids[debiasing],
            X_train=X_train,
            y_train=y_train,
            sensitive_col=s,
            model_path=mitigated_model_path,
            cv=5,
            random_state=seed,
            verbose=True
        )

        mitigated_accuracy_dict, mitigated_fairness_dict = test_model(mitigated, s, X_test, y_test)

    # for key in unmitigated_accuracy_dict.keys(): print(key,unmitigated_accuracy_dict[key], mitigated_accuracy_dict[key])
    # for key in unmitigated_fairness_dict.keys(): print(key,unmitigated_fairness_dict[key], mitigated_fairness_dict[key])
    print("\tDP-diff", unmitigated_fairness_dict['DP-diff'], "-->", mitigated_fairness_dict['DP-diff'])
    print("\tDP-ratio", unmitigated_fairness_dict['DP-ratio'], "-->", mitigated_fairness_dict['DP-ratio'])

    return mitigated, mitigated_params, mitigated_accuracy_dict, mitigated_fairness_dict, trans_model

def discriminator_training(X_train, y_train, X_test, y_test, discr_model, debiasing, flv, s, bias, d, n_samples, var=False):
    print(">>> (3) Training discriminator ({}) on new labels...".format(
                            model_types[discr_model].__name__))
    discr_model_path = f"../out/models/discr_{discr_model}_{debiasing}_{flv}-{s}-{d}_{n_samples}_prob_{bias}.model"
    if var: discr_model_path = f"../out/models/var_discr_{discr_model}_{debiasing}_{flv}-{s}-{d}_{n_samples}_prob_{bias}.model"

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

    discriminator, discriminator_params = train_best_model(discriminator_halving_params, X_train,
                                                           y_train, discr_model_path,
                                                           verbose=True)

    discr_accuracy_dict, discr_fairness_dict = test_model(discriminator, s, X_test, y_test)

    print_metrics_table(keys_of_dict(discr_accuracy_dict,
                                     ['accuracy', 'auc', 'f1_score', 'precision', 'recall',
                                      'matthew_coeff']))
    print_metrics_table(discr_fairness_dict)


    return discriminator, discriminator_params, discr_accuracy_dict, discr_fairness_dict

def get_probing_sets(probing_sets_dict, dim, d, sensitives, sensitives_maps, verbose=True):
    # TODO: shuffle
    from src.utils.data import generate_synth_dataset, sensitive_encoding
    probing_sets = {}

    if verbose: print(">>> Generating synthetic probing sets")
    for bias, params in probing_sets_dict.items():
        if verbose: print("\t"+bias)
        # FIXME: path
        X, Y, _ = generate_synth_dataset(dim, params, dataset_name=f"{d}_{bias}.probing", seed=420) #TODO: try also the same seed of the generated dataset
        # pre-process probing set (with the same preprocessor as d)
        # load d preprocessor
        with open(config.preprocessors_path.format(d), "rb") as f:
            dataset_preprocessor = pickle.load(f)
        if "bias" in d:
            X['Q'] = X['Q'].astype("object")
            sensitives_maps['A'] = {1: 1, 0: 0}
        X = X.reset_index(drop=True)
        Y= Y.reset_index(drop=True)
        # preprocess X
        X = dataset_preprocessor.transform(X)
        # encode sensitive and labels
        #FIXME: encoding
        Y=label_encoding(Y, map={1: 1, 0: 0}, verbose=False) # synth
        sensitive_encoding(X, sensitives, sensitives_maps, verbose=False)
        probing_sets[bias]= (X, Y)

        with open(config.probing_sets_path + f"/{d}_probing-set-{bias}.preprocessed", "wb") as f:
            pickle.dump((X,Y), f)

    return probing_sets

def fix_new_dataset(X_train, y_train, mode="RandomOverSampling"):
    if mode == "RandomOverSampling":
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_train = X_train.reset_index(drop=True)
        #y_train = y_train.reset_index(drop=True)
    return X_train, y_train

def get_knowledge_of_debiasing(discriminator, X_probing, explainer_path, background=None, mode="kernel"):
    shap_values = None
    expected_value = None
    explainer = None

    if os.path.exists(explainer_path):
        with open(explainer_path, "rb") as f:
          artifact = pickle.load(f) # FIXME: avoid for efficiency purposes
        print("Shap values are already computed.", explainer_path)
    else:
        if mode=="tree" and isinstance(discriminator, XGBClassifier):
            # treeSHAP
            explainer = shap.TreeExplainer(discriminator)
            shap_values = explainer.shap_values(X_probing)
            expected_value = explainer.expected_value
        elif mode=="kernel":
            background = shap.sample(background, 100)  # also for preprocessing should be valid

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
            expected_value = explainer.expected_value

        artifact = {
            "shap_values": shap_values,
            #"X": X_probing,
            "background": background,
            "expected_value": expected_value,
            "feature_names": X_probing.columns.tolist()
            #"explainer": explainer
        }

        with open(explainer_path, "wb") as f:
            pickle.dump(artifact, f)
        print(">>> Shap explainer saved to disk", explainer_path)
    return artifact

def model_probing(mitigated, X_probing, s, trans_model, debiasing):
    print(f">>> Probing mitigated {debiasing} model...")

    if isinstance(mitigated, ExponentiatedGradient):
        y_u = mitigated.predict(X_probing, random_state=42)
    elif debiasing == 'prl':
        if trans_model is None: raise ValueError("trans_model is None")
        #mitigated_params = [trans_model.get_params(), mitigated_params] #FIXME: non è un fixme, solo per note
        y_u = mitigated.predict(trans_model.transform(X_probing)) #FIXME: non è un fixme, solo per note
    else:
        y_u = mitigated.predict(X_probing, sensitive_features=X_probing[s], random_state=42)

        # y_u = mitigated.predict(X_probing, random_state=42) if isinstance(mitigated, ExponentiatedGradient)  else mitigated.predict(X_probing, sensitive_features=X_probing[s], random_state=42)
    return y_u

def disagreement(predictions):
    return (np.any(predictions != predictions[0], axis=0)).astype(int)

def model_probing_var(mitigated_models_dict, X, s, debiasing):
    preds_var = []
    for key, value in mitigated_models_dict.items():
        mitigated_model_var = value["mitigated_var"]
        mitigated_trans_model_var = value["trans_model_var"]
        mitigated_model_var_preds = model_probing(mitigated_model_var, X, s, mitigated_trans_model_var, debiasing)
        preds_var.append(np.array(mitigated_model_var_preds))
    return disagreement(np.array(preds_var))

def label_generation(y_mitigated, y_probing, data_info, target_encoding):
    from src.exps import explain_fav_unfav_mapping
    # FAV/UNFAV MAPPING
    fav_unfav_mapping = data_info['fairness']['fav_labels_mapping']
    print("\tFAV-UNFAV mapping", fav_unfav_mapping)
    print("\tFAV-UNFAV meaning", explain_fav_unfav_mapping(fav_unfav_mapping, target_encoding, cat=True))

    # ENCODE y_mitigated
    y_u_encoded = label_encoding(pd.Series(y_mitigated).astype(str), map=fav_unfav_mapping)
    # ENCODE     y_probing
    y_probing_encoded = label_encoding(pd.Series(y_probing).astype(str), map=fav_unfav_mapping)
    # NEW LABELS: Y'
    y_prime = [1 if y_i_u != y_i else 0 for y_i_u, y_i in zip(y_u_encoded, y_probing_encoded)]

    return np.array(y_prime), y_u_encoded, y_probing_encoded

def print_label_gen_stats(stats):
    print("\nStats of the corrected instances\n-------------------------------------".upper())

    print(f"Samples corrected:\t {(stats['n_corrected']/stats['total_samples'])*100} %")
    print(f"> from protected:\t {(stats['protected_corrected']/stats['n_corrected'])*100} %")
    print(f"> from unprotected:\t {(stats['unprotected_corrected']/stats['n_corrected'])*100} %")
    print()
    print("Per group".upper())
    print(f"> protected:\t\t {stats["protected_corrected"]/stats['n_protected']*100} %")
    print(f"> unprotected:\t\t {stats["unprotected_corrected"]/stats['n_unprotected']*100} %")

    print()

def stats_of_label_gen(X, labels, s, verbose=False):
    # TODO: add other stats, maybe SFAV -> FAV and viceversa, more granular
    stats = {}

    protected_mask = X[s] == 1
    unprotected_mask = X[s] == 0

    labels_protected = labels[protected_mask]
    labels_unprotected = labels[unprotected_mask]

    stats["total_samples"] = labels.shape[0]
    stats["n_protected"] = X[protected_mask].shape[0]
    stats["n_unprotected"] = X[unprotected_mask].shape[0]
    stats["n_corrected"] = labels[labels==1].shape[0]
    stats["protected_corrected"] = labels_protected[labels_protected==1].shape[0]
    stats["unprotected_corrected"] = labels_unprotected[labels_unprotected==1].shape[0]

    if verbose: print_label_gen_stats(stats)

    return stats

def main():
    parser = argparse.ArgumentParser(description="Synth data experiments.")
    parser.add_argument("d", type=str, help="Name of one of the available datasets in the repo")
    parser.add_argument("-ms", "--max_size", type=int, help="Limit row numbers for the specified dataset", default=None)
    parser.add_argument("-np", "--n_probing", type=int, help="Number of samples used for probing the mitigated model.", default=5000)
    args = parser.parse_args()

    d = args.d
    max_size = args.max_size
    n_probing = args.n_probing

    from_exp = 1
    exp_no = 1
    results_df = pd.DataFrame(columns=res_cols)
    checkpoint_exists = False


    # load data
    X, Y, data_info, sensitives, sensitives_maps, target_encoding = data_loading(d, max_size=max_size)
    n_samples = X.shape[0]

    # check if a checkpoint already exists
    if os.path.exists(config.checkpoint_path.format(d + f"_{n_samples}")):
        checkpoint_exists = True
        results_df = pd.read_pickle(config.checkpoint_path.format(d + f"_{n_samples}"))
        from_exp = len(results_df)
        print("Checkpoint containing {} tests loaded.".format(from_exp))

    for s in sensitives:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=True,
                                                            stratify=Y)
        # index reset
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        for unm_model in unm_model_pool:
            start_time_unmitigated = time()
            unmitigated, unmitigated_params, unmitigated_accuracy_dict, unmitigated_fairness_dict = unmitigated_training(
                unm_model, s, X_train, y_train, X_test, y_test, d, n_samples)
            end_time_unmitigated = time()

            for flv in fairness_levels.keys():
                for debiasing in debiasing_pool:

                    start_time_debiasing = time()
                    mitigated, mitigated_params, mitigated_accuracy_dict, mitigated_fairness_dict, trans_model = mitigated_training(
                        X_train, y_train, X_test, y_test, unmitigated, unmitigated_fairness_dict, debiasing, flv, s, d, n_samples)
                    end_time_debiasing = time()

                    # train different mitigated models with different seeds (on the search for now)
                    mitigated_models_var = {}
                    for seed in debiasing_seeds:
                        print(seed, flv, debiasing)

                        mitigated_var, mitigated_params_var, mitigated_accuracy_dict_var, mitigated_fairness_dict_var, trans_model_var = mitigated_training(
                            X_train, y_train, X_test, y_test, unmitigated, unmitigated_fairness_dict, debiasing, flv, s,
                            d, n_samples, seed=seed)

                        mitigated_models_var[seed] = {
                            "mitigated_var": mitigated_var,
                            "mitigated_params_var": mitigated_params_var ,
                            "mitigated_accuracy_dict_var": mitigated_accuracy_dict_var,
                            "mitigated_fairness_dict_var": mitigated_fairness_dict_var,
                            "trans_model_var": trans_model_var
                        }

                    # get probing sets
                    probing_sets = get_probing_sets(probing_sets_dict, n_probing, d, sensitives, sensitives_maps, verbose=True)

                    for bias in probing_sets.keys():
                        print("PROBING SET: ", bias)
                        X_probing, y_probing = probing_sets[bias]
                        X_probing = X_probing[X.columns]  # force columns to be in the same order
                        # probing the mitigated model
                        y_mitigated = model_probing(mitigated, X_probing, s, trans_model, debiasing)

                        # probe for variance
                        y_var = model_probing_var(mitigated_models_var, X_probing, s, debiasing)

                        # LABEL GENERATION
                        y_prime, y_u_encoded, y_probing_encoded = label_generation(y_mitigated, y_probing, data_info, target_encoding)
                        # STATS OF D'
                        new_dataset_stats = stats_of_label_gen(X_probing, y_prime, s, verbose=True)
                        # PREPARE TRAINING DATA FOR DISCRIMINATOR
                        X_prob_train, X_prob_test, y_prime_train, y_prime_test = train_test_split(X_probing, y_prime,
                                                                                                  test_size=test_size,
                                                                                                  random_state=42,
                                                                                                  shuffle=True,
                                                                                                  stratify=y_prime)
                        X_prob_train = X_prob_train.reset_index(drop=True)
                        X_prob_test = X_prob_test.reset_index(drop=True)

                        # PREPARE TRAINING DATA FOR DISCRIMINATOR VAR
                        var_classes, var_counts = np.unique(y_var, return_counts=True)
                        can_train_var_discriminator = (
                                len(var_classes) > 1 and var_counts.min() >= 2
                        )

                        if can_train_var_discriminator:
                            X_prob_train_var, X_prob_test_var, y_var_train, y_var_test = train_test_split(X_probing, y_var,
                                                                                                      test_size=test_size,
                                                                                                      random_state=42,
                                                                                                      shuffle=True,
                                                                                                      stratify=y_var)
                            X_prob_train_var = X_prob_train_var.reset_index(drop=True)
                            X_prob_test_var = X_prob_test_var.reset_index(drop=True)



                        # APPLY OVERSAMPLING TO D'
                        #X_prob_train_os, y_prime_train_os = fix_new_dataset(X_prob_train, y_prime_train)

                        # DISCRIMINATOR TRAINING
                        for discr_model in discr_model_pool:
                            if checkpoint_exists:
                                if exp_no <= from_exp:
                                    # FIXME: add check
                                    exp_no += 1
                                    continue
                            start_time_discriminator = time()

                            discriminator, discriminator_params, discr_accuracy_dict, discr_fairness_dict = discriminator_training(
                                X_prob_train, y_prime_train, X_prob_test, y_prime_test, discr_model, debiasing,
                                flv, s, bias, d, n_samples)
                            end_time_discriminator = time()

                            # GETTINKG KNOWLEDGE ABOUT DEBIASING
                            explainer_path = f"../out/models/shap_{discr_model}_{unm_model}-{debiasing}-{flv}-{s}-{d}_{n_samples}_prob_{bias}.model"

                            artifacts = get_knowledge_of_debiasing(discriminator, X_probing, explainer_path, mode="tree")

                            # TRAIN DISCRIMINATOR VAR
                            if can_train_var_discriminator: # if there is no variance
                                discriminator_var, discriminator_params_var, discr_accuracy_dict_var, discr_fairness_dict_var = discriminator_training(
                                X_prob_train_var, y_var_train, X_prob_test_var, y_var_test, discr_model, debiasing,
                                flv, s, bias, d, n_samples, var=True)
                                # SHAP VALUES FOR VAR
                                explainer_path_var = f"../out/models/var_shap_{discr_model}_{unm_model}-{debiasing}-{flv}-{s}-{d}_{n_samples}_prob_{bias}.model"
                                artifacts_var = get_knowledge_of_debiasing(discriminator_var, X_probing,
                                                                           explainer_path_var, mode="tree")

                            else: discriminator_var, discriminator_params_var, discr_accuracy_dict_var, discr_fairness_dict_var, artifacts_var = None, None, None, None, None


                            end_time = time()
                            # RESULTS
                            results_df.loc[exp_no] = {
                                "total_time": end_time - start_time_unmitigated,  # for one whole run of exp #
                                # Data Info
                                "dataset": d,
                                "n_samples": n_samples,
                                "sensitive": s,
                                "test_size": test_size,
                                "probing_set": bias,
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
                                "debiasing_combined_score": combined_score(mitigated_accuracy_dict,
                                                                           mitigated_fairness_dict),
                                "debiasing_exp_time": end_time_debiasing - start_time_debiasing,
                                # Discriminator Model
                                "discr_model_type": model_types[discr_model].__name__,
                                "discr_model_accuracy": discr_accuracy_dict,  # JSON with objs
                                "discr_model_fairness": discr_fairness_dict,  # JSON with objs,
                                "discr_model_params": discriminator_params,
                                "discr_model_exp_time": end_time_discriminator - start_time_discriminator,
                                "y_u": y_u_encoded,  # predictions of discriminator
                                "y_probing": y_probing_encoded,  # ground_truths
                                "y_prime": y_prime,  # new labels
                                # XAI
                                "new_dataset_stats": new_dataset_stats,
                                # Discriminator var model
                                "discriminator_var_accuracy": discr_accuracy_dict_var,
                                "discriminator_var_fairness": discr_fairness_dict_var,
                                "discriminator_var_params": discriminator_params_var,
                                "y_var": y_var,
                                "mitigated_models_var": mitigated_models_var
                            }
                            exp_no += 1
                            # results_df.to_pickle(config.checkpoint_path.format(d))
                            results_df.to_pickle(config.checkpoint_path.format(d + f"_{n_samples}"))
        # save results to disk and delete checkpoints
        results_df.to_pickle(config.results_path.format(d + f"_{n_samples}"))
        if os.path.exists(config.checkpoint_path.format(d + f"_{n_samples}")): os.remove(config.checkpoint_path.format(d + f"_{n_samples}"))

if __name__ == "__main__":
    try:
        main()
        send_notification("Completed", title="✅")
        memory.clear()
    except Exception as e:
        loc = get_compact_traceback()
        send_notification(f"{type(e).__name__}: {e}\n{loc}", title="❌")
        print()
        raise

import atexit

def on_exit():
    send_notification("Script terminated (possibily due to crashes).", title="💥")
    print()

atexit.register(on_exit)

