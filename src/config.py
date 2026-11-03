# Paths
datasets_path = "../rsc/datasets"
datasets_info_path = "../rsc/datasets.json"
datasets_md_dict_path = "../rsc/datasets_md_dict.json"
md_path = "../doc/datasets.md"
checkpoint_path = "../out/checkpoints_{}.df"
results_path = "../out/results_{}.df"
probing_sets_path = "../out/data"
preprocessors_path = "../rsc/preprocessors/{}.preprocessor"

# Synth data gen
probing_sets_dict = {
    "no_bias": {},
    "historically_biased_y": {"l_y":10},
    "historically_biased_r": {"l_h_r":10},
    "historically_biased_q": {"l_q":10, "l_h_q":10},
    "measurement_biased_y": {"l_m_y":10, "l_m":10},
    #"representational_biased": {"p_u": 0.5}
}
"""
probing_sets_dict = {
 "no_bias": {"l_y":4, "l_m_y":0, "thr_supp":1, "l_h_r":1.5,  "l_h_q":1, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":5, "l_r_q":0, "l_m_y_non_linear":False},
 "historically_biased_y": {"l_y":4, "l_m_y":0, "thr_supp":1, "l_h_r":10,  "l_h_q":1, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":5, "l_r_q":0, "l_m_y_non_linear":False},
 "measurement_biased_y": {"l_y":4, "l_m_y":10, "thr_supp":1, "l_h_r":10,  "l_h_q":1, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":5, "l_r_q":0, "l_m_y_non_linear":False},
}"""

debiasing_seeds = [45, 111, 234, 897, 10]#, 96, 188, 911, 6, 3204]

# Vanilla Results DataFrame
res_cols = [
    "total_time", # for one whole run of exp
    # Data Info
    "dataset",
    "n_samples",
    "sensitive",
    "probing_set",
    "test_size",
    # Unmitigated Model
    "unm_model_type",
    "unm_model_accuracy", # JSON with objs
    "unm_model_fairness", # JSON with objs
    "unm_model_params",
    "unm_model_exp_time",
    # Mitigated Model (Debiasing)
    "debiasing_lv",
    "debiasing_type",
    "debiasing_accuracy", # JSON with objs
    "debiasing_fairness", # JSON with objs
    "debiasing_params",
    "debiasing_combined_score",
    "debiasing_exp_time",
    # Discriminator Model
    "discr_model_type",
    "discr_model_accuracy", # JSON with objs
    "discr_model_fairness", # JSON with objs
    "discr_model_params",
    "discr_model_exp_time",
    "y_u",
    "y_probing",
    "y_prime",
    "new_dataset_stats",
    # Discriminator var model
    "discriminator_var_accuracy",
    "discriminator_var_fairness",
    "discriminator_var_params",
    "y_var",
    "mitigated_models_var"
    # Models -> disk,
    # XAI structures, explainers and plots -> disk
    # or a dictionary containing all "pickables" files
]

# Grids for grid searches
model_grids = {
    "lr":   [
    # lbfgs, newton-cg, sag → penalty l2
    {
        "penalty": ["l2"],
        "C": [0.1, 1.0],
        "solver": ["lbfgs"],
        "max_iter": [1000],
        "tol": [1e-4, 1e-2],
        "fit_intercept": [True],
        "class_weight": ["balanced", None],
    },
    # liblinear → l1/l2 (l1_ratio non serve)
    {
        "penalty": ["l1", "l2"],
        "C": [0.1, 1.0],
        "solver": ["liblinear"],
        "dual": [False],
        "fit_intercept": [True],
        "intercept_scaling": [1.0],
        "class_weight": ["balanced"],
        "tol": [1e-4],
    },
    # saga → supporta l1, l2, elasticnet (l1_ratio solo per elasticnet)
    {
        "penalty": ["elasticnet"],
        "C": [0.1, 1.0],
        "solver": ["saga"],
        "l1_ratio": [0.0, 0.5],
        "fit_intercept": [True],
        "class_weight": ["balanced"],
        "tol": [1e-3],
        "max_iter": [1000],
    }],
    "to":{
        "constraints": ["demographic_parity"],
        "objective": ["accuracy_score", "balanced_accuracy_score"],
        "predict_method": ["predict_proba"],
        "grid_size": [100, 500, 1000],
        "flip": [False],
        #"tol": [None, 1e-3, 1e-2],
        "prefit": [True]
    }
#{
 #       "constraints": ["equalized_odds"],
 #       "objective": ["accuracy_score", "balanced_accuracy_score"],
  #      "predict_method": ["predict"],
   #     "grid_size": [100, 500, 1000],
    #    "flip": [False],

     #   "prefit": [True]
    #}
    ,
    "xgb": {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
},
    "mlp": {
    'hidden_layer_sizes': [(50,), (100,), (50,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [300]
},
    "eg": {
        #"estimator": [LogisticRegression, XGBClassifier],
        #"constraints": [DemographicParity, EqualizedOdds, ErrorRateParity],
        #"eps": [0.01, 0.05],
        "eta0": [0.5, 2.0],
        "max_iter": [100],
        "nu": [None, 1e-6],
        "run_linprog_step": [True, False]
    },
    "prl": {
    #"n_prototypes": [5, 15, 30], # OLD
    "n_prototypes": [2, 3, 4], #for synth?
    #"fairness_weight": [0.25, 0.5, 75.0],
    #"target_weight": [0.75, 0.5, 0.25],
    "reconstruct_weight": [1.0]
}
}

fairness_lv = {
    "low": {
        "to": {
            "tol": [1e-1],
            #"grid_size": [100]
        },
        "eg": {
            "eps": [0.3]
        },
        "prl": {
            "fairness_weight": [0.05],
        }
    },
    "medium": {
        "to": {
            "tol": [1e-2],
            #"grid_size": [500]
        },
        "eg": {
            "eps": [0.1]
        },
        "prl": {
            "fairness_weight": [1.0],
        }
    },
    "high": {
        "to": {
            "tol": [1e-3],
            #"grid_size": [1000]
        },
        "eg": {
            "eps": [0.01]
        },
        "prl": {
            "fairness_weight": [10.0],
        }
    }
}