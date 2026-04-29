# Paths
datasets_path = "../rsc/datasets"
datasets_info_path = "../rsc/datasets.json"
datasets_md_dict_path = "../rsc/datasets_md_dict.json"
md_path = "../doc/datasets.md"
checkpoint_path = "../out/checkpoints_{}.df"
results_path = "../out/results_{}.df"

# Vanilla Results DataFrame
res_cols = [
    "total_time", # for one whole run of exp
    # Data Info
    "dataset",
    "sensitive",
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
    "y_prime"
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
    "n_prototypes": [5, 15, 30],
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