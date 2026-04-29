import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import set_config
set_config(transform_output='pandas')

# TODO: synth data generation

# LOAD DATASET
def load_data(dataset, datasets_info, path):
    """
    Load the specified dataset from disk to memory.
    :param dataset: name of the dataset (available in the json file and path).
    :param datasets_info: dictionary containing all the info about the available datasets.
    :param path: general path where the actual datasets are stored.
    :return: X_train, y_train, X_test, y_test
    """
    # TODO: [!LOW] handle other formats
    X_train = None
    y_train = None

    X_test = None
    y_test = None

    dataset_info = datasets_info[dataset]
    dataset_info_settings = dataset_info["settings"]
    dataset_info_preprocessing = dataset_info["preprocessing"]

    # LOAD TRAIN/ALL DATA + LABELS
    X_train = pd.read_csv("{}/{}/{}".format(path, dataset, dataset_info_settings["train_path"]), names=dataset_info_preprocessing['feature_names'], index_col=None, **dataset_info_settings["train_load_params"])

    y_train = X_train.pop(X_train.columns[dataset_info_preprocessing["target_index"]])

    # LOAD TEST DATA IF EXISTS
    if dataset_info_settings["test_path"] is not None:
        X_test = pd.read_csv("{}/{}/{}".format(path, dataset, dataset_info_settings["test_path"]), names=dataset_info_preprocessing['feature_names'], index_col=None, **dataset_info_settings["test_load_params"])
        y_test = X_test.pop(X_test.columns[dataset_info_preprocessing["target_index"]])

    return X_train, y_train, X_test, y_test

def create_pipeline_from_json(dict_pipeline):
    """
    It creates a pre-processing pipeline from the info specified in the json file.
    :param dict_pipeline: dictionary containing the steps of the pipeline.
    :return: Pipeline object.
    """
    pip = []

    for step in dict_pipeline.keys():
        params = dict_pipeline[step]['params']
        if 'dtype' in params and isinstance(params['dtype'], str):
            if params['dtype'] == "np.int64": params['dtype'] = np.int64
        #elif params['dtype'] == "float": params['dtype'] = float

        if step == "imputer": pip.append(SimpleImputer(**params))
        if step == "onehot": pip.append(OneHotEncoder(**params))
        if step == "scaler":
            if dict_pipeline[step]["type"] == "min_max": pip.append(MinMaxScaler(**params))
            if dict_pipeline[step]["type"] == "standard": pip.append(StandardScaler(**params))
    pipeline = make_pipeline(*pip)
    return pipeline

def label_encoding(labels, map=None, verbose=False):
    """
    Encodes labels.
    :param labels: labels to be encoded.
    :param map: custom mapping;
    :param verbose: if want to print the mapping.
    :return: encoded labels.
    """
    # standard mapping
    if map is None:
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        mapping = {}
        for c in le.classes_:
            mapping[c] = le.transform([c]).tolist()[0]
        if verbose: print("\t {}".format(mapping))
        return pd.DataFrame(encoded, columns=[labels.name])
    # custom mapping
    else:
        if verbose: print("\t {}".format(map))
        return labels.map(map)

def sensitive_encoding(X, sensitives, maps, verbose=False):
    """
    Encodes sensitives attributes.
    :param X: dataset containing the sensitives columns.
    :param sensitives: list of sensitives attributes (column names).
    :param maps: (dictionary) sensitive attributes mappings.
    :param verbose: if want to print the mapping.
    :return: encoded dataset.
    """
    for s in sensitives: X[s] = label_encoding(X[s], maps[s], verbose=verbose)
    return X

def preprocessing(data, sensitives, dataset_info):
    """
    Preprocesses data.
    :param data: data to be preprocessed.
    :param sensitives: list containing sensitives attributes (column names).
    :param dataset_info: dictionary containing all the info about the dataset.
    :return: preprocessed data.
    """
    info_preprocessing = dataset_info["preprocessing"]

    # DROP SPECIFIED COLUMNS #FIXME: maybe change
    if "drop_cols" in info_preprocessing: data = data.drop(columns=info_preprocessing["drop_cols"])
    # REPLACE CUSTOM MISSING VALUE WITH STANDARD NaN
    if info_preprocessing["custom_mv"] is not None: data=data.replace(info_preprocessing["custom_mv"], np.nan)

    # CREATE AND APPLY SPECIFIED PIPELINE IN JSON CONFIG FILE
    num_cols = data.select_dtypes(include='number').columns.difference(sensitives)
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.difference(sensitives)
    bool_cols = data.select_dtypes(include=['bool','boolean']).columns.difference(sensitives)

    cat_pipeline = create_pipeline_from_json(dataset_info['preprocessing']['pipeline']["cat_pip"])
    num_pipeline = create_pipeline_from_json(dataset_info['preprocessing']['pipeline']["num_pip"])
    bool_pipeline = create_pipeline_from_json(dataset_info['preprocessing']['pipeline']["bool_pip"])

    # pipeline creation
    preprocessor = ColumnTransformer(transformers=[('cat', cat_pipeline, cat_cols),
                                                   ('num', num_pipeline, num_cols),
                                                   ('bool', bool_pipeline, bool_cols)
                                                   ],
                                         remainder='passthrough',
                                         verbose_feature_names_out=False)
    # apply pipeline
    prep_data = preprocessor.fit_transform(data)

    # re-order columns
    feature_names = preprocessor.get_feature_names_out()
    final_columns = []

    for col in data.columns:
        related_cols = [f for f in feature_names if f.startswith(col + '_') or f == col]
        final_columns.extend(related_cols)
    prep_data = prep_data[final_columns]

    return prep_data
