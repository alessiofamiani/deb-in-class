# TODO: salvare gli scaler

# ------------ SAVE DATA INFO MARKDOWN  -----------------
def update_dataset_md(d, datasets_md_dict, datasets_md_dict_path, dataset, data_info):
    from utils.test import explain_fav_unfav_mapping
    datasets_md_dict[d] = {
        "name": dataset.metadata.name,
        "repository_url": dataset.metadata.repository_url,
        "summary": dataset.metadata.abstract,
        "sensitive_attributes":  list(data_info['fairness']['sensitives'].keys()),
        "target": dataset.metadata.target_col[0],
        "size": "{} x {}".format(dataset.metadata.num_instances, dataset.metadata.num_features),
        "fav_unfav_mapping": dict(sorted(explain_fav_unfav_mapping(data_info['fairness']['fav_labels_mapping'], data_info['preprocessing']['target_encoding'], cat=True).items())),
    }

    with open(datasets_md_dict_path, "w") as f:
     json.dump(datasets_md_dict, f, indent=2)
    return datasets_md_dict[d]
def save_dataset_md(md_path, datasets_md_dict):
    from utils.test import mapping_to_string
    header = "# Datasets \n| Name | Summary | Sensitive attributes | Target | Size (instances x features ) | Fav/Unfav mapping |\n|------|---------|----------------------|--------|------|----|"
    lines = [header]

    datasets_md_dict = dict(sorted(datasets_md_dict.items()))

    # Loop over datasets and create table rows
    for key, data in datasets_md_dict.items():
        name = data.get("name", "")
        url = data.get("repository_url", "")
        summary = data.get("summary", "").replace("\n", " ")  # remove line breaks
        sensitive = ", ".join(data.get("sensitive_attributes", ""))
        target = data.get("target", "")
        size = data.get("size", "")
        fav_unfav_mapping = mapping_to_string(data.get("fav_unfav_mapping", ""))

        row = f"| **[{name}]({url})** | {summary} | {sensitive} | {target} | {size} | {fav_unfav_mapping} |"
        lines.append(row)

    # Write to markdown file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\tMarkdown file saved at {md_path} ✅")

import os
import json
import config
import pickle
import argparse
from ucimlrepo import fetch_ucirepo

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing.")
    parser.add_argument("d", type=str, help="Name of one of the available datasets in the repo")

    args = parser.parse_args()

    d = args.d

    # ------------ LOAD DATASET INFOS -----------------
    f = open(config.datasets_info_path, "r")
    datasets_info = json.load(f)
    f.close()

    if d not in datasets_info: raise ValueError("Dataset not supported.")
    data_info = datasets_info[d]
    sensitives = list(data_info['fairness']['sensitives'].keys()) # names of the columns
    sensitives_maps = data_info['fairness']['sensitives']# mapping for multiple values subst. for sensitives
    target_encoding = data_info['preprocessing']['target_encoding']

    print(d)
    print(">>> Dataset Info")
    print("\tsensitives:", sensitives)

    print("\t\t", sensitives_maps)
    print("\ttarget:", target_encoding)

    # ┌───────────────────────────────┐
    # │          DATA LOADING         │
    # └───────────────────────────────┘
    print(">>> Data loading...")
    if os.path.exists(config.datasets_path + "/{}.dataset".format(d)):
        with open(config.datasets_path + "/{}.dataset".format(d), "rb") as f:
            X, Y = pickle.load(f)
        print("\tDataset loaded from disk.")
    else:
        # Download from UCI repo
        if d == "adult": dataset = fetch_ucirepo(id=2)
        elif d == "bank_marketing": dataset = fetch_ucirepo(id=222)
        elif d == "credit_card": dataset = fetch_ucirepo(id=350)
        else: raise ValueError("Dataset not supported.")

        # save dataset, it doesn't work on python 3.9 (but in 3.14)
        # with open(config.datasets_path + "/{}.dataset".format(d), "wb") as f:
        #    pickle.dump(dataset, f)
        # print("\tDataset cached to disk.")

        # save md
        datasets_md_dict = {}

        if os.path.exists(config.datasets_md_dict_path):
            f = open(config.datasets_md_dict_path, "r")
            datasets_md_dict = json.load(f)
            f.close()
            if d not in datasets_md_dict: update_dataset_md(d, datasets_md_dict, config.datasets_md_dict_path, dataset, data_info)
        else: update_dataset_md(d, datasets_md_dict, config.datasets_md_dict_path, dataset, data_info)
        save_dataset_md(config.md_path, datasets_md_dict)

        X = dataset.data.features.copy()
        Y = dataset.data.targets.copy()

    # ┌───────────────────────────────┐
    # │       ️ PRE-PROCESSING        │
    # └───────────────────────────────┘

    print(">>> Pre-processing...")
    Y = Y.astype(str)
    if 'rename_map' in data_info['preprocessing']:
        X = X.rename(columns=data_info['preprocessing']['rename_map'])

    if d == "adult": Y[dataset.metadata.target_col[0]] = Y[dataset.metadata.target_col[0]].str.replace('.', '',
                                                                                                       regex=False)
    if 'cols_encoding' in data_info['preprocessing']:
        X[list(data_info['preprocessing']['cols_encoding'].keys())] = X[data_info['preprocessing']['cols_encoding'].keys()].astype("str")

        for col in data_info['preprocessing']['cols_encoding'].keys():

            X[col] = X[col].map(data_info['preprocessing']['cols_encoding'][col])
    if d == 'bank_marketing':
        X[list(data_info['preprocessing']['cols_encoding'].keys())] = X[
            data_info['preprocessing']['cols_encoding'].keys()].astype("boolean")
        X.loc[:, 'contact'] = X['contact'].fillna(X['contact'].mode()[0])

    if "bias" in d:
        X['Q'] = X['Q'].astype("object")
        sensitives_maps['A'] = {1: 1, 0: 0} # forcing cause otherwise is not working

    from utils.data import preprocessing, label_encoding, sensitive_encoding
    print("\t> ----- Pre-processing")
    X_prep = preprocessing(X, sensitives, data_info, d)
    if d == "adult" or d == "bank_marketing" or d == "credit_card":
        Y_prep = label_encoding(Y[dataset.metadata.target_col[0]], map=target_encoding, verbose=True)
    else:
        #Y = Y.to_frame()
        Y_prep = label_encoding(Y, map=target_encoding, verbose=True) # synth
    sensitive_encoding(X_prep, sensitives, sensitives_maps, verbose=True) # FIXME: synth

    with open(config.datasets_path + "/{}.preprocessed".format(d), "wb") as f:
        pickle.dump((X_prep, Y_prep), f)
    print("\tPre-processed data cached to disk.")
if __name__ == "__main__":
    main()