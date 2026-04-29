import matplotlib.pyplot as plt
def test_accuracy(predictions, labels, predictions_prob=None):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay

    acc_dict = {}
    acc_dict['accuracy_score'] = accuracy_score(labels, predictions)
    acc_dict['f1_score'] = f1_score(labels, predictions)
    acc_dict['precision'] = precision_score(labels, predictions, zero_division=0)
    acc_dict['recall'] = recall_score(labels, predictions)
    acc_dict['matthew_coeff'] = matthews_corrcoef(labels, predictions)
    if predictions_prob is not None:
        acc_dict['roc_objs'] = fpr, tpr, thresholds = roc_curve(labels, predictions_prob)
        acc_dict['auc'] = roc_auc_score(labels, predictions_prob)
    else:
        acc_dict['auc'] = None
        acc_dict['roc_objs'] = None

    acc_dict['cm'] = confusion_matrix(predictions, labels)
    acc_dict['cm_plot'] = ConfusionMatrixDisplay.from_predictions(
                labels,
                predictions,
                normalize='true'
            )
    plt.clf()
    plt.close()
    return acc_dict

def test_fairness(predictions, labels, sens):
    from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio, equal_opportunity_difference, equal_opportunity_ratio

    fairness_dict = {}
    fairness_dict['DP-diff'] = demographic_parity_difference(labels, predictions, sensitive_features=sens)
    fairness_dict['DP-ratio'] = demographic_parity_ratio(labels, predictions, sensitive_features=sens)
    fairness_dict['EOpp-diff'] = equal_opportunity_difference(labels, predictions, sensitive_features=sens)
    fairness_dict['EOpp-ratio'] = equal_opportunity_ratio(labels, predictions, sensitive_features=sens)
    fairness_dict['EOdds-diff']= equalized_odds_difference(labels, predictions, sensitive_features=sens)
    fairness_dict['EOdds-ratio'] = equalized_odds_ratio(labels, predictions, sensitive_features=sens)

    # TODO: notare che esiste anche MetricFrame in fairlearn, dove uno passa le metriche che vuole siano calcolate sui gruppi

    return fairness_dict


def test_model(model, s, X_test, y_test, predictions_prob=True, custom_sensitives=None):
    from fairlearn.reductions import ExponentiatedGradient
    from fairlearn.postprocessing import ThresholdOptimizer
    debiasing_models = [ThresholdOptimizer, ExponentiatedGradient]

    preds = model.predict(X_test, sensitive_features=X_test[s], random_state=42) if isinstance(model, ThresholdOptimizer) else model.predict(X_test)

    if type(model) not in debiasing_models: preds_prob = model.predict_proba(X_test)
    else: preds_prob = None

    accuracy_metrics = test_accuracy(
        preds,
        y_test,
        **({"predictions_prob": preds_prob[:, 1]} if predictions_prob and type(model) not in debiasing_models else {})
    )
    fairness_metrics = test_fairness(preds, y_test,  X_test[s] if custom_sensitives is None else custom_sensitives)
    return accuracy_metrics, fairness_metrics

def combined_score(acc_metrics, fair_metrics, acc_weight=0.7, fair_weight=0.3):
    accuracy_keys = ['accuracy_score', 'f1_score', 'precision', 'recall']
    fairness_keys = ['DP-ratio', 'EOpp-ratio', 'EOdds-ratio']

    a = [acc_metrics[key] for key in accuracy_keys]
    f = [fair_metrics[key] for key in fairness_keys]

    # a more refined metric? Rugg approved
    score = acc_weight*(sum(a)/len(a)) + fair_weight*(sum(f)/len(f))

    return score

def explain_fav_unfav_mapping(fav_unfav_enc, target_enc, cat=False):
    inv_d = {}
    for key, value in fav_unfav_enc.items():
        for k, v in target_enc.items():
            if int(key) == v:
                if cat:
                    if value == 1: inv_d[k] = "FAV"
                    else: inv_d[k] = "UNFAV"
                else: inv_d[k] = value
    return inv_d

def keys_of_dict(dictionary, keys):
    return {k: dictionary[k] for k in keys if k in dictionary}

def print_metrics_table(metrics):
    print(f"{'Metric':<12} | Value")
    print("-" * 28)

    for k, v in metrics.items():
        print(f"{k:<12} | {float(v):.4f}")
    print()

def roc_plot(acc_dict):
    fpr, tpr, thresholds = acc_dict['roc_objs']
    plt.plot(fpr, tpr, label=f"AUC = {acc_dict['auc']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def mapping_to_string(mapping):
    str = ""
    for k, v in mapping.items():
        str = str + f"{v}: {k}, "
    return str[:-2]