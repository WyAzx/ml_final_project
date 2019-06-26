from sklearn import metrics
import numpy as np
import pandas as pd

TOXICITY_COLUMN = 'target'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]

# From baseline kernel


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN] > 0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup] > 0.5]
    return compute_auc((subgroup_examples[label] > 0.5), subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup] > 0.5) & (df[label] <= 0.5)]
    non_subgroup_positive_examples = df[(df[subgroup] <= 0.5) & (df[label] > 0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label] > 0.5, examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup] > 0.5) & (df[label] > 0.5)]
    non_subgroup_negative_examples = df[(df[subgroup] <= 0.5) & (df[label] <= 0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label] > 0.5, examples[model_name])


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {'subgroup': subgroup, 'subgroup_size': len(dataset[dataset[subgroup] > 0.5]),
                  SUBGROUP_AUC: compute_subgroup_auc(dataset, subgroup, label_col, model),
                  BPSN_AUC: compute_bpsn_auc(dataset, subgroup, label_col, model),
                  BNSP_AUC: compute_bnsp_auc(dataset, subgroup, label_col, model)}
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def get_metric_e2e(dev_df, result):
    dev_df['result'] = result
    overall_auc = calculate_overall_auc(dev_df, 'result')
    bias_auc_df = compute_bias_metrics_for_model(dev_df, IDENTITY_COLUMNS, 'result', 'target')
    subgroup_auc = power_mean(bias_auc_df[SUBGROUP_AUC], -5),
    pbsn_auc = power_mean(bias_auc_df[BPSN_AUC], -5),
    bnsp_auc = power_mean(bias_auc_df[BNSP_AUC], -5)
    final_auc = get_final_metric(bias_auc_df, overall_auc)
    return final_auc, overall_auc, subgroup_auc, pbsn_auc, bnsp_auc
