import os
import datetime
import pandas as pd
import numpy as np
import pickle
import sys
# from keras.engine.saving import load_model

# sys.path.append('keras_layers')
from sklearn.model_selection import train_test_split

from data_loader import PredictDataGenerator
from evaluation import get_final_metric, calculate_overall_auc, compute_bias_metrics_for_model
# from models.bert_base_model import get_bert_multi_model
# from utils import get_config, get_bert_config
from sklearn.metrics import roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


class Evaluator(object):

    def __init__(self, model, model_name, dev_df, weight_path, max_len, batch_size=32):
        self.model = model
        self.model_name = model_name
        self.dev_df = dev_df
        # self.weight_path = weight_path
        # self.model.load_weights(weight_path)
        self.result = None
        self.overall_auc = None
        self.final_auc = None
        self.bias_auc_df = None
        self.dev_gen = PredictDataGenerator(dev_df['tokenized_text'].values, batch_size, max_len)

    def predict(self):
        result = self.model.predict_generator(self.dev_gen.__iter__(), len(self.dev_gen))
        self.result = result
        return result

    def evaluate(self, save_path='result/'):
        if self.result is None:
            print('RUN PREDICT FIRST')
            return
        result = self.result[0]
        self.dev_df[self.model_name] = result
        self.overall_auc = calculate_overall_auc(self.dev_df, self.model_name)

        print('OVERALL AUC: ', self.overall_auc)

        self.bias_auc_df = compute_bias_metrics_for_model(self.dev_df, IDENTITY_COLUMNS, self.model_name, 'target')

        print('BIAS AUC:\n', self.bias_auc_df)

        self.final_auc = get_final_metric(self.bias_auc_df, self.overall_auc)

        print('FINAL AUC: ', self.final_auc)

        save_dir = os.path.join(save_path, self.model_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if os.path.isfile(save_dir):
            save_dir += '_' + datetime.datetime.now().strftime("%m%d%H")

        with open(save_dir, 'w', encoding='utf8') as f:
            f.write('OVERALL AUC: {}\n'.format(self.overall_auc))
            f.write('BIAS AUC:\n{}\n\n'.format(self.bias_auc_df.to_string()))
            f.write('FINAL AUC: {}\n'.format(self.final_auc))


class JigsawEvaluator(object):

    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        sub_auc = self._power_mean(bias_metrics[0])
        bpsn_auc = self._power_mean(bias_metrics[1])
        bnsp_auc = self._power_mean(bias_metrics[2])
        bias_score = np.average([sub_auc, bpsn_auc, bnsp_auc])
        overall_score = self._calculate_overall_auc(y_pred)
        final_auc = (1 - self.overall_model_weight) * bias_score + self.overall_model_weight * overall_score
        return final_auc, overall_score, sub_auc, bpsn_auc, bnsp_auc, bias_metrics


if __name__ == '__main__':
    model_path = 'save_models/bert.weights-uncased-decay-weight.h5'
    text_path = 'tok_text_uncased.pkl'
    df_path = 'data/train.csv'
    max_len = 512
    df = pd.read_csv(df_path)
    texts = pickle.load(open(text_path, 'rb'))
    df['tokenized_text'] = texts
    _, test_df = train_test_split(df, test_size=0.055, random_state=59)
    # train_config = get_config()
    # bert_config = get_bert_config(train_config)
    # model = get_bert_multi_model(bert_config)
    # model.load_weights(model_path)
    idens = test_df[IDENTITY_COLUMNS].values
    label = test_df['target'].values
    text = test_df['tokenized_text'].values

    # val_gen = PredictDataGenerator(text, 32, max_len)
    # res = model.predict_generator(val_gen.__iter__(), len(val_gen))[0].flatten()
    import json

    def sigmoid(x, derivative=False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm
    res = json.load(open('predict_jigsaw.logits.json', 'r'))
    res = np.array([sigmoid(r[0]) for r in res])
    print(res[:10])
    eva = JigsawEvaluator(label, idens)
    final_auc, overall_auc, sub_auc, bpsn_auc, bnsp_auc, bias_metrics = eva.get_final_metric(res)
    print('Final AUC:{}\nOverall AUC:{}\nSub AUC:{}\nBPSN AUC:{}\nBNSP AUC:{}\n'.format(final_auc, overall_auc,
                                                                                        sub_auc, bpsn_auc,
                                                                                        bnsp_auc))
    print('Detail Bias:\n', bias_metrics)

    np.save('results/xlnet_res.npy', res)
