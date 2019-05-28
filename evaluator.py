import os
import datetime
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('keras_layers')
from sklearn.model_selection import train_test_split

from data_loader import PredictDataGenerator
from evaluation import get_final_metric, calculate_overall_auc, compute_bias_metrics_for_model
from models.bert_base_model import get_bert_multi_model
from utils import get_config, get_bert_config
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


class Evaluator(object):

    def __init__(self, model, model_name, dev_df, weight_path, max_len, batch_size=32):
        self.model = model
        self.model_name = model_name
        self.dev_df = dev_df
        self.weight_path = weight_path
        self.model.load_weights(weight_path)
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


if __name__ == '__main__':
    model_path = 'save_models/bert.weights-uncased-ml-e2.h5'
    text_path = 'tok_text_uncased.pkl'
    df_path = 'data/train.csv'
    max_len = 256
    model_name = 'bert_uncased_base'
    df = pd.read_csv(df_path)
    texts = pickle.load(open(text_path, 'rb'))
    df['tokenized_text'] = texts
    _, test_df = train_test_split(df, test_size=0.055, random_state=59)
    train_config = get_config()
    bert_config = get_bert_config(train_config)
    model = get_bert_multi_model(bert_config)
    eva = Evaluator(model, model_name, test_df, model_path, max_len)
    eva.predict()
    eva.evaluate()
