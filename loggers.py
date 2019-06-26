import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, f1_score
from evaluation import get_metric_e2e
from bpe.keras_weight_tool import save_model_weights
from evaluator import JigsawEvaluator


class KFoldLogger(Callback):
    def __init__(self, model_name, val_gen, val_true, val_iden, out_path='save_models/', patience=2, lr_patience=1):
        self.auc = 0
        self.auc_list = []
        self.path = out_path
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0
        self.model_name = model_name
        self.val_gen = val_gen
        self.evaluator = JigsawEvaluator(val_true, val_iden)
        self.pred = None
        super().__init__()

    def on_epoch_end(self, epoch, log=None):
        if 'elmo' in self.model_name:
            save_model_weights(self.model, self.path + 'weights.{}.{}.pkl'.format(epoch, self.model_name))
        cv_pred = self.model.predict_generator(self.val_gen.__iter__(), len(self.val_gen))[0].flatten()
        final_auc, overall_auc, sub_auc, bpsn_auc, bnsp_auc, bias_metrics = self.evaluator.get_final_metric(cv_pred)
        print('Final AUC:{}\nOverall AUC:{}\nSub AUC:{}\nBPSN AUC:{}\nBNSP AUC:{}\n'.format(final_auc, overall_auc,
                                                                                            sub_auc, bpsn_auc,
                                                                                            bnsp_auc))
        print('Detail Bias:\n', bias_metrics)
        if self.auc < final_auc:
            self.pred = cv_pred
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(final_auc, 4)))
            self.auc = final_auc
            self.auc_list = [overall_auc, sub_auc, bpsn_auc, bnsp_auc]
            # self.model.save_weights(self.path + 'weights.{}.h5'.format(self.model_name))
            save_model_weights(self.model, self.path + 'weights.{}-{}.{}.pkl'.format(epoch, np.round(self.auc * 100, 3),
                                                                                     self.model_name))
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(final_auc, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(K.get_value(self.model.optimizer.lr))
                K.set_value(self.model.optimizer.lr, 0.75 * lr)
                print("Setting lr to {}".format(0.75 * lr))
                self.no_improve_lr = 0
        return

    def on_train_end(self, logs=None):
        with open('Result.csv', 'a', encoding='utf8') as f:
            f.write('{},{},{},{},{},{}\n'.format(self.model_name, self.auc, *self.auc_list))


class Logger(Callback):
    def __init__(self, model, model_name, val_gen, val_df, out_path='save_models/', patience=10, lr_patience=3):
        self.auc = 0
        self.auc_list = []
        self.path = out_path
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0
        self.train_model = model
        self.model_name = model_name
        self.val_gen = val_gen
        self.val_df = val_df
        super().__init__()

    def on_epoch_end(self, epoch, log=None):
        cv_pred = self.model.predict_generator(self.val_gen.__iter__(), len(self.val_gen))[0].flatten()
        final_auc, overall_auc, subgroup_auc, pbsn_auc, bnsp_auc = get_metric_e2e(self.val_df, cv_pred)
        print('Final AUC:{}\nOverall AUC:{}\nSub AUC:{}\nBPSN AUC:{}\nBNSP AUC:{}\n'.format(final_auc, overall_auc,
                                                                                            subgroup_auc, pbsn_auc,
                                                                                            bnsp_auc))
        if self.auc < final_auc:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(final_auc, 4)))
            self.auc = final_auc
            self.auc_list = [overall_auc, subgroup_auc, pbsn_auc, bnsp_auc]
            # self.train_model.save_weights(self.path + 'weights.{}.h5'.format(self.model_name))
            save_model_weights(self.train_model, self.path + 'weights.{}.pkl'.format(self.model_name))
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(final_auc, 4)))
            # if self.no_improve >= self.patience:
            #     self.model.stop_training = True
            # if self.no_improve_lr >= self.lr_patience:
            #     lr = float(K.get_value(self.model.optimizer.lr))
            #     K.set_value(self.model.optimizer.lr, 0.75 * lr)
            #     print("Setting lr to {}".format(0.75 * lr))
            #     self.no_improve_lr = 0
        return

    def on_train_end(self, logs=None):
        with open('Result.csv', 'a', encoding='utf8') as f:
            f.write('{},{},{},{},{},{}\n'.format(self.model_name, self.auc, *self.auc_list))


class BaseLogger(Callback):
    def __init__(self, val_gen, val_labels, out_path='save_models/weight.iden_lstm_new.h5', patience=10, lr_patience=3):
        super().__init__()
        self.f1 = 0
        self.path = out_path
        self.val_gen = val_gen
        self.val_labels = val_labels
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0

    def on_epoch_end(self, epoch, logs=None):
        cv_pred = self.model.predict_generator(self.val_gen.__iter__(), len(self.val_gen))
        pred_list = list(zip(*cv_pred))
        cv_true = self.val_labels
        true_list = list(zip(*cv_true))
        f1_list = []
        for i in range(len(pred_list)):
            f1 = f1_score(true_list[i], [int(x >= 0.5) for x in pred_list[i]])
            f1_list.append(f1)
        f1 = np.average(f1_list)
        print(f1_list)
        if self.f1 < f1:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(f1, 4)))
            self.f1 = f1
            self.model.save(self.path, overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(f1, 4)))

        return
