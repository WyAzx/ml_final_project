import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from evaluation import get_metric_e2e


class Logger(Callback):
    def __init__(self, model, model_name, val_gen, val_df, out_path='save_models/', patience=10, lr_patience=3):
        self.auc = 0
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
        cv_pred = self.model.predict_generator(self.val_gen.__iter__(), len(self.val_gen))[0]
        final_auc, overall_auc, subgroup_auc, pbsn_auc, bnsp_auc = get_metric_e2e(self.val_df, cv_pred)
        print('Final AUC:{}\nOverall AUC:{}\nSub AUC:{}\nBPSN AUC:{}\nBNSP AUC:{}\n'.format(final_auc, overall_auc,
                                                                                            subgroup_auc, pbsn_auc,
                                                                                            bnsp_auc))
        if self.auc < final_auc:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(final_auc, 4)))
            self.auc = final_auc
            self.train_model.save_weights(self.path + 'weights.{}.h5'.format(self.model_name))
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
