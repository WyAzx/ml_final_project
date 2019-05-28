import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class Logger(Callback):
    def __init__(self, out_path='save_models/', patience=10, lr_patience=3, model=None, val_text=None, val_label=None):
        self.auc = 0
        self.path = out_path
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0
        self.train_model = model
        self.val_text = val_text
        self.val_label = val_label
        super().__init__()

    def on_epoch_end(self, epoch, log=None):
        cv_pred = self.train_model.predict([self.val_text, np.zeros_like(self.val_text)], batch_size=64)
        cv_true = self.val_label
        auc_val = roc_auc_score(cv_true, cv_pred)
        if self.auc < auc_val:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(auc_val, 4)))
            self.auc = auc_val
            self.train_model.save(self.path + 'weights.{}-{:.4f}.h5'.format(epoch, self.auc), overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(K.get_value(self.model.optimizer.lr))
                K.set_value(self.model.optimizer.lr, 0.75 * lr)
                print("Setting lr to {}".format(0.75 * lr))
                self.no_improve_lr = 0
        return
