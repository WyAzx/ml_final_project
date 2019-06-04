import sys

sys.path.append("..")
import pickle

from keras.models import load_model

from bpe.bpe_textcnn import DataLoader, get_text_cnn


def save_model_weights(model, path):
    """  传入完整的model对象；weights
    :param model: Keras model object
    :param path  weight存储路径
    :return:
    """
    weights = {}
    for layer in model.layers:
        layer_name = layer.name
        if layer_name.startswith("embedding_"):
            continue
        weights[layer_name] = layer.get_weights()
    with open(path, "wb") as f:
        # pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(weights, f)
    return weights


def load_model_weights(model, path):
    """ keras model
    :param model: keras model
    :param path weights存储路径
    :return:
    """
    with open(path, 'rb') as f:
        weights = pickle.load(f)
    for layer in model.layers:
        layer_name = layer.name
        if layer_name.startswith("embedding_"):
            continue
        weights[layer_name] = layer.set_weights(weights[layer_name])
    return model


def get_model():
    vocab_size = 100 * 1000
    embedding_dim = 100
    max_sequence_len = 512
    num_classes = 1
    data_loader = DataLoader(vocab_size, embedding_dim, max_sequence_len)
    model = get_text_cnn(vocab_size, max_sequence_len, embedding_dim,
                         num_classes,
                         embedding_matrix=data_loader.bpemb_en_100k.vectors)  # 使用bpe
    return model


def demo():
    model_1 = load_model("text_cnn.hdf5")
    weights_path = "model_weights.pickle"
    save_model_weights(model_1, weights_path)
    model_2 = get_model()
    load_model_weights(model_2, weights_path)

    print(model_1.layers[-1].get_weights())
    print(model_2.layers[-1].get_weights())


if __name__ == "__main__":
    demo()
