import numpy as np
from tqdm import tqdm


def tokenize_examples(examples, tokenizer, max_len):
    max_len -= 2
    all_tokens = []
    longer = 0
    for example in tqdm(examples):
        tokens = tokenizer.tokenize(example)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        all_tokens.append(one_token)
    print('TRUNCATE EXAMPLES:', longer)
    return all_tokens


def tokenize_example(example, tokenizer, max_len):
    max_len -= 2
    longer = 0
    tokens = tokenizer.tokenize(example)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        longer += 1
    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
    return one_token


def truncate_ids(text_ids, max_len):
    if len(text_ids) > max_len:
        new_text_ids = text_ids[: max_len-1] + [text_ids[-1]]
        return new_text_ids
    return text_ids


def seq_padding(X, max_len=512):
    X = [truncate_ids(x, max_len) for x in X]
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


class DataGenerator(object):
    def __init__(self, x, y, tokenizer, batch_size=32, max_len=512):
        # self.x = tokenize_examples(x, tokenizer, max_len)
        self.x = x
        self.y = y
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.steps = len(self.x) // self.batch_size
        if len(self.x) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.x)))
            np.random.shuffle(idxs)
            X, Y = [], []
            for i in idxs:
                x_ = tokenize_example(self.x[i], self.tokenizer, self.max_len)
                X.append(x_)
                Y.append(self.y[i])
                if len(X) == self.batch_size or i == idxs[-1]:
                    X = seq_padding(X, self.max_len)
                    X = np.array(X)
                    Y = np.array(Y)
                    segment = np.zeros_like(X)
                    yield [X, segment], [Y]
                    X, Y = [], []


class AllDataGenerator(object):
    def __init__(self, text, y, y_aux, example_weights, batch_size=32, max_len=512):
        # self.x = tokenize_examples(x, tokenizer, max_len)
        self.text = text
        self.y = y
        self.y_aux = y_aux
        self.example_weight = example_weights
        self.batch_size = batch_size
        self.steps = len(self.text) // self.batch_size
        self.max_len = max_len
        if len(self.text) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.text)))
            np.random.shuffle(idxs)
            X, Y, Y_aux, EW = [], [], [], []
            for i in idxs:
                X.append(self.text[i])
                Y.append(self.y[i])
                Y_aux.append(self.y_aux[i])
                EW.append(self.example_weight[i])
                if len(X) == self.batch_size or i == idxs[-1]:
                    X = seq_padding(X, self.max_len)
                    X = np.array(X)
                    Y = np.array(Y)
                    Y_aux = np.array(Y_aux)
                    EW = np.array(EW)
                    segment = np.zeros_like(X)
                    yield [X, segment], [Y, Y_aux], [EW, np.ones_like(EW)]
                    X, Y, Y_aux, EW = [], [], [], []


class PredictDataGenerator(object):
    def __init__(self, text, batch_size=32, max_len=512):
        # self.x = tokenize_examples(x, tokenizer, max_len)
        self.text = text
        self.batch_size = batch_size
        self.steps = len(self.text) // self.batch_size
        self.max_len = max_len
        if len(self.text) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X = []
            for i in range(len(self.text)):
                X.append(self.text[i])
                if len(X) == self.batch_size or i == len(self.text) - 1:
                    X = seq_padding(X, self.max_len)
                    X = np.array(X)
                    segment = np.zeros_like(X)
                    yield [X, segment]
                    X = []
