import numpy as np


def tokenize_examples(examples, tokenizer, max_len):
    max_len -= 2
    all_tokens = []
    longer = 0
    for example in examples:
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


def seq_padding(X):
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
                    X = seq_padding(X)
                    X = np.array(X)
                    Y = np.array(Y)
                    segment = np.zeros_like(X)
                    yield [X, segment], [Y]
                    X, Y = [], []
