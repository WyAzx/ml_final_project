from queue import Queue
from random import shuffle
from threading import Thread
import time
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
        new_text_ids = text_ids[: max_len - 1] + [text_ids[-1]]
        return new_text_ids
    return text_ids


def seq_padding(X, max_len=512, truncate=True):
    if truncate:
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


class GeneralPredictGenerator(object):
    def __init__(self, text, batch_size=512, max_len=512):
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
                    X = seq_padding(X, self.max_len, False)
                    X = np.array(X)
                    yield [X]
                    X = []


class Batcher(object):
    def __init__(self, inputs, outputs, sample_weights, batch_size):
        self.inputs = inputs
        self.out_puts = outputs
        self.sample_weight = sample_weights
        self.batch_size = batch_size

        self._batch_queue = Queue(200)
        self._example_queue = Queue(maxsize=len(self.inputs[0]))

        self.reset_examples_queue()

        self._num_batch_q_threads = 4

        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        self._watch_thread = Thread(target=self.watch_threads)
        self._watch_thread.daemon = True
        self._watch_thread.start()

    def fill_batch_queue(self):

        while True:
            inputs = []
            for _ in range(self.batch_size * 50):
                idx = self._example_queue.get()
                exp_input = [inp[idx] for inp in self.inputs]
                exp_output = [out[idx] for out in self.out_puts]
                exp_sw = [sw[idx] for sw in self.sample_weight] if self.sample_weight is not None else None
                exp = (exp_input, exp_output, exp_sw)
                inputs.append(exp)
            inputs = sorted(inputs, key=lambda inp: len(inp[0][0]))  # sort by length of encoder sequence

            # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
            batches = []
            for i in range(0, len(inputs), self.batch_size):
                exps = inputs[i:i + self.batch_size]
                batch_inputs, batch_outputs, batch_sample_weight = zip(*exps)
                batch_inputs = [seq_padding(inp, truncate=False) for inp in zip(*batch_inputs)]
                batch_inputs = [np.array(inp) for inp in batch_inputs]
                batch_outputs = zip(*batch_outputs)
                batch_outputs = [np.array(out) for out in batch_outputs]
                batch_sample_weight = zip(*batch_sample_weight) if batch_sample_weight is not None else None
                batch_sample_weight = [np.array(sw) for sw in
                                       batch_sample_weight] if batch_sample_weight is not None else None
                batches.append([batch_inputs, batch_outputs, batch_sample_weight])
            shuffle(batches)
            for b in batches:  # each b is a list of Example objects
                self._batch_queue.put(b)

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def reset_examples_queue(self):
        idxs = list(range(len(self.inputs[0])))
        np.random.shuffle(idxs)
        for idx in idxs:
            self._example_queue.put(idx)

    def next_batch(self):
        if self._batch_queue.qsize() == 0:
            print(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

        if self._example_queue.qsize() == 0:
            self.reset_examples_queue()

        batch = self._batch_queue.get()  # get the next Batch
        return batch


class GeneralDataGenerator(object):
    def __init__(self, inputs, outputs, sample_weights, batch_size):
        self.inputs = inputs
        self.out_puts = outputs
        self.sample_weight = sample_weights
        self.batch_size = batch_size
        self.batcher = Batcher(inputs, outputs, sample_weights, batch_size)
        self.steps = len(self.inputs[0]) // self.batch_size
        if len(self.inputs[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            batch = self.batcher.next_batch()
            yield batch


class SeqDataGenerator(object):
    def __init__(self, text, y, y_aux, example_weights, batch_size=512, max_len=512):
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
                    X = seq_padding(X, self.max_len, False)
                    X = np.array(X)
                    Y = np.array(Y)
                    Y_aux = np.array(Y_aux)
                    EW = np.array(EW)
                    yield [X], [Y, Y_aux], [EW, np.ones_like(EW)]
                    X, Y, Y_aux, EW = [], [], [], []