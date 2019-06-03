import re
from collections import Counter, defaultdict

from bpemb import BPEmb


def get_stats(vocab):
    """
    :param vocab:
    :return:
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        # print(symbols)
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab_in):
    vocab_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in vocab_in:
        w_out = p.sub("".join(pair), word)
        vocab_out[w_out] = vocab_in[word]
    return vocab_out


def bpe(text, num_merges):
    # 空格不编码，加入特殊后缀，以便恢复
    vocab = {" ".join(word) + " _": count for word, count in Counter(text.split(" ")).most_common() if word}
    print(vocab)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        # print(pairs)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)
        print(vocab)
        print('-' * 10)


def test():
    bpemb_en = BPEmb(lang="en", dim=100)
    s = "Stratford"
    res1 = bpemb_en.encode(s)
    res2 = bpemb_en.encode_ids(s)
    print(res1)
    print(res2)

    bpemb_en_100k = BPEmb(lang="en", vs=100000, dim=100)  # 40 M；词表越大切分越少
    s = "hello world !"
    bpemb_en_100k.encode_ids(s)
    res1 = bpemb_en_100k.encode(s)
    res2 = bpemb_en_100k.encode_ids(s)
    print(res1)
    print(res2)


if __name__ == "__main__":
    # text = "hello world !"
    # num_merges = 10
    # bpe(text, num_merges)
    test()
