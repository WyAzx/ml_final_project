
### 原理
> BPE，（byte pair encoder）字节对编码，也可以叫做digram coding双字母组合编码，主要目的是为了数据压缩，算法描述为字符串里频率最常见的一对字符被一个没有在这个字符中出现的字符代替的层层迭代过程。

- [《Neural Machine Translation of Rare Words with Sub - 琪琪怪怪的文章 - 知乎](https://zhuanlan.zhihu.com/p/38574684)
- [一分钟搞懂的算法之BPE算法](https://zhuanlan.zhihu.com/p/38130825)

### 实践   
>就是一种新的编码方式，
只需要使用bpe预训练编码bpemb_en_100k.encode_ids(s)代替我们的自己的字典,
embedding初始化我们的embedding就可以了  

```python
# !pip install bpemb 
# https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs100000.d100.w2v.bin.tar.gz
from pathlib import Path
from bpemb import BPEmb
vocab_size=100000
embedding_dim=100
bpemb_en_100k = BPEmb(lang="en", vs=vocab_size, dim=embedding_dim,
cache_dir=Path.home() / Path(".cache/bpemb"))  # 40 M；词表越大切分越少


s="hello world !"
s_ids=bpemb_en_100k.encode_ids(s)
[34542, 501, 54039]
embedding_matrix=bpemb_en_100k.vectors

```

- [github:Pre-trained subword embeddings Byte-Pair Encoding (BPE)](https://github.com/bheinzerling/bpemb)
- [kaggle: example usage: pre-trained BPE embeddings](https://www.kaggle.com/lefant/example-usage-pre-trained-bpe-embeddings)


> 




