import json
import os
from collections import namedtuple

Config = namedtuple('Config', ['BERT_DIR', 'DATA_DIR'])
BertConfig = namedtuple('BertConfig', ['config', 'check_point', 'vocab'])


def get_config() -> Config:
    config = Config(**json.load(open('config.json')))
    return config


def get_bert_config(config: Config) -> BertConfig:
    bert_dir = config.BERT_DIR
    bert_config = BertConfig(**{
        'config': os.path.join(bert_dir, 'bert_config.json'),
        'check_point': os.path.join(bert_dir, 'bert_model.ckpt'),
        'vocab': os.path.join(bert_dir, 'vocab.txt')
    })
    return bert_config
