# ELMo Approach

Main idea (why ELMo?)

In the competition description, it said that

```txt
When the Conversation AI team first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.
```

Thus, hopefully the embedding way of ELMo will solve this problem.

## Getting Started

```sh
# Install the dependencies
pip install allennlp
```

## Approaches

### ELMo + Scikit Learn Regression

## Links

* [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

### ELMo

* [AllenNLP ELMo](https://allennlp.org/elmo)
  * [Tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
    * [Using ELMo interactively](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-interactively)
  * [Training models with tensorflow](https://github.com/allenai/bilm-tf)
  * [AllenNLP](http://www.allennlp.org/)
    * [github - allenai/allennlp](https://github.com/allenai/allennlp) - An open-source NLP research library, built on PyTorch
