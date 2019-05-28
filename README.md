# Machine Learning Final Project

Kaggle Competition: [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

## TODO

一些优化方向思路，仅供参考:D

- EDA

- *Preprocess*

    可参考之前[比赛的](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557#latest-533843)思路，解决OOV问题
    - BPE
    - TTA

- *Model*
    - Sequence model: bilstm, HAN...
    - Bert fine tune
    
- *Loss*
    - Focal loss (根据identity决定alpha)

- *Metrics*
    - ~~实现[比赛AUC指标评测方法](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation)~~

- *Argument*
    - Adversarial Training
    - 训练identity分类器(40W examples), 对jigsaw上个比赛数据进行分类,抽取identity负样本
 

- *Tricks*
    - multi-task 
    - [sample_weights](https://www.kaggle.com/thousandvoices/simple-lstm)

## Preprocessed Data

- Managed with `git-lfs` (X): can not upload new objects to public fork
- Dealing with OOV and data imbalanced problem
