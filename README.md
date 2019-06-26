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
    - VFAE 

- *Tricks*
    - multi-task 
    - [sample_weights](https://www.kaggle.com/thousandvoices/simple-lstm)

## Preprocessed Data

- Managed with `git-lfs` (X): can not upload new objects to public fork
- Dealing with OOV and data imbalanced problem

## 项目需求
detect toxic comments ― and minimize unintended model bias

根据评论数据文本，判断是否为 toxicity 
               （ toxicity is defined as anything rude, disrespectful or otherwise 				likely to make someone leave a discussion. ）
，并减小模型的unintended bias，使模型输出结果更加`公正`


## 模型建立
模型搭建位于`models`文件夹下，`bert`模型位于`keras_layers/keras_bert`
- Word2v + bpe + 2xbiLSTM
- ELMo + 2xbiLSTM
- Word2v + DGCNN
- bert fine-tune

## bias 优化
- Sample Weight
	- Subgroup及Subgroup负样本Loss加权

- Custom object function
	- Rank Loss、Focal Loss

- Data Augmentation
    - 扩充Subgroup sample
    - 平衡Subgroup 正负样本比例

## 实验结果

model | final CV
:-: | :-:
bert | 0.939
bert+sample weight | 0.941
bert+sample weight+label identity | 0.943
bilstm+sample weight+5 fold | 0.938
bilstm+sample weight+aug | 0.940
dgcnn+sample weight+5 fold | 0.937
elmo+bilstm+sample weight | 0.938


## 训练
在trainer当中选择训练模型, (TODO: 添加命令行参数)
```python
nohup python -u trainer.py > train.log 2>&1 &
```
