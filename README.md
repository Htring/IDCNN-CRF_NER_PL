# 背景
其中主要使用pytorch_lightning来组织模型的训练，使用torchtext以及pytorch_lighting对语料处理，使用seqeval来评估序列标注的结果，使用pytorch-crf来实现CRF层。

本程序使用的Python程序包，主要如下：

- python 3.7
- pytorch 1.10,
- pytorch_lightning 1.15
- pytorch-crf 0.7.2
- torchtext 0.11.0
- seqeval 1.2.2

关于本程序的讲解可参考我的博客：[【NLP】基于Pytorch的IDCNN-CRF命名实体识别(NER)实现](https://blog.csdn.net/meiqi0538/article/details/124644060?spm=1001.2014.3001.5501)

## 数据集

本程序数据来源于：https://github.com/luopeixiang/named_entity_recognition.

为了能够使用seqeval工具评估模型效果，将原始数据中“M-”、“E-”开头的标签处理为“I-”

## 模型效果

```text
Testing: 100%|██████████| 4/4 [00:04<00:00,  1.06s/it]
               precision    recall  f1-score   support

        CONT       1.00      1.00      1.00        28
         EDU       0.97      0.96      0.96       112
         LOC       0.80      0.67      0.73         6
        NAME       0.98      0.98      0.98       112
         ORG       0.89      0.92      0.90       553
         PRO       0.76      0.79      0.78        33
        RACE       1.00      1.00      1.00        14
       TITLE       0.92      0.93      0.93       772

   micro avg       0.92      0.93      0.92      1630
   macro avg       0.92      0.91      0.91      1630
weighted avg       0.92      0.93      0.92      1630

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'val_f1': 0.9232643118148598}
--------------------------------------------------------------------------------
Testing: 100%|██████████| 4/4 [00:08<00:00,  2.18s/it]

Process finished with exit code 0

```

**f1值达到0.923，在BiLSTM-CRF中的效果是0.928。可以看出效果很接近。**

## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)
