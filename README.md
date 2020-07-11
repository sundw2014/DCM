Knowledge Transfer via Dense Cross-layer Mutual-distillation
=============

This code was used for experiments in Knowledge Transfer via Dense Cross-layer Mutual-distillation (ECCV'2020) by Anbang Yao and Dawei Sun. This code is based on the official pytorch implementation of WRN (https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch).

Knowledge Distillation (KD) based methods adopt the oneway Knowledge Transfer (KT) scheme in which training a lower-capacity student network is guided by a pre-trained high-capacity teacher network. Recently, Deep Mutual Learning (DML) presented a two-way KT strategy, showing that the student network can be also helpful to improve the teacher network. In this paper, we propose Dense Crosslayer Mutual-distillation (DCM), an improved two-way KT method in which the teacher and student networks are trained collaboratively from scratch. To augment knowledge representation learning, well-designed auxiliary classifiers are added to certain hidden layers of both teacher and student networks. To boost KT performance, we introduce dense bidirectional KD operations between the layers appended with classifiers. After training, all auxiliary classifiers are discarded, and thus there are no extra parameters introduced to final models. We test our method on a variety of KT tasks, showing its superiorities over related methods.

bibtex:
```
@inproceedings{Yao2019DKS,
  title={Knowledge Transfer via Dense Cross-layer Mutual-distillation},
  author={Yao, Anbang and Sun, Dawei},
  booktitle={Proceedings of European Conference on Computer Vision},
  year={2020}
}
```

# Usage

### Install requirements

```
pip install -r requirements.txt
```

### Run DCM on CIFAR-100

```
# train wrn-28-10 v.s. wrn-28-10
cd wrn/
mkdir logs
python main.py --save ./logs/wrn-28-10 --depth 28 --width 10 --dataroot [path to the CIFAR dataset]
```

```
# train wrn-28-10 v.s. mobilenet
cd wrn_mobi
mkdir logs
python main.py --save ./logs/wrn-28-10_mobilenet --dataroot [path to the CIFAR dataset]
```
