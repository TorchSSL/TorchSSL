TorchSSL is deprecated and no longer maintained. Please refer **[USB](https://github.com/microsoft/Semi-supervised-learning)**, an upgrade version of TorchSSL. Training in USB only takes **12.5%** of training time of using TorchSSL, and produce better results.

# TorchSSL

<img src="./figures/logo.png"  width = "100" height = "100" align='right' />

A Pytorch-based toolbox for semi-supervised learning. This is also the official implementation for [FlexMatch: boosting semi-supervised learning using curriculum pseudo labeling](https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html) published at NeurIPS 2021. [[arXiv](https://arxiv.org/abs/2110.08263)]  [[Zhihu article](https://zhuanlan.zhihu.com/p/422930830)] [[Video](http://coming)]

## News and Updates

*17/08/2022*
 - TorchSSL (this repo) is no longer maintained and updated. **We have created/updated a more comprehensive codebase and benchmark for Semi-Supervised Learning - [USB](https://github.com/microsoft/Semi-supervised-learning)**. It is built upon TorchSSL but more flexible to use and more extensiable, containing datasets spanning Computer Vision, Natural Language Processing, and Audio Processing. 

*15/02/2021*

- The logs and model weights are shared! We notice that some model weights are missing. We will try to upload the missing model weights in the future.
- The results of BestAcc have been updated! I use single P100 for CIFAR-10 and SVHN, single P40 for STL-10, signle V100-32G for CIFAR-100.

## Introduction

TorchSSL is an all-in-one toolkit based on PyTorch for semi-supervised learning (SSL). Currently, we implmented 9 popular SSL algorithms to enable fair comparison and boost the development of SSL algorithms.

**Supported algorithms:** In addition to fully-supervised (as a baseline), TorchSSL supports the following popular algorithms:

1. PiModel (NeurIPS 2015) [1]
2. MeanTeacher (NeurIPS 2017) [2]
3. PseudoLabel (ICML 2013) [3]
4. VAT (Virtual adversarial training, TPAMI 2018) [4]
5. MixMatch (NeurIPS 2019) [5]
6. UDA (Unsupervised data augmentation, NeurIPS 2020) [6]
7. ReMixMatch (ICLR 2019) [7]
8. FixMatch (NeurIPS 2020) [8]
9. FlexMatch (NeurIPS 2021) [9]

Besides, we implement our Curriculum Pseudo Labeling (CPL) method for Pseudo-Label (Flex-Pseudo-Label) and UDA (Flex-UDA).

**Supported datasets:** TorchSSL currently supports 5 popular datasets in SSL research:

1. CIFAR-10
2. CIFAR-100
3. STL-10
4. SVHN
5. ImageNet

## Main Results

The results are best accuracies with standard errors. In the results, "40", "250", "1000" etc. under the dataset row denote different numbers of labeled samples (e.g., "40" in CIFAR-10 means that there are only 4 labeled samples for each class). We use random seed 0,1,2 for all experiments. All configs are included under the `config/` folder. You can directly cite these results in your own research. 

Note FullySupervised results are from training the model using all training data in the dataset, regardless of the label amount denoted in the table.

### CIFAR-10 and CIFAR-100
|                      |            |  CIFAR-10  |            |   |            | CIFAR100   |            |
|----------------------|------------|------------|------------|---|------------|------------|------------|
|                      | 40         | 250        | 4000       |   | 400        | 2500       | 10000      |
| FullySupervised      | 95.38±0.05 | 95.39±0.04 | 95.38±0.05 |   | 80.7±0.09  | 80.7±0.09  | 80.73±0.05 |
| PiModel [1]          | 25.66±1.76 | 53.76±1.29 | 86.87±0.59 |   | 13.04±0.8  | 41.2±0.66  | 63.35±0.0  |
| PseudoLabel [3]      | 25.39±0.26 | 53.51±2.2  | 84.92±0.19 |   | 12.55±0.85 | 42.26±0.28 | 63.45±0.24 |
| PseudoLabel_Flex [9] | 26.26±1.96 | 53.86±1.81 | 85.25±0.19 |   | 14.28±0.46 | 43.88±0.51 | 64.4±0.15  |
| MeanTeacher [2]      | 29.91±1.6  | 62.54±3.3  | 91.9±0.21  |   | 18.89±1.44 | 54.83±1.06 | 68.25±0.23 |
| VAT [4]              | 25.34±2.12 | 58.97±1.79 | 89.49±0.12 |   | 14.8±1.4   | 53.16±0.79 | 67.86±0.19 |
| MixMatch [5]         | 63.81±6.48 | 86.37±0.59 | 93.34±0.26 |   | 32.41±0.66 | 60.24±0.48 | 72.22±0.29 |
| ReMixMatch [7]       | 90.12±1.03 | 93.7±0.05  | 95.16±0.01 |   | 57.25±1.05 | 73.97±0.35 | 79.98±0.27 |
| UDA [6]              | 89.38±3.75 | 94.84±0.06 | 95.71±0.07 |   | 53.61±1.59 | 72.27±0.21 | 77.51±0.23 |
| UDA_Flex [9]         | 94.56±0.52 | 94.98±0.07 | 95.76±0.06 |   | 54.83±1.88 | 72.92±0.15 | 78.09±0.1  |
| FixMatch [8]         | 92.53±0.28 | 95.14±0.05 | 95.79±0.08 |   | 53.58±0.82 | 71.97±0.16 | 77.8±0.12  |
| FlexMatch [9]        | 95.03±0.06 | 95.02±0.09 | 95.81±0.01 |   | 60.06±1.62 | 73.51±0.2  | 78.1±0.15  |

### STL-10 and SVHN
|                      |            |  STL-10    |            |   |            | SVHN       |            |
|----------------------|------------|------------|------------|---|------------|------------|------------|
|                      | 40         | 250        | 1000       |   | 40         | 250        | 1000       |
| FullySupervised      | None       | None       | None       |   | 97.87±0.02 | 97.87±0.01 | 97.86±0.01 |
| PiModel [1]          | 25.69±0.85 | 44.87±1.5  | 67.22±0.4  |   | 32.52±0.95 | 86.7±1.12  | 92.84±0.11 |
| PseudoLabel [3]      | 25.32±0.99 | 44.55±2.43 | 67.36±0.71 |   | 35.39±5.6  | 84.41±0.95 | 90.6±0.32  |
| PseudoLabel_Flex [9] | 26.58±2.19 | 47.94±2.5  | 67.95±0.37 |   | 36.79±3.64 | 79.58±2.11 | 87.95±0.54 |
| MeanTeacher [2]      | 28.28±1.45 | 43.51±2.75 | 66.1±1.37  |   | 63.91±3.98 | 96.55±0.03 | 96.73±0.05 |
| VAT [4]              | 25.26±0.38 | 43.58±1.97 | 62.05±1.12 |   | 25.25±3.38 | 95.67±0.12 | 95.89±0.2  |
| MixMatch [5]         | 45.07±0.96 | 65.48±0.32 | 78.3±0.68  |   | 69.4±8.39  | 95.44±0.32 | 96.31±0.37 |
| ReMixMatch [7]       | 67.88±6.24 | 87.51±1.28 | 93.26±0.14 |   | 75.96±9.13 | 93.64±0.22 | 94.84±0.31 |
| UDA [6]              | 62.58±8.44 | 90.28±1.15 | 93.36±0.17 |   | 94.88±4.27 | 98.08±0.05 | 98.11±0.01 |
| UDA_Flex [9]         | 70.47±2.1  | 90.97±0.45 | 93.9±0.25  |   | 96.58±1.51 | 97.34±0.83 | 97.98±0.05 |
| FixMatch [8]         | 64.03±4.14 | 90.19±1.04 | 93.75±0.33 |   | 96.19±1.18 | 97.98±0.02 | 98.04±0.03 |
| FlexMatch [9]        | 70.85±4.16 | 91.77±0.39 | 94.23±0.18 |   | 91.81±3.2  | 93.41±2.29 | 93.28±0.3  |

### ImageNet

|                      |100k labels |            |
|----------------------|------------|------------|
|                      | top-1      | top-5      |
| FixMatch [8]         | 56.34      | 78.20      |
| FlexMatch [9]        | 58.15      | 80.52      |

## Logs and weights

You can download the shared logs and weights here.

https://1drv.ms/u/s!AlpW9hcyb0KvmyCfsCjGvhDXG5Nb?e=Xc6amH

## Usage

Before running or modifing the code, you need to:
1. Clone this repo to your machine.
2. Make sure Anaconda or Miniconda is installed.
3. Run `conda env create -f environment.yml` for environment initialization.

### Run the experiments

It is convenient to perform experiment with TorchSSL. For example, if you want to run FlexMatch algorithm:

1. Modify the config file in `config/flexmatch/flexmatch.yaml` as you need
2. Run `python flexmatch.py --c config/flexmatch/flexmatch.yaml`

### Customization

If you want to write your own algorithm, please follow the following steps:

1. Create a directory for your algorithm, e.g., `SSL`, write your own model file `SSl/SSL.py` in it. 
2. Write the training file in `SSL.py`
3. Write the config file in `config/SSL/SSL.yaml`

## Citing TorchSSL

If you think this toolkit or the results are helpful to you and your research, please cite our paper:

```
@article{zhang2021flexmatch},
  title={FlexMatch: Boosting Semi-supervised Learning with Curriculum Pseudo Labeling},
  author={Zhang, Bowen and Wang, Yidong and Hou, Wenxin and Wu, Hao and Wang, Jindong and Okumura, Manabu and Shinozaki, Takahiro},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Maintainers

Yidong Wang<sup>1</sup>, Hao Chen<sup>2</sup>, Yue Fan<sup>3</sup>, Hao Wu<sup>1</sup>, Bowen Zhang<sup>1</sup>, Wenxin Hou<sup>1,4</sup>, Yuhao Chen<sup>5</sup>, Jindong Wang<sup>4</sup>

Tokyo Institute of Technology<sup>1</sup>

Carnegie Mellon University<sup>2</sup>

Max-Planck-Institut für Informatik<sup>3</sup>

Microsoft Research Asia<sup>4</sup>

Megvii<sup>5</sup>

## Contributing

1. You are welcome to open an issue on bugs, questions, and suggestions.
2. If you want to join TorchSSL team, please e-mail Yidong Wang (646842131@qq.com; yidongwang37@gmail.com) for more information. We plan to add more SSL algorithms and expand TorchSSL from CV to NLP and Speech.

## Statements

*For ImageNet datasets:* Please download the ImageNet 2014 dataset (unchanged from 2012) from the official site (link: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
Extract the train and the val set into *subfolders* (the test set is not used), and put them under `train/` and `val/` respectively. Each subfolder will represent a class.
Note: the offical val set is not zipped into subfolders by classes, you may want to use: https://github.com/jiweibo/ImageNet/blob/master/valprep.sh, which is a nice script for preparing the file structure.

## References

[1] Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, and Tapani Raiko.  Semi-supervised learning with ladder networks. InNeurIPS, pages 3546–3554, 2015.

[2] Antti Tarvainen and Harri Valpola.  Mean teachers are better role models:  Weight-averagedconsistency targets improve semi-supervised deep learning results. InNeurIPS, pages 1195–1204, 2017.

[3] Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning methodfor  deep  neural  networks.   InWorkshop  on  challenges  in  representation  learning,  ICML,volume 3, 2013.

[4] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training:a regularization method for supervised and semi-supervised learning.IEEE TPAMI, 41(8):1979–1993, 2018.

[5] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and ColinRaffel. Mixmatch: A holistic approach to semi-supervised learning.NeurIPS, page 5050–5060,2019.

[6] Qizhe Xie, Zihang Dai, Eduard Hovy, Thang Luong, and Quoc Le. Unsupervised data augmen-tation for consistency training.NeurIPS, 33, 2020.

[7] David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang,and Colin Raffel.   Remixmatch:  Semi-supervised learning with distribution matching andaugmentation anchoring. InICLR, 2019.

[8] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raf-fel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li.  Fixmatch:  Simplifying semi-supervised learning with consistency and confidence.NeurIPS, 33, 2020.

[9] Bowen Zhang, Yidong Wang, Wenxin Hou, Hao wu, Jindong Wang, Okumura Manabu, and Shinozaki Takahiro. FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. NeurIPS, 2021.
