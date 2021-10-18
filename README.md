# TorchSSL: A PyTorch-based Toolbox for Semi-Supervised Learning

An all-in-one toolkit based on PyTorch for semi-supervised learning (SSL). We implmented 8 popular SSL algorithms to enable fair comparison and boost the development of SSL algorithms.


## Supported algorithms

We support fully supervised training + 9 popular SSL algorithms as listed below:

1. Pi-Model [1]
2. MeanTeacher [2]
3. Pseudo-Label [3]
4. VAT [4]
5. MixMatch [5]
6. UDA [6]
7. ReMixMatch [7]
8. FixMatch [8]
9. FlexMatch [9]

Besides, we implement our Curriculum Pseudo Labeling (CPL) method for Pseudo-Label (Flex-Pseudo-Label) and UDA (Flex-UDA).

## Supported datasets

We support 5 popular datasets in SSL research as listed below:

1. CIFAR-10
2. CIFAR-100
3. STL-10
4. SVHN
5. ImageNet


## Installation

1. Prepare conda
2. Run `conda env create -f environment.yml`


## Usage

It is convenient to perform experiment with TorchSSL. For example, if you want to perform FlexMatch algorithm:

1. Modify the config file in `config/flexmatch/flexmatch.yaml` as you need
2. Run `python flexmatch --c config/flexmatch/flexmatch.yaml`

## Customization

If you want to write your own algorithm, please follow the following steps:

1. Create a directory for your algorithm, e.g., `SSL`, write your own model file `SSl/SSL.py` in it. 
2. Write the training file in `SSL.py`
3. Write the config file in `config/SSL/SSL.yaml`

## Results
![avatar](./figures/cf10.png)
![avatar](./figures/cf100.png)
![avatar](./figures/stl.png)
![avatar](./figures/svhn.png)

### Citation
If you think this toolkit or the results are helpful to you and your research, please cite our paper:

```
@article{zhang2021flexmatch},
  title={FlexMatch: Boosting Semi-supervised Learning with Curriculum Pseudo Labeling},
  author={Zhang, Bowen and Wang, Yidong and Hou Wenxin and Wu, Hao and Wang, Jindong and Okumura, Manabu and Shinozaki, Takahiro},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

### Maintainer
Yidong Wang<sup>1</sup>, Hao Wu<sup>2</sup>, Bowen Zhang<sup>1</sup>, Wenxin Hou<sup>1,3</sup>, Jindong Wang<sup>3</sup>

Shinozaki Lab<sup>1</sup> http://www.ts.ip.titech.ac.jp/

Okumura Lab<sup>2</sup> http://lr-www.pi.titech.ac.jp/wp/

Microsoft Research Asia<sup>3</sup>

### References

[1] 