## TorchSSL: A PyTorch-based Toolbox for Semi-Supervised Learning

An all-in-one toolkit based on PyTorch for semi-supervised learning (SSL). We implmented 8 popular SSL algorithms to enable fair comparison and boost the development of SSL algorithms.


### Implemented Algorithms & Datasets

We support fully supervised training + 8 popular SSL algorithms as listed below:

1. Pi-Model
2. MeanTeacher
3. Pseudo-Label
4. VAT
5. MixMatch
6. UDA
7. ReMixMatch
8. FixMatch

Besides, we implement our Curriculum Pseudo Labeling (CPL) method for Pseudo-Label (Flex-Pseudo-Label), FixMatch (FlexMatch), and UDA (Flex-UDA).

We support 5 popular datasets in SSL research as listed below:

1. CIFAR-10
2. CIFAR-100
3. STL-10
4. SVHN
5. ImageNet


### Installation

1. Prepare conda
2. Run `conda env create -f environment.yml`


### Usage

It is convenient to perform experiment with TorchSSL. For example, if you want to perform FlexMatch algorithm:

1. Modify the config file in `config/flexmatch/flexmatch.yaml` as you need
2. Run `python flexmatch --c config/flexmatch/flexmatch.yaml`

### Customization

If you want to write your own algorithm, please follow the following steps:

1. Create a directory for your algorithm, e.g., `SSL`, write your own model file `SSl/SSL.py` in it. 
2. Write the training file in `SSL.py`
3. Write the config file in `config/SSL/SSL.yaml`

### Results
| Dataset      | CIFAR-10   |       |       | CIFAR-100 |      |       | STL-10 |     |      | SVHN |     |
|--------------|----|----|----|-----------|------|-------|--------|-----|------|------|-----| 
| Label Amount | 40 | 250|4000|400	|2500	|10000	|40	|250	|1000	|40	|1000| 
|PL	|69.51±4.55	|41.02±3.56	|13.15±1.84	|86.10±1.50	|58.00±0.38	|36.48±0.13	|74.48±1.48	|55.63±5.38	|31.80±0.29	|60.32±2.46	|9.56±0.25|
|Flex-PL	|**65.41±1.35**	|**36.37±1.57**	|**10.82±0.04**	|**74.85±1.53**	|**44.15±0.19**	|**29.13±0.26**	|**69.26±0.60**	|**41.28±0.46**	|**24.63±0.14**	|**36.90±1.19**	|**8.64±0.08**|
|UDA	|7.33±2.03	|5.11±0.07	|4.20±0.12	|44.99±2.28	|27.59±0.24	|22.09±0.19	|37.31±3.03	|12.07±1.50	|6.65±0.25	|4.40±2.31	|**1.93±0.01**|
|Flex-UDA	|**5.33±0.13**	|**5.05±0.02**	|**4.07±0.06**	|**33.64±0.92**	|**24.34±0.20**	|**20.07±0.13**	|**12.84±2.60**	|**8.05±0.21**	|**5.77±0.08**	|**3.78±1.67**	|1.97±0.06|
|FixMatch	|6.78±0.50	|4.95±0.07	|4.09±0.02	|46.76±0.79	|28.15±0.81	|22.47±0.66	|35.42±6.43	|10.49±1.03	|6.20±0.20	|**4.36±2.16**	|**1.97±0.03**|
|FlexMatch	|**4.99±0.16**	|**4.80±0.06**	|**3.95±0.03**	|**32.44±1.99**	|**23.85±0.23**	|**19.92±0.06**	|**10.87±1.15**	|**7.71±0.14**	|**5.56±0.22**	|5.36±2.38	|2.86±0.91|
|Fully-Supervised|	4.45±0.12|	|	|	|19.07±0.18	|	|	|	|	|	|2.14±0.02	|

### References

[1] "FlexMatch: Boosting Semi-Supervised Learningwith Curriculum Pseudo Labeling"

### Maintainer
Yidong Wang<sup>1</sup>, Hao Wu<sup>2</sup>, Bowen Zhang<sup>1</sup>, Wenxin Hou<sup>1,3</sup>, Jindong Wang<sup>3</sup>

Shinozaki Lab<sup>1</sup> http://www.ts.ip.titech.ac.jp/

Okumura Lab<sup>2</sup> http://lr-www.pi.titech.ac.jp/wp/

Microsoft Research Asia<sup>3</sup>


### Citation
If you think this toolkit or the results are helpful to you and your research, please cite [1]!

```
