# Self-Supervised GANs with Label Augmentation

This is a PyTorch implementation of [Self-Supervised GANs with Label Augmentation](https://arxiv.org/abs/2106.08601).

<div align=center><img height=300 width=600 src=/imgs/model.png /></div>

## Usage

The folder [BigGAN-PyTorch](/BigGAN-PyTorch) includes the code of the main experiments (on CIFAR-10, STL-10, and Tiny-ImageNet) in the paper, and the folder [MoG](/MoG) includes the jupyter of the one-dimensional synthetic experiment.

![results](/imgs/1d.png)

## Citation

Please cite our paper if you find the code useful for your research.

```
@inproceedings{NEURIPS2021_6cb5da35,
 author = {Hou, Liang and Shen, Huawei and Cao, Qi and Cheng, Xueqi},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {13019--13031},
 publisher = {Curran Associates, Inc.},
 title = {Self-Supervised GANs with Label Augmentation},
 url = {https://proceedings.neurips.cc/paper/2021/file/6cb5da3513bd26085ee3fad631ebb37a-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
