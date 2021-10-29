## How To Use This Code
You will need:

- [PyTorch](https://PyTorch.org/), version 1.6.0
- tqdm, numpy, scipy, and h5py

Before running SSGAN-LA on Tiny-ImageNet, you need to manually download the Tiny-ImageNet dataset and store it into [data](./) folder with the following structure.

```
tiny_imagenet
    train
        cls0
            img0
            img1
            ...
        cls1
            img0
            ...
        ...
    val
        cls0
            img0
            ...
        cls1
            img0
            ...
        ...
```

Before first running methods on a dataset named `DATASET={C10,S10,TINY}`, please run the following command to prepare the statistics for calculating FID.

```shell
python calculate_inception_moments.py --dataset DATASET --data_root data
```

- run SSGAN-LA on CIFAR-10

```sh
sh scripts/launch_cifar10_ema.sh
```

- run SSGAN-LA on STL-10

```sh
sh scripts/launch_stl10_ema.sh
```

- run SSGAN-LA on Tiny-ImageNet

```sh
sh scripts/launch_tiny_ema.sh
```

## Acknowledgements

This code is developed based on [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).
