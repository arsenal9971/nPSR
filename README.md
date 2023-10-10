# NeuralPoisson

### [**Paper**](https://arxiv.org/abs/2308.01766) | [**Project Page**](https://github.com/arsenal9971/PoissonNet/)

![](./media/figure_benchmark.png)

This repository contains the implementation of the paper:

NeuralPoisson: Resolution-Agnostic Neural Shape Reconstruction

We are currently working on a cleaned-up version of this code that includes more documentation and pre-trained weights. If you find our code or paper useful, please consider citing
```bibtex
@article{anonymous,
  title={NeuralPoisson: Resolution-Agnostic Neural Shape Reconstruction},
  author={Anonymous},
  journal={arXiv preprint},
  year={2023}
}
```


## Installation

You need to first install all the dependencies. For that you can use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `poissonnet` using
```
conda env create -f environment.yaml
conda activate poissonnet
```

## Training - Quick Start

First, run the script to get the demo data:

```bash
bash scripts/download_data.sh
```
