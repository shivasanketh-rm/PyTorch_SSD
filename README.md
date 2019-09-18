# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).


### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#acknowledgement'>Acknowledgement</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: Currently only support Python 3+.
- Then download the dataset by following the [instructions](#datasets) below.
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/), and aim to add [ImageNet](http://www.image-net.org/) support soon.


## Datasets
##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  

## Demos
To demo on an image 
```Shell
python demo.py --weights <path to weights file> --input_frame <path to imput image> 
```

<img align="center" src= "https://github.com/shivasanketh-rm/PyTorch_SSD/blob/master/demo/detections_a.png">

## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |


#### Download a pre-trained network
- Currently, use any of the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325)
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    2. If using [pip](https://pypi.python.org/pypi/pip):

```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

## Acknowledgement 
- Most of the base code was cloned from the repository - https://github.com/amdegroot/ssd.pytorch 

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](http://www.webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo):
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow)
