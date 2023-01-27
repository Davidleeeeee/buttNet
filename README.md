# buttNet with PyTorch



Official implementation of the [buttNet](https://arxiv.org/abs/2301.10584) in PyTorch to detect negative samples only use positive samples to train.

## Quick start

### install

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the training set:
```bash
Link：https://pan.baidu.com/s/1-u6dBLCDPyRE3ocm6YZFSA 
Code：uygq
```


## Usage
**Note : Use Python 3.6 or newer**


### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  --images              positive images path to train
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.

### Train example

```console
python train.py --images D:\data\positive_train
```

### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.


```console
> python predict.py -h
usage: predict.py [-h] [--load FILE] --input INPUT [INPUT ...]

Predict masks from input images

optional arguments:
  --load FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        path of input images
```
### Prediction example
```console
predict.py --input D:\data\predict --load D:\buttNet\checkpoints\ok4_7Patches_epoch7.pth
```

## Pretrained model
A [pretrained model](https://github.com/Davidleeeeee/buttNet/releases/download/v1/ok4_7Patches_epoch7.pth) is available to test LED component

### Description
This model was trained only with 5k positive images and It's also very easy to train with your own single GPU(gtx1060) with your own data
